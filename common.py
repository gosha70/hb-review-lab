"""Shared helpers: JSONL I/O, secret redaction, question schema, change units."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNIT_CHARS = 1800   # diff characters per unit; sized for a 1,024-token Laya window
MAX_UNITS = 40      # units per change; larger changes are flagged incomplete


# ---------- JSONL ----------

def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, row) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------- secret redaction (applied before anything is stored or sent out) ----------

_SECRETS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
]
_ASSIGNED = re.compile(
    r"(?i)((?:api[_-]?key|secret|passwd|password|token|private[_-]?key)\w*\s*[:=]\s*)"
    r"(['\"])([^'\"\s]{8,})\2")


def redact(text: str | None) -> str | None:
    if not text:
        return text
    for rx in _SECRETS:
        text = rx.sub("[REDACTED]", text)
    return _ASSIGNED.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]{m.group(2)}", text)


# ---------- questions ----------

def load_questions(path=None) -> dict:
    return json.loads(Path(path or ROOT / "questions.json").read_text(encoding="utf-8"))


def options(q: dict) -> list:
    """Answer options in ascending order of severity. Score levels are ints, choices are labels."""
    c = q["criteria"]
    return list(range(len(c))) if q["type"] == "score" else list(c)


def level(q: dict, value) -> int:
    """Position of an answer on the question's low-to-high scale."""
    if q["type"] == "score":
        return max(0, min(len(q["criteria"]) - 1, int(round(float(value)))))
    return options(q).index(value)


# ---------- change units ----------

def _chunks(patch: str, limit: int) -> list[str]:
    """Split a file patch into pieces of at most `limit` characters, on hunk then line boundaries."""
    out, cur = [], ""
    for hunk in re.split(r"(?m)^(?=@@ )", patch):
        if not hunk:
            continue
        if len(cur) + len(hunk) <= limit:
            cur += hunk
            continue
        if cur:
            out.append(cur)
            cur = ""
        if len(hunk) <= limit:
            cur = hunk
            continue
        for line in hunk.splitlines(keepends=True):
            # A line longer than a unit is spread over several units; no text is dropped.
            for piece in (line[i:i + limit] for i in range(0, len(line), limit)):
                if cur and len(cur) + len(piece) > limit:
                    out.append(cur)
                    cur = ""
                cur += piece
    if cur:
        out.append(cur)
    return out


def build_units(case: dict, unit_chars: int = UNIT_CHARS, max_units: int = MAX_UNITS):
    """One classifier input per diff piece. Returns (units, incomplete_reasons).

    The units cover the whole change only when incomplete_reasons is empty. Reasons:
      unit_cap           the change needs more than `max_units` units; only the first are returned
      file_without_diff  a file has line changes but GitHub returned no text diff (too large)
      unreadable_file    a file changed with no readable diff at all (binary, or empty)
      files_not_fetched  the revision that was retrieved has more files than were fetched
    Callers must treat an incomplete change as not fully classified, never as safe.
    """
    files = case["files"]
    header = (
        f"Repository: {case['repo']}\n"
        f"Change: {case['title']}\n"
        f"Description: {(case.get('body') or '').strip()[:300]}\n"
        f"Files changed: {len(files)} "
        f"(+{sum(f['additions'] for f in files)} -{sum(f['deletions'] for f in files)})\n")
    units, reasons = [], []
    if case.get("files_complete") is False:
        reasons.append("files_not_fetched")
    for f in files:
        if not f.get("patch"):
            if f["additions"] or f["deletions"]:
                reasons.append("file_without_diff")
            elif f["status"] != "renamed":      # a pure rename changes no content
                reasons.append("unreadable_file")
            continue
        parts = _chunks(f["patch"], unit_chars)
        for i, part in enumerate(parts, 1):
            units.append({
                "unit": f"u{len(units) + 1}",
                "path": f["path"],
                "state": f"{header}File: {f['path']} [{f['status']}] part {i}/{len(parts)}\nDiff:\n{part}",
            })
    if len(units) > max_units:
        reasons.append("unit_cap")
    return units[:max_units], sorted(set(reasons))


def unread_files(case: dict) -> list[str]:
    """Paths whose content change could not be read (no text diff)."""
    return [f["path"] for f in case["files"]
            if not f.get("patch") and (f["additions"] or f["deletions"] or f["status"] != "renamed")]


# ---------- external-provider allowlist ----------

def allowed_repos(provider: str, path) -> set[str]:
    """Repositories whose code may be sent to `provider`, from the allowlist file.

    The file maps a provider name to the exact owner/name repositories it may receive:
        {"anthropic": ["me/app", "django/django"], "typesafe": ["django/django"]}
    There are no wildcards. A repository that is not listed is never sent.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f"No external-provider allowlist at {p}.\n"
            f"Create it (see external_allowlist.example.json) listing the repositories whose code "
            f"may be sent to {provider}.")
    data = json.loads(p.read_text(encoding="utf-8"))
    repos = data.get(provider, [])
    if not isinstance(repos, list) or any(not isinstance(r, str) or "*" in r or "/" not in r for r in repos):
        raise SystemExit(f"{p}: '{provider}' must be a list of exact owner/name repositories (no wildcards).")
    return set(repos)


# ---------- authorship ----------

_AI_MARKERS = {
    "claude": r"co-authored-by:\s*claude|generated with \[?claude code|noreply@anthropic\.com|claude\.ai/code",
    "copilot": r"co-authored-by:\s*copilot|copilot-swe-agent|copilot@users\.noreply",
    "codex": r"co-authored-by:\s*codex|chatgpt\.com/codex|openai codex|chatgpt-codex-connector",
    "cursor": r"co-authored-by:\s*cursor|cursoragent",
    "aider": r"co-authored-by:\s*aider|\(aider\)|^aider:",
    "devin": r"devin-ai-integration|co-authored-by:\s*devin",
    "gemini": r"co-authored-by:\s*gemini|gemini-cli|google-labs-jules",
}
_AI_MARKERS = {k: re.compile(v, re.I | re.M) for k, v in _AI_MARKERS.items()}


def written_by(*texts: str | None) -> str:
    """The AI coding agent named in commit trailers, PR text or author login, else "unmarked".

    "unmarked" means no marker was found. It does not prove a human wrote the change.
    """
    blob = "\n".join(t for t in texts if t)
    for agent, rx in _AI_MARKERS.items():
        if rx.search(blob):
            return agent
    return "unmarked"


def attribute(commits: list[tuple[str, str]] | None, *other: tuple[str, str | None]) -> tuple[str, list[dict]]:
    """Who wrote one revision of a change. Returns (label, evidence).

    `commits` are the (sha, message) pairs that make up the revision being classified, or None
    when they could not be determined. `other` are (source, text) pairs that describe the whole
    revision, such as the author login.

      <agent>   every commit carries that agent's marker, or the revision as a whole does
      mixed     more than one agent, or some commits marked and others not
      unmarked  no marker anywhere (this does not prove a human wrote it)
      unknown   the revision's commits could not be determined
    """
    evidence = []
    for source, text in other:
        agent = written_by(text)
        if agent != "unmarked":
            evidence.append({"source": source, "agent": agent})
    whole = {e["agent"] for e in evidence}
    if commits is None:
        return "unknown", evidence
    per_commit = [written_by(msg) for _, msg in commits]
    evidence += [{"source": f"commit:{sha[:7]}", "agent": a}
                 for (sha, _), a in zip(commits, per_commit) if a != "unmarked"]
    agents = whole | {a for a in per_commit if a != "unmarked"}
    if not agents:
        return "unmarked", evidence
    if len(agents) > 1 or (not whole and "unmarked" in per_commit):
        return "mixed", evidence
    return agents.pop(), evidence
