"""Shared helpers: JSONL I/O, secret redaction, question schema, change units."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNIT_CHARS = 2250    # characters per unit state (header + diff piece); sized for a 1,024-token Laya window
HEADER_CHARS = 950   # cap on the change header that every unit of a change carries
MIN_DIFF_CHARS = 200
MAX_UNITS = 60       # units per change; larger changes are flagged incomplete
MAX_FILES_LISTED = 8

# Heuristic path strata, used for sampling and in the change header. These are NOT risk labels.
SENSITIVE = {
    "security": re.compile(
        r"auth|login|session|token|secret|crypt|passw|permission|acl|oauth|jwt|saml|sso|"
        r"cors|csrf|sanitiz|security|credential|\.env",
        re.I,
    ),
    "data_migration": re.compile(r"migrations?/|\.sql$|schema|alembic|flyway|liquibase", re.I),
    "infra_ci": re.compile(
        r"dockerfile|docker-compose|\.github/workflows|terraform|\.tf$|k8s|kubernetes|helm|"
        r"deploy|nginx|\.gitlab-ci",
        re.I,
    ),
    "dependencies": re.compile(
        r"(^|/)(package(-lock)?\.json|yarn\.lock|pnpm-lock\.yaml|requirements[^/]*\.txt|"
        r"pyproject\.toml|poetry\.lock|uv\.lock|go\.(mod|sum)|cargo\.(toml|lock)|pom\.xml|"
        r"build\.gradle(\.kts)?|gemfile(\.lock)?)$",
        re.I,
    ),
    "ui": re.compile(r"\.(tsx|jsx|vue|svelte|css|scss|html)$", re.I),
}

_TEST = re.compile(
    r"(^|/)(tests?|__tests__|specs?|testing)/|(^|/)test_[^/]*\.py$|_test\.(go|py|rb|rs|cc|cpp)$|"
    r"\.(test|spec)\.[cm]?[jt]sx?$|Tests?\.(java|kt|cs|scala)$", re.I)
_DOCS = re.compile(r"\.(md|mdx|rst|adoc|txt)$|(^|/)docs?/", re.I)


def file_kind(path: str) -> str:
    return "test" if _TEST.search(path) else "docs" if _DOCS.search(path) else "source"


def file_strata(path: str) -> list[str]:
    return [k for k, rx in SENSITIVE.items() if rx.search(path)]


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


def _short(path: str, limit: int = 60) -> str:
    if len(path) <= limit:
        return path
    return ("…/" + "/".join(path.split("/")[-2:]))[-limit:]


def change_header(case: dict) -> str:
    """The context every unit of a change carries: what the whole change touches.

    Built from the prediction view only (title, description and files of the revision being
    classified). Path strata are taken per file here, never from case["strata"], which the audit
    computes from the final pull request and may include files added after the review.
    """
    files = case["files"]
    lines = lambda f: f["additions"] + f["deletions"]
    status = Counter({"added": "added", "removed": "deleted", "renamed": "renamed"}.get(f["status"], "modified")
                     for f in files)
    kinds = {k: [f for f in files if file_kind(f["path"]) == k] for k in ("source", "test", "docs")}
    per_kind = "; ".join(
        f"{k} {len(v)} files +{sum(f['additions'] for f in v)}/-{sum(f['deletions'] for f in v)}"
        for k, v in kinds.items() if v)
    src_lines = sum(lines(f) for f in kinds["source"])
    ratio = f"{sum(lines(f) for f in kinds['test']) / src_lines:.2f}" if src_lines else "none"
    strata = Counter(s for f in files for s in file_strata(f["path"]))
    dirs = Counter(f["path"].split("/")[0] if "/" in f["path"] else "." for f in files)
    unread = len(unread_files(case))
    # The repository name is left out on purpose: it would let the classifier learn a prior per
    # repository instead of reading the change.
    head = (
        f"Change: {' '.join((case['title'] or '').split())[:150]}\n"
        f"Description: {' '.join((case.get('body') or '').split())[:300]}\n"
        f"Summary: {len(files)} files ({status['added']} added, {status['deleted']} deleted, "
        f"{status['renamed']} renamed, {status['modified']} modified); {per_kind or 'no files'}; "
        f"test lines per source line {ratio}; "
        f"strata: {', '.join(f'{k} {n}' for k, n in sorted(strata.items())) or 'none'}; "
        f"directories: {', '.join(d for d, _ in dirs.most_common(5))}; "
        f"{unread} files without readable diff\n")[:HEADER_CHARS - 1]
    if not head.endswith("\n"):
        head += "\n"
    # Files in a stratum first, then source, then the rest; each group by lines changed.
    ranked = sorted(files, key=lambda f: (0 if file_strata(f["path"]) else 1 if file_kind(f["path"]) == "source" else 2,
                                          -lines(f), f["path"]))
    rows = []
    for f in ranked[:MAX_FILES_LISTED]:
        tags = file_strata(f["path"])
        rows.append(f"  {_short(f['path'])} {file_kind(f['path'])} {f['status']} +{f['additions']}/-{f['deletions']}"
                    + (f" [{','.join(tags)}]" if tags else "") + "\n")

    def render(n: int) -> str:
        more = f"  ({len(files) - n} more files)\n" if len(files) > n else ""
        return head + "Files:\n" + "".join(rows[:n]) + more

    n = len(rows)
    while n and len(render(n)) > HEADER_CHARS:
        n -= 1
    return render(n) if len(render(n)) <= HEADER_CHARS else head


def build_units(case: dict, unit_chars: int = UNIT_CHARS, max_units: int = MAX_UNITS):
    """One classifier input per diff piece. Returns (units, incomplete_reasons).

    Every unit's state is the change header followed by one piece of one file's diff, and is at
    most `unit_chars` characters, so the header's length comes out of the room for the diff.
    Each unit also carries `body` (its state without the header).

    The units cover the whole change only when incomplete_reasons is empty. Reasons:
      unit_cap           the change needs more than `max_units` units; only the first are returned
      file_without_diff  a file has line changes but GitHub returned no text diff (too large)
      unreadable_file    a file changed with no readable diff at all (binary, or empty)
      files_not_fetched  the revision that was retrieved has more files than were fetched
    Callers must treat an incomplete change as not fully classified, never as safe.
    """
    files = case["files"]
    header = change_header(case)
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
        label = f"File: {f['path']} [{f['status']}]"
        room = unit_chars - len(header) - len(f"{label} part 999/999\nDiff:\n")
        parts = _chunks(f["patch"], max(MIN_DIFF_CHARS, room))
        for i, part in enumerate(parts, 1):
            body = f"{label} part {i}/{len(parts)}\nDiff:\n{part}"
            units.append({"unit": f"u{len(units) + 1}", "path": f["path"], "header": header,
                          "body": body, "state": header + body})
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

      <agent>   every commit carries that agent's marker, or no commit is marked and the
                revision as a whole (login, description) is
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
    # Commit-level evidence decides whether a revision is mixed. A marker in the description or
    # login is supplementary: it cannot vouch for commits that carry no marker of their own.
    partly_marked = "unmarked" in per_commit and any(a != "unmarked" for a in per_commit)
    if len(agents) > 1 or partly_marked:
        return "mixed", evidence
    return agents.pop(), evidence


def questions_sha(questions: dict) -> str:
    """Fingerprint of a question set. Labels made under one set are not valid under another."""
    import hashlib
    return hashlib.sha256(json.dumps(questions, sort_keys=False, ensure_ascii=False).encode()).hexdigest()[:16]


def state_sha(state: str) -> str:
    """Fingerprint of one unit's input, stored with its label so later drift is detected."""
    import hashlib
    return hashlib.sha256(state.encode("utf-8")).hexdigest()[:16]


# ---------- review-loop text ----------
# Agent-written descriptions often record how review went ("DeepSeek: PASS, notes only",
# "second and third review rounds"). That is hindsight, so such lines are kept out of the
# prediction view.
#
# The patterns are deliberately narrow. Some repositories are about review tooling, and plain
# words such as "reviewer", "round" or "finding" describe the change itself there; "12 rounds"
# of a hash is security evidence. So a round counts only as a counted *review* round, and a
# reviewer counts only when a result is attached to it.

_COUNT = r"(?:first|second|third|fourth|fifth|one|two|three|four|five|six|\d+)"
_NAMED = r"(?:deepseek|codex)"
# Verdicts are matched in capitals only, as review tools print them. Lower case is ordinary
# prose: "Codex: failed requests now retry with backoff" describes a change.
_VERDICT = r"(?-i:PASS|FAIL|INCONCLUSIVE)"
_REVIEW_OUTCOME = re.compile("|".join([
    # counted review rounds: "second and third review rounds", "four review rounds", "3 rounds of review"
    rf"\b{_COUNT}(?:\s+and\s+{_COUNT})?\s+(?:peer[- ])?review\s+rounds?\b",
    rf"\b{_COUNT}\s+rounds?\s+of\s+(?:peer[- ])?review\b",
    r"\breview\s+round\s+\d",
    # a named reviewer with a verdict: "DeepSeek: PASS", "DeepSeek on the build: PASS"
    rf"\b{_NAMED}\b[^.\n]{{0,60}}?:\s*{_VERDICT}\b",
    # the verdict before the reviewer: "reviewed to PASS", "PASS by the gating reviewer (DeepSeek)"
    rf"\breviewed\s+to\s+{_VERDICT}\b",
    rf"\b{_VERDICT}\b[^.\n]{{0,40}}?\bby\s+(?:the\s+gating\s+reviewer|{_NAMED})\b",
    # a review line with a verdict: "Review: PASS after 2 rounds."
    rf"^\s*(?:[-*]\s*)?(?:peer[- ])?review\s*:\s*{_VERDICT}\b",
    # a named reviewer with its rounds: "DeepSeek, two rounds on the spec" (not "round-trip")
    rf"\b{_NAMED}\b[^.\n]{{0,30}}?\brounds?\b(?!-trip)",
    # a named reviewer and what it turned up: "Codex reviewed it and flagged two issues"
    rf"\b{_NAMED}\s+reviewed\b[^.\n]{{0,40}}?\b(?:flagged|found|raised)\b",
    # a review with a tally: "Codex review: 2 findings", "peer review: no findings"
    rf"\b(?:{_NAMED}|peer)[- ]review\b[^.\n]{{0,20}}?:\s*(?:\d+|no)\s+(?:findings?|warnings?|notes?|blocking)",
    # "Reviewer pass at 3599cca: five findings, all taken"
    r"\breviewer\s+pass\b[^.\n]{0,80}?\bfindings?\b",
    # an ordinal review with what it did: "the second review found"
    r"\b(?:first|second|third|fourth|re)[- ]review\s+(?:found|passed|failed|flagged|raised)\b",
    # a bare verdict: "verdict: PASS", "PASS, notes only"
    rf"\bverdict\s*:\s*{_VERDICT}\b",
    rf"\b{_VERDICT}, notes only\b",
]), re.I | re.M)
# Broader than the stripper on purpose: anything that might be a review outcome. Used only to
# warn, so a person reads the examples; it would remove too much if used to strip.
_REVIEW_HINT = re.compile(
    rf"\bdeepseek\b|\bcodex\b|\b{_VERDICT}\b|\breview(?:ed)?[- ]rounds?\b|\brounds?\s+of\s+review\b|"
    r"\bgating\s+reviewer\b|\breviewed\s+to\b", re.I)


def review_hint(text: str | None) -> bool:
    """True when text mentions a reviewer tool, a verdict or a review round at all."""
    return bool(_REVIEW_HINT.search(text or ""))


_REVIEW_HEADING = re.compile(r"^(#{1,6})\s*(?:peer[- ])?review[- ](?:rounds?|loop|results?|verdicts?|history)\b", re.I)
_HEADING = re.compile(r"^(#{1,6})\s")


def strip_review_loop(text: str | None) -> tuple[str, list[str]]:
    """(text without review-outcome lines, the lines removed).

    Removes lines that report a counted review round or a reviewer's result, and whole sections
    under a heading such as "## Review rounds". Lines that merely mention a reviewer, a round or
    a finding are kept: they may describe the change.
    """
    kept, removed = [], []
    section = 0     # heading level of a review section being dropped; 0 when not in one
    for line in (text or "").splitlines():
        h = _HEADING.match(line)
        if section and h and len(h.group(1)) <= section:
            section = 0
        m = _REVIEW_HEADING.match(line)
        if m:
            section = len(m.group(1))
        if section or _REVIEW_OUTCOME.search(line):
            if line.strip():
                removed.append(line.strip())
        else:
            kept.append(line)
    return "\n".join(kept).strip(), removed
