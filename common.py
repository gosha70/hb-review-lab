"""Shared helpers: JSONL I/O, secret redaction, question schema, change units."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNIT_CHARS = 1800   # diff characters per unit; sized for a 1,024-token Laya window
MAX_UNITS = 40      # units per change; larger changes are flagged units_complete=False


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
            if len(line) > limit:
                line = line[:limit - 20] + " [line cut]\n"
            if len(cur) + len(line) > limit:
                out.append(cur)
                cur = ""
            cur += line
    if cur:
        out.append(cur)
    return out


def build_units(case: dict, unit_chars: int = UNIT_CHARS, max_units: int = MAX_UNITS):
    """One classifier input per diff piece. Returns (units, complete).

    A change is never silently truncated: when it needs more than `max_units` units, the first
    `max_units` are returned with complete=False so callers can treat it as not fully classified.
    """
    files = case["files"]
    header = (
        f"Repository: {case['repo']}\n"
        f"Change: {case['title']}\n"
        f"Description: {(case.get('body') or '').strip()[:300]}\n"
        f"Files changed: {len(files)} "
        f"(+{sum(f['additions'] for f in files)} -{sum(f['deletions'] for f in files)})\n")
    units = []
    for f in files:
        if not f.get("patch"):
            continue
        parts = _chunks(f["patch"], unit_chars)
        for i, part in enumerate(parts, 1):
            units.append({
                "unit": f"u{len(units) + 1}",
                "path": f["path"],
                "state": f"{header}File: {f['path']} [{f['status']}] part {i}/{len(parts)}\nDiff:\n{part}",
            })
    return units[:max_units], len(units) <= max_units
