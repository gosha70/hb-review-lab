#!/usr/bin/env python3
"""Step 1c: teacher labels. Claude reads each change (prediction view plus outcome view) and
answers the questions in questions.json for the whole change and for each change unit.

Runs through the Message Batches API (asynchronous, half price). Labels are appended to
--out with label_source "inferred"; nothing here is human-validated.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python label_changes.py --allow-external anthropic --limit 20   # trial
    python label_changes.py --allow-external anthropic

Re-running resumes: finished cases are skipped and a submitted batch is picked up again.
The diffs and review comments are sent to the Anthropic API, so the run must be confirmed with
--allow-external anthropic, and only repositories listed under "anthropic" in
external_allowlist.json are sent. scrape_changes.py redacts common secret formats first; that is a
safety net, not a guarantee. Every case sent is logged to <out>.sent.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from common import (MAX_UNITS, UNIT_CHARS, allowed_repos, append_jsonl, build_units, load_questions,
                    options, questions_sha, read_jsonl)

MODEL = "claude-opus-5-5"
BATCH_SIZE = 1000

SYSTEM = """You label code changes for a review-routing classifier.

For the change as a whole, and separately for each numbered unit (one piece of the diff), answer
every question below with exactly one of its options.

Label the change as it was submitted: the risk it carried and the review it needed at that
moment. The "Outcome" section is hindsight (review comments, CI, reverts). Use it as evidence of
what the change actually needed, but do not label the later fixes.

The classifier sees one unit at a time, each under the same change header (shown below), so it
knows what else the change touches. Label a unit for its own part in the change, read with that
context: raise it when the unit is risky because of what the rest of the change does, but do not
raise it merely because another unit is risky. The change-level label for a question is never
lower than the highest label among its units.

Questions:
{rubric}"""


def rubric(questions: dict) -> str:
    lines = []
    for qid, q in questions.items():
        lines.append(f"\n{qid}: {q['instructions']}")
        crit = q["criteria"]
        items = enumerate(crit) if q["type"] == "score" else crit.items()
        lines += [f"  {k} = {v}" for k, v in items]
    return "\n".join(lines)


def schema(questions: dict) -> dict:
    props = {qid: {"type": "string", "enum": [str(o) for o in options(q)]} for qid, q in questions.items()}

    def obj(extra):
        p = {**extra, **props}
        return {"type": "object", "properties": p, "required": list(p), "additionalProperties": False}

    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "change": obj({}),
            "units": {"type": "array", "items": obj({"unit": {"type": "string"}})},
        },
        "required": ["rationale", "change", "units"],
        "additionalProperties": False,
    }


def prompt(case: dict, units: list[dict], incomplete: list[str]) -> str:
    o = case["outcome"]
    parts = [
        f"Repository: {case['repo']}", f"Change: {case['title']}",
        f"Description:\n{case.get('body') or '(none)'}", "",
        "Files:", *[f"  {f['path']} [{f['status']}] +{f['additions']} -{f['deletions']}"
                    + ("" if f.get("patch") else " (no text diff)") for f in case["files"]],
    ]
    if incomplete:
        parts.append(f"Parts of this change are not shown ({', '.join(incomplete)}).")
    parts += ["", "Outcome (hindsight):", f"  signals: {json.dumps(o['observed'])}"]
    parts += [f"  review {r['state']} by {'human' if r['human'] else 'bot/author'}: {r['body'] or ''}"
              for r in o["reviews"]]
    parts += [f"  comment on {c['path']}:{c['line']} by {'human' if c['human'] else 'bot/author'}: {c['body']}"
              for c in o["review_comments"]]
    if units:
        parts += ["", "=== change header the classifier sees on every unit ===", units[0]["header"].rstrip()]
    for u in units:
        parts += ["", f"=== unit {u['unit']} ===", u["body"]]
    parts += ["", f"Label the change and each of the {len(units)} units."]
    return "\n".join(parts)


def params(case: dict, questions: dict, model: str, unit_chars: int, max_units: int) -> dict:
    units, incomplete = build_units(case, unit_chars, max_units)
    return {
        "model": model,
        "max_tokens": 16000,
        "thinking": {"type": "adaptive"},
        "system": SYSTEM.format(rubric=rubric(questions)),
        "messages": [{"role": "user", "content": prompt(case, units, incomplete)}],
        "output_config": {"format": {"type": "json_schema", "schema": schema(questions)}},
    }


def parse(text: str, case: dict, questions: dict, unit_chars: int, max_units: int) -> dict:
    """Validate the teacher's JSON against the case's units. Raises ValueError when unusable."""
    data = json.loads(text)
    units, incomplete = build_units(case, unit_chars, max_units)

    def conv(d):
        out = {}
        for qid, q in questions.items():
            v = d[qid]
            v = int(v) if q["type"] == "score" else v
            if v not in options(q):
                raise ValueError(f"{qid}: {v!r} is not an option")
            out[qid] = v
        return out

    by_unit = {u["unit"]: conv(u) for u in data["units"]}
    missing = [u["unit"] for u in units if u["unit"] not in by_unit]
    if missing:
        raise ValueError(f"no labels for units {missing}")
    return {
        "change": conv(data["change"]),
        "units": {u["unit"]: by_unit[u["unit"]] for u in units},
        "rationale": data["rationale"],
        "units_complete": not incomplete,
        "incomplete_reasons": incomplete,
    }


def cid(case_id: str) -> str:
    return hashlib.sha1(case_id.encode()).hexdigest()[:40]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default="data/cases.jsonl")
    ap.add_argument("--out", default="data/labels.jsonl")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=0, help="label at most N new cases (0 = all)")
    ap.add_argument("--unit-chars", type=int, default=UNIT_CHARS)
    ap.add_argument("--max-units", type=int, default=MAX_UNITS)
    ap.add_argument("--allow-external", choices=["anthropic"],
                    help="required: confirms these diffs and review comments may be sent to Anthropic")
    ap.add_argument("--allowlist", default="external_allowlist.json",
                    help="JSON file naming the repositories whose code may be sent to each provider")
    args = ap.parse_args(argv)
    if args.allow_external != "anthropic":
        sys.exit("This step sends source diffs and review comments to the Anthropic API.\n"
                 "Re-run with --allow-external anthropic to confirm. Only repositories listed under "
                 "'anthropic' in the allowlist file are sent.")
    allowed = allowed_repos("anthropic", args.allowlist)

    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    questions = load_questions(args.questions)
    cases = {c["case_id"]: c for c in read_jsonl(args.cases)}
    by_cid = {cid(k): k for k in cases}
    pending_path = Path(args.out).with_suffix(".pending.json")
    errors_path = Path(args.out).with_suffix(".errors.jsonl")
    sent_path = Path(args.out).with_suffix(".sent.jsonl")

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    def collect(batch_id: str) -> None:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            log(f"  batch {batch_id}: {batch.request_counts.processing} still processing")
            time.sleep(60)
        ok = bad = tin = tout = 0
        for result in client.messages.batches.results(batch_id):
            case_id = by_cid.get(result.custom_id)
            if case_id is None:
                continue
            try:
                if result.result.type != "succeeded":
                    raise ValueError(f"request {result.result.type}")
                msg = result.result.message
                tin, tout = tin + msg.usage.input_tokens, tout + msg.usage.output_tokens
                if msg.stop_reason != "end_turn":
                    raise ValueError(f"stop_reason {msg.stop_reason}")
                text = next(b.text for b in msg.content if b.type == "text")
                labels = parse(text, cases[case_id], questions, args.unit_chars, args.max_units)
            except (ValueError, KeyError, StopIteration) as e:
                append_jsonl(errors_path, {"case_id": case_id, "error": str(e)})
                bad += 1
                continue
            append_jsonl(args.out, {
                "case_id": case_id, "label_source": "inferred", "teacher": args.model,
                "questions_sha": questions_sha(questions),
                "unit_chars": args.unit_chars, "max_units": args.max_units, **labels,
            })
            ok += 1
        log(f"  batch {batch_id}: {ok} labelled, {bad} failed, {tin} input / {tout} output tokens")

    if pending_path.exists():
        for batch_id in json.loads(pending_path.read_text()):
            log(f"Resuming batch {batch_id}")
            collect(batch_id)
        pending_path.unlink()

    done = {r["case_id"] for r in read_jsonl(args.out)}
    blocked = sorted({c["repo"] for c in cases.values()} - allowed)
    if blocked:
        log(f"Not on the anthropic allowlist, will not be sent: {', '.join(blocked)}")
    todo = [c for k, c in cases.items() if k not in done and c["repo"] in allowed]
    no_text = [c for c in todo if not build_units(c, args.unit_chars, args.max_units)[0]]
    if no_text:
        log(f"{len(no_text)} changes have no readable diff; not sent. They stay in the dataset as incomplete.")
    todo = [c for c in todo if c not in no_text]
    if args.limit:
        todo = todo[:args.limit]
    log(f"{len(done)} cases already labelled, {len(todo)} to label with {args.model}")

    for start in range(0, len(todo), BATCH_SIZE):
        chunk = todo[start:start + BATCH_SIZE]
        batch = client.messages.batches.create(requests=[
            Request(custom_id=cid(c["case_id"]),
                    params=MessageCreateParamsNonStreaming(
                        **params(c, questions, args.model, args.unit_chars, args.max_units)))
            for c in chunk])
        pending_path.write_text(json.dumps([batch.id]))
        for c in chunk:   # audit trail of what left this machine
            append_jsonl(sent_path, {"provider": "anthropic", "batch": batch.id, "case_id": c["case_id"],
                                     "repo": c["repo"], "sent_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        log(f"Submitted batch {batch.id} ({len(chunk)} cases)")
        collect(batch.id)
        pending_path.unlink()
    if errors_path.exists():
        log(f"Failures are listed in {errors_path}; re-run to retry them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
