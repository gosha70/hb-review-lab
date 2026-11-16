#!/usr/bin/env python3
"""Second teacher: label a subset of changes with an OpenAI model, as a potential-bias screen.

An LLM judge can favour output from its own model family. The first teacher is Claude, so the
changes attributed to an AI agent (and a group of unmarked ones) are labelled again by a model
from another provider, with the same rubric, prompt and answer schema.

This is a teacher-disagreement screen. It cannot establish self-preference: disagreement may
reflect ordinary model differences, and the unmarked group is the newest unmarked changes, not
cases matched by repository, date, size or risk. Changes whose authorship is `mixed` or
`unknown` are left out. Compare the two teachers with:

    pip install openai
    export OPENAI_API_KEY=...
    python label_second_teacher.py --allow-external openai --model <openai-model-id>
    python evaluate.py teachers claude=data/labels.jsonl openai=data/labels.openai.jsonl

Only repositories listed under "openai" in external_allowlist.json are sent. Every case sent is
logged to <out>.sent.jsonl. Re-running resumes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from common import MAX_UNITS, UNIT_CHARS, allowed_repos, append_jsonl, build_units, load_questions, read_jsonl
from label_changes import SYSTEM, parse, prompt, rubric, schema


def select(cases: list[dict], control: int) -> list[dict]:
    """Every change attributed to one AI agent, plus the newest `control` unmarked ones."""
    ai = [c for c in cases if c.get("written_by", "unmarked") not in ("unmarked", "mixed", "unknown")]
    rest = sorted((c for c in cases if c.get("written_by", "unmarked") == "unmarked"),
                  key=lambda c: c["date"], reverse=True)
    return ai + rest[:control]


def request(case: dict, questions: dict, model: str, unit_chars: int, max_units: int) -> dict:
    units, incomplete = build_units(case, unit_chars, max_units)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM.format(rubric=rubric(questions))},
            {"role": "user", "content": prompt(case, units, incomplete)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "change_labels", "strict": True, "schema": schema(questions)}},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default="data/cases.jsonl")
    ap.add_argument("--first", default="data/labels.jsonl",
                    help="first teacher's labels; only cases it labelled are sent, with the same unit settings")
    ap.add_argument("--out", default="data/labels.openai.jsonl")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--model", required=True, help="OpenAI model id")
    ap.add_argument("--control", type=int, default=200, help="number of unmarked changes to include")
    ap.add_argument("--limit", type=int, default=0, help="label at most N new cases (0 = all selected)")
    ap.add_argument("--allow-external", choices=["openai"],
                    help="required: confirms these diffs and review comments may be sent to OpenAI")
    ap.add_argument("--allowlist", default="external_allowlist.json")
    args = ap.parse_args(argv)
    if args.allow_external != "openai":
        sys.exit("This step sends source diffs and review comments to the OpenAI API.\n"
                 "Re-run with --allow-external openai to confirm. Only repositories listed under "
                 "'openai' in the allowlist file are sent.")
    allowed = allowed_repos("openai", args.allowlist)

    import openai
    client = openai.OpenAI()
    questions = load_questions(args.questions)
    first = {r["case_id"]: r for r in read_jsonl(args.first)}
    done = {r["case_id"] for r in read_jsonl(args.out)}
    cases = [c for c in read_jsonl(args.cases) if c["case_id"] in first]
    todo = [c for c in select(cases, args.control) if c["case_id"] not in done and c["repo"] in allowed]
    if args.limit:
        todo = todo[:args.limit]
    sent_path = Path(args.out).with_suffix(".sent.jsonl")
    errors_path = Path(args.out).with_suffix(".errors.jsonl")
    print(f"{len(done)} already labelled, {len(todo)} to label with {args.model}", file=sys.stderr)

    ok = bad = 0
    for c in todo:
        f = first[c["case_id"]]
        append_jsonl(sent_path, {"provider": "openai", "case_id": c["case_id"], "repo": c["repo"],
                                 "sent_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        try:
            resp = client.chat.completions.create(**request(c, questions, args.model, f["unit_chars"], f["max_units"]))
            choice = resp.choices[0]
            if choice.finish_reason != "stop" or not choice.message.content:
                raise ValueError(f"finish_reason {choice.finish_reason}")
            labels = parse(choice.message.content, c, questions, f["unit_chars"], f["max_units"])
        except (ValueError, KeyError, openai.APIError) as e:
            append_jsonl(errors_path, {"case_id": c["case_id"], "error": str(e)})
            bad += 1
            continue
        append_jsonl(args.out, {"case_id": c["case_id"], "label_source": "inferred", "teacher": args.model,
                                "unit_chars": f["unit_chars"], "max_units": f["max_units"], **labels})
        ok += 1
    print(f"{ok} labelled, {bad} failed" + (f" (see {errors_path})" if bad else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
