#!/usr/bin/env python3
"""Second teacher: label a subset of changes again with another model, to check the first.

The second teacher is a model served locally over an OpenAI-compatible chat-completions API,
by default Qwen on the DGX Spark's vLLM server. It gets the same rubric, prompt and answer
schema as the first teacher (Claude). Nothing leaves your network in this mode.

Which changes it labels (--select, comma-separated):
  random         a seeded random --fraction of the labelled changes (default 0.2)
  disagreements  changes where the first teacher's label conflicts with an observed signal:
                 changes were requested, CI failed and was fixed before merge, or the change
                 was reverted, and yet the first teacher said no human review is needed or
                 scored the risk 0 or 1
  authors        every change attributed to one AI agent plus the newest --control unmarked ones

    pip install openai
    python label_second_teacher.py --base-url http://<spark>:8001/v1 --model qwen38-27b
    python evaluate.py teachers claude=data/labels.jsonl qwen=data/labels.second.jsonl

Comparing two teachers is a disagreement screen. It cannot establish bias or correctness: a gap
may reflect ordinary model differences, and the groups are not matched.

To use a hosted model instead, pass --provider openai. That sends code out, so it needs
--allow-external openai and an "openai" entry in external_allowlist.json. Every case sent, in
either mode, is logged to <out>.sent.jsonl. Re-running resumes.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

from common import allowed_repos, append_jsonl, build_units, load_questions, questions_sha, read_jsonl
from label_changes import SYSTEM, parse, prompt, rubric, schema

AGENTS_EXCLUDED = ("unmarked", "mixed", "unknown")


def conflicts(case: dict, label: dict) -> bool:
    """An observed sign of trouble, on a change the first teacher rated as needing little."""
    o = case["outcome"]["observed"]
    # CI that is red on the final commit is left out: in a repository whose CI stays red it says
    # nothing about the change. CI that failed and was then fixed before merge does.
    trouble = o.get("changes_requested") or o.get("ci_red_then_green") or o.get("reverted")
    ch = label["change"]
    return bool(trouble) and (ch.get("human_review") == "none" or ch.get("risk_score", 9) <= 1)


def select(cases: list[dict], first: dict, modes: list[str], fraction: float = 0.2,
           control: int = 200, seed: int = 0) -> list[dict]:
    chosen = {}
    if "random" in modes:
        pool = sorted(cases, key=lambda c: c["case_id"])
        for c in random.Random(seed).sample(pool, round(len(pool) * fraction)):
            chosen[c["case_id"]] = c
    if "disagreements" in modes:
        for c in cases:
            if conflicts(c, first[c["case_id"]]):
                chosen[c["case_id"]] = c
    if "authors" in modes:
        rest = sorted((c for c in cases if c.get("written_by", "unmarked") == "unmarked"),
                      key=lambda c: c["date"], reverse=True)
        for c in [c for c in cases if c.get("written_by", "unmarked") not in AGENTS_EXCLUDED] + rest[:control]:
            chosen[c["case_id"]] = c
    return list(chosen.values())


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
                    help="first teacher's labels; only cases it labelled are used, with the same unit settings")
    ap.add_argument("--out", default="data/labels.second.jsonl")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--model", required=True, help="model id on the server, e.g. qwen38-27b")
    ap.add_argument("--provider", choices=["local", "openai"], default="local")
    ap.add_argument("--base-url", help="local: the server's OpenAI-compatible URL, e.g. http://<spark>:8001/v1")
    ap.add_argument("--select", default="random,disagreements",
                    help="comma-separated: random, disagreements, authors")
    ap.add_argument("--fraction", type=float, default=0.2, help="share of changes for --select random")
    ap.add_argument("--control", type=int, default=200, help="unmarked changes for --select authors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="label at most N new cases (0 = all selected)")
    ap.add_argument("--allow-external", choices=["openai"],
                    help="--provider openai only: confirms diffs and review comments may be sent to OpenAI")
    ap.add_argument("--allowlist", default="external_allowlist.json")
    args = ap.parse_args(argv)
    modes = [m.strip() for m in args.select.split(",") if m.strip()]
    if not modes or set(modes) - {"random", "disagreements", "authors"}:
        sys.exit("--select takes a comma-separated list of: random, disagreements, authors")

    allowed = None      # local: no external gate, every repository may be read by your own server
    if args.provider == "local":
        if not args.base_url:
            sys.exit("--provider local needs --base-url, the address of your own server "
                     "(for example http://<spark>:8001/v1).")
    else:
        if args.allow_external != "openai":
            sys.exit("--provider openai sends source diffs and review comments to the OpenAI API.\n"
                     "Re-run with --allow-external openai to confirm. Only repositories listed under "
                     "'openai' in the allowlist file are sent.")
        allowed = allowed_repos("openai", args.allowlist)

    import openai
    if args.provider == "local":
        # vLLM ignores the key, but the client refuses to start without one.
        client = openai.OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY") or "local")
    else:
        client = openai.OpenAI()

    questions = load_questions(args.questions)
    sha = questions_sha(questions)
    first = {r["case_id"]: r for r in read_jsonl(args.first) if r.get("questions_sha") == sha}
    done = {r["case_id"] for r in read_jsonl(args.out)}
    cases = [c for c in read_jsonl(args.cases) if c["case_id"] in first]
    picked = select(cases, first, modes, args.fraction, args.control, args.seed)
    todo = [c for c in picked if c["case_id"] not in done and (allowed is None or c["repo"] in allowed)]
    if args.limit:
        todo = todo[:args.limit]
    sent_path = Path(args.out).with_suffix(".sent.jsonl")
    errors_path = Path(args.out).with_suffix(".errors.jsonl")
    where = args.base_url if args.provider == "local" else "OpenAI"
    print(f"{len(picked)} changes selected ({', '.join(modes)}); {len(done)} already labelled, "
          f"{len(todo)} to label with {args.model} at {where}", file=sys.stderr)

    ok = bad = 0
    for c in todo:
        f = first[c["case_id"]]
        append_jsonl(sent_path, {"provider": args.provider, "endpoint": where, "case_id": c["case_id"],
                                 "repo": c["repo"], "sent_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
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
                                "questions_sha": sha, "selected_by": modes,
                                "unit_chars": f["unit_chars"], "max_units": f["max_units"], **labels})
        ok += 1
    print(f"{ok} labelled, {bad} failed" + (f" (see {errors_path})" if bad else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
