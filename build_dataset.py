#!/usr/bin/env python3
"""Step 1d: turn cases + teacher labels into Laya training and evaluation files.

Writes to --out:
    train.jsonl          one row per change unit, in the format laya-train and laya-evals read
                         ({state, questions, expected}, plus id/case_id/tags)
    test.jsonl           same format, held out
    test_changes.jsonl   one row per held-out change: its change-level labels and unit ids

Split: within each repository the newest --test-frac of changes go to test, so the model is
always evaluated on changes made after the ones it trained on. Repositories named with
--holdout-repo go to test entirely. A change is never split across train and test.

Usage:
    python build_dataset.py --cases data/cases.jsonl --labels data/labels.jsonl --out data
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import build_units, load_questions, read_jsonl, write_jsonl


def split(cases: list[dict], test_frac: float, holdout: set[str]) -> tuple[list, list]:
    by_repo = defaultdict(list)
    for c in cases:
        by_repo[c["repo"]].append(c)
    train, test = [], []
    for repo, rows in by_repo.items():
        rows.sort(key=lambda c: c["date"])
        if repo in holdout:
            test += rows
            continue
        n_test = max(1, round(len(rows) * test_frac)) if len(rows) > 1 else 0
        train += rows[:len(rows) - n_test]
        test += rows[len(rows) - n_test:]
    return train, test


def unit_rows(case: dict, label: dict, questions: dict) -> list[dict]:
    units, _ = build_units(case, label["unit_chars"], label["max_units"])
    return [{
        "id": f"{case['case_id']}::{u['unit']}", "case_id": case["case_id"],
        "state": u["state"], "questions": questions, "expected": label["units"][u["unit"]],
        "tags": [case["repo"], case["kind"]],
    } for u in units if u["unit"] in label["units"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default="data/cases.jsonl")
    ap.add_argument("--labels", default="data/labels.jsonl")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--out", default="data")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--holdout-repo", action="append", default=[], help="owner/name (repeatable)")
    args = ap.parse_args(argv)

    questions = load_questions(args.questions)
    labels = {l["case_id"]: l for l in read_jsonl(args.labels)}
    cases = [c for c in read_jsonl(args.cases) if c["case_id"] in labels]
    if not cases:
        sys.exit("No labelled cases. Run scrape_changes.py and label_changes.py first.")
    train, test = split(cases, args.test_frac, set(args.holdout_repo))

    out = Path(args.out)
    train_rows = [r for c in train for r in unit_rows(c, labels[c["case_id"]], questions)]
    test_rows = [r for c in test for r in unit_rows(c, labels[c["case_id"]], questions)]
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "test.jsonl", test_rows)
    by_case = defaultdict(list)
    for r in test_rows:
        by_case[r["case_id"]].append(r["id"])
    write_jsonl(out / "test_changes.jsonl", [{
        "case_id": c["case_id"], "repo": c["repo"], "date": c["date"],
        "expected": labels[c["case_id"]]["change"], "units": by_case[c["case_id"]],
        "units_complete": labels[c["case_id"]]["units_complete"],
    } for c in test])

    print(f"train: {len(train)} changes, {len(train_rows)} units")
    print(f"test:  {len(test)} changes, {len(test_rows)} units")
    for qid in questions:
        tr = Counter(str(r["expected"][qid]) for r in train_rows)
        te = Counter(str(labels[c["case_id"]]["change"][qid]) for c in test)
        print(f"  {qid}: train units {dict(sorted(tr.items()))} | test changes {dict(sorted(te.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
