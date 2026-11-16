#!/usr/bin/env python3
"""Step 1d: turn cases + teacher labels into Laya training and evaluation files.

Writes to --out:
    train.jsonl          one row per change unit, in the format laya-train and laya-evals read
                         ({state, questions, expected}, plus id/case_id/tags)
    val.jsonl            same format; for tuning training settings
    test.jsonl           same format; for the final comparison only
    test_changes.jsonl   one row per test change: its change-level labels and unit ids

Split: within each repository, in date order, the oldest changes train, the next --val-frac
validate and the newest --test-frac test, so evaluation is always on changes made after the
ones trained on. Repositories named with --holdout-repo go to test entirely. A change is never
split across files.

Usage:
    python build_dataset.py --cases data/cases.jsonl --labels data/labels.jsonl --out data
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import build_units, load_questions, read_jsonl, unread_files, write_jsonl


def split(cases: list[dict], val_frac: float, test_frac: float, holdout: set[str]):
    by_repo = defaultdict(list)
    for c in cases:
        by_repo[c["repo"]].append(c)
    train, val, test = [], [], []
    for repo, rows in by_repo.items():
        rows.sort(key=lambda c: c["date"])
        if repo in holdout:
            test += rows
            continue
        n = len(rows)
        n_test = max(1, round(n * test_frac)) if n > 1 else 0
        n_val = max(1, round(n * val_frac)) if n - n_test > 1 and val_frac > 0 else 0
        train += rows[:n - n_test - n_val]
        val += rows[n - n_test - n_val:n - n_test]
        test += rows[n - n_test:]
    return train, val, test


def unit_rows(case: dict, label: dict, questions: dict) -> list[dict]:
    units, _ = build_units(case, label["unit_chars"], label["max_units"])
    return [{
        "id": f"{case['case_id']}::{u['unit']}", "case_id": case["case_id"],
        "state": u["state"], "questions": questions, "expected": label["units"][u["unit"]],
        "tags": [case["repo"], case["kind"], f"by:{case.get('written_by', 'unmarked')}"],
    } for u in units if u["unit"] in label["units"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default="data/cases.jsonl")
    ap.add_argument("--labels", default="data/labels.jsonl")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--holdout-repo", action="append", default=[], help="owner/name (repeatable)")
    args = ap.parse_args(argv)

    questions = load_questions(args.questions)
    labels = {l["case_id"]: l for l in read_jsonl(args.labels)}
    cases = [c for c in read_jsonl(args.cases) if c["case_id"] in labels]
    if not cases:
        sys.exit("No labelled cases. Run scrape_changes.py and label_changes.py first.")
    train, val, test = split(cases, args.val_frac, args.test_frac, set(args.holdout_repo))

    out = Path(args.out)
    train_rows = [r for c in train for r in unit_rows(c, labels[c["case_id"]], questions)]
    val_rows = [r for c in val for r in unit_rows(c, labels[c["case_id"]], questions)]
    test_rows = [r for c in test for r in unit_rows(c, labels[c["case_id"]], questions)]
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "val.jsonl", val_rows)
    write_jsonl(out / "test.jsonl", test_rows)
    by_case = defaultdict(list)
    for r in test_rows:
        by_case[r["case_id"]].append(r["id"])
    write_jsonl(out / "test_changes.jsonl", [{
        "case_id": c["case_id"], "repo": c["repo"], "date": c["date"],
        "expected": labels[c["case_id"]]["change"], "units": by_case[c["case_id"]],
        "units_complete": labels[c["case_id"]]["units_complete"],
        "incomplete_reasons": labels[c["case_id"]].get("incomplete_reasons", []),
        "unread_files": unread_files(c), "written_by": c.get("written_by", "unmarked"),
    } for c in test])

    print(f"train: {len(train)} changes, {len(train_rows)} units")
    print(f"val:   {len(val)} changes, {len(val_rows)} units")
    print(f"test:  {len(test)} changes, {len(test_rows)} units "
          f"({sum(not labels[c['case_id']]['units_complete'] for c in test)} incomplete)")
    for qid in questions:
        tr = Counter(str(r["expected"][qid]) for r in train_rows)
        te = Counter(str(labels[c["case_id"]]["change"][qid]) for c in test)
        print(f"  {qid}: train units {dict(sorted(tr.items()))} | test changes {dict(sorted(te.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
