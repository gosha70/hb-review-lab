#!/usr/bin/env python3
"""Step 1d: turn cases + teacher labels into Laya training and evaluation files.

Writes to --out:
    train.jsonl          one row per change unit, in the format laya-train and laya-evals read
                         ({state, questions, expected}, plus id/case_id/tags)
    val.jsonl            same format; for tuning training settings
    test.jsonl           same format; for the final comparison only
    test_changes.jsonl   one row per test change: its change-level labels and unit ids
    majority.json        per question, the most common unit label and the most common change
                         label in the train split: the constant baseline the report compares with

Split: within each repository, in date order, the oldest changes train, the next --val-frac
validate and the newest --test-frac test, so evaluation is always on changes made after the
ones trained on. Repositories named with --holdout-repo go to test entirely. A change is never
split across files.

Usage:
    python build_dataset.py --cases data/cases.jsonl --labels data/labels.jsonl --out data
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import (MAX_UNITS, UNIT_CHARS, build_units, load_questions, options, questions_sha, read_jsonl, state_sha, strip_review_loop,
                    unread_files,
                    write_jsonl)


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


class StaleLabel(Exception):
    pass


def majority(values: list, q: dict):
    """The most common answer; ties go to the lower option. None when there are no values."""
    counts = Counter(values)
    best = None
    for o in options(q):
        if counts[o] and (best is None or counts[o] > counts[best]):
            best = o
    return best


def unit_rows(case: dict, label: dict, questions: dict) -> list[dict]:
    units, _ = build_units(case, label["unit_chars"], label["max_units"])
    # The teacher labelled a specific input. If the header or the chunking has changed since,
    # the rebuilt unit is a different input and the label no longer belongs to it.
    shas = label.get("unit_sha")
    if label["units"] and (shas is None or any(shas.get(u["unit"]) != state_sha(u["state"])
                                               for u in units if u["unit"] in label["units"])):
        raise StaleLabel(case["case_id"])
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
    # Labels answer the questions as they were when the teacher ran. If questions.json has
    # changed since (wording, options or order), those labels no longer mean the same thing.
    sha = questions_sha(questions)
    stale = [k for k, l in labels.items() if l.get("questions_sha") != sha]
    if stale:
        sys.exit(f"{len(stale)} of {len(labels)} labels in {args.labels} were made with a different "
                 f"questions.json (current fingerprint {sha}).\n"
                 "Label again into a new file, or restore the questions file the labels were made with.")
    # A change with no readable diff has no units and so no teacher labels. It is kept as an
    # incomplete case so it shows up in coverage and escalation counts.
    cases = []
    for c in read_jsonl(args.cases):
        if c["case_id"] in labels:
            cases.append(c)
        else:
            units, reasons = build_units(c)
            if not units:
                labels[c["case_id"]] = {"change": None, "units": {}, "unit_chars": UNIT_CHARS,
                                        "max_units": MAX_UNITS, "units_complete": False,
                                        "incomplete_reasons": reasons or ["no_readable_diff"]}
                cases.append(c)
    if not any(labels[c["case_id"]]["change"] for c in cases):
        sys.exit("No labelled cases. Run scrape_changes.py and label_changes.py first.")
    train, val, test = split(cases, args.val_frac, args.test_frac, set(args.holdout_repo))

    out = Path(args.out)
    drifted = []

    def rows_of(group):
        rows = []
        for c in group:
            try:
                rows += unit_rows(c, labels[c["case_id"]], questions)
            except StaleLabel:
                drifted.append(c["case_id"])
        return rows

    train_rows, val_rows, test_rows = rows_of(train), rows_of(val), rows_of(test)
    if drifted:
        sys.exit(f"{len(drifted)} labelled changes no longer produce the units the teacher saw "
                 f"(first: {drifted[0]}).\nThe header or unit code, or cases.jsonl, changed after "
                 "labelling. Label those changes again, or restore the code the labels were made with.")
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

    train_changes = [labels[c["case_id"]]["change"] for c in train if labels[c["case_id"]]["change"]]
    (out / "majority.json").write_text(json.dumps({
        "source": "train split only",
        "unit": {qid: majority([r["expected"][qid] for r in train_rows], q) for qid, q in questions.items()},
        "change": {qid: majority([ch[qid] for ch in train_changes], q) for qid, q in questions.items()},
    }, indent=1))

    print(f"train: {len(train)} changes, {len(train_rows)} units")
    print(f"val:   {len(val)} changes, {len(val_rows)} units")
    print(f"test:  {len(test)} changes, {len(test_rows)} units "
          f"({sum(not labels[c['case_id']]['units_complete'] for c in test)} incomplete, "
          f"{sum(not labels[c['case_id']]['change'] for c in test)} of them with no readable diff)")
    leaked = sum(bool(strip_review_loop(r["state"].split("\nFiles:\n")[0])[1])
                 for r in train_rows + val_rows + test_rows)
    if leaked:
        print(f"WARNING: {leaked} unit headers still mention review rounds or reviewer results.")
    print("Effective training size is changes, not units: units of one change share a header.")
    for qid, q in questions.items():
        per = Counter(ch[qid] for ch in train_changes)
        print(f"  {qid}: train changes per answer {{{', '.join(f'{o}: {per[o]}' for o in options(q))}}}; "
              f"rarest answer has {min(per[o] for o in options(q))}")
    for qid in questions:
        tr = Counter(str(r["expected"][qid]) for r in train_rows)
        te = Counter(str(labels[c["case_id"]]["change"][qid]) for c in test if labels[c["case_id"]]["change"])
        print(f"  {qid}: train units {dict(sorted(tr.items()))} | test changes {dict(sorted(te.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
