# hb-review-lab

Builds a code-review routing dataset from GitHub history, fine-tunes Laya on it on a DGX Spark,
and compares the tuned model with Jev.

The classifier answers six questions about a code change (`questions.json`): risk score,
criticality, and whether it needs human, UX, security and performance review.

## 1. Gather data

Needs Python 3.10+ and a GitHub token with read access. All GitHub access is read-only.

```bash
export GITHUB_TOKEN=...

# 1a. Inventory: every PR and default-branch commit, with review/CI/revert signals
python repo_audit.py --owner <login> --out audit_out
python repo_audit.py --repo django/django --repo grafana/grafana --max-prs 500 --max-commits 0 --out audit_out

# 1b. Diffs and review outcomes for those changes (secrets redacted)
python scrape_changes.py --candidates audit_out/candidates.jsonl --out data/cases.jsonl

# 1c. Teacher labels from Claude (Batches API). Try 20 cases first.
pip install anthropic
export ANTHROPIC_API_KEY=...
python label_changes.py --limit 20
python label_changes.py

# 1d. Laya train/test files, split by time within each repository
python build_dataset.py --out data
```

Every step resumes if interrupted. Results land in `audit_out/` and `data/`, both git-ignored
because they contain repository content.

How the data is shaped:

- A change is cut into **units** (pieces of one file's diff, about 1,800 characters) so each
  fits Laya's context window. Nothing is dropped silently: a change needing more than 40 units
  is marked `units_complete: false`.
- Each case keeps a **prediction view** (title, description, diff) apart from an **outcome
  view** (review comments, CI, reverts). Only the teacher sees the outcome view.
- For a reviewed PR the diff is the one that stood at the first human review, when GitHub still
  has that commit (`diff_source` records which).
- Labels are `label_source: "inferred"`: the teacher's judgement, not human-validated.

## 2. Train Laya on the DGX Spark

Needs Docker with the NVIDIA Container Toolkit. `spark/run.sh` clones Laya at a pinned revision
and uses its own ARM64 / CUDA 13.0 container (`compose.spark.yaml`), adding mounts for `data/`
and `out/`.

```bash
spark/run.sh smoke     # GPU inference check
spark/run.sh dryrun    # validate data/train.jsonl, print the optimizer budget
spark/run.sh train     # fine-tune + calibrate + before/after eval on data/test.jsonl
```

The checkpoint is written to `out/hb-review/`, the log to `out/hb-review.train.log`.
Settings are environment variables, for example:

```bash
BASE=multilingual NAME=hb-review-ml TRAIN_ARGS="--epochs 8 --loss soft-ce" spark/run.sh train
```

## 3. Evaluate and compare with Jev

```bash
spark/run.sh predict                    # tuned Laya   -> out/pred.laya.jsonl
spark/run.sh predict english base       # untuned Laya -> out/pred.base.jsonl

export TYPESAFE_API_KEY=...
python evaluate.py predict --backend jev --out out/pred.jev.jsonl

python evaluate.py report laya=out/pred.laya.jsonl base=out/pred.base.jsonl jev=out/pred.jev.jsonl
```

The report (`out/report.md`, `out/report.json`) gives, per question and backend: unit accuracy
and calibration error, change-level accuracy, under- and over-calls, and how many changes that
needed escalation were missed. A change-level answer is the highest answer among its units.

The Jev step sends the held-out diffs to TypeSafe's API, and step 1c sends all diffs to
Anthropic's. Check that is acceptable for private repositories before running them.

## Tests

```bash
python -m unittest discover -s tests
```
