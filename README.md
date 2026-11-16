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
cp external_allowlist.example.json external_allowlist.json   # then list the repositories you allow
pip install anthropic
export ANTHROPIC_API_KEY=...
python label_changes.py --allow-external anthropic --limit 20
python label_changes.py --allow-external anthropic

# 1d. Laya train/val/test files, split by time within each repository
python build_dataset.py --out data
```

Every step resumes if interrupted. Results land in `audit_out/` and `data/`, both git-ignored
because they contain repository content.

How the data is shaped:

- A change is cut into **units** (pieces of one file's diff, about 1,800 characters) so each
  fits Laya's context window. Nothing is dropped silently: a change is marked
  `units_complete: false`, with the reason, when it needs more than 40 units, when a changed
  file has no readable diff (too large, or binary), or when not all files of the retrieved
  revision were fetched.
- Each case keeps a **prediction view** (title, description, diff) apart from an **outcome
  view** (review comments, CI, reverts). Only the teacher sees the outcome view.
- For a reviewed PR the diff is the one that stood at the first human review, when GitHub still
  has that commit (`diff_source` records which). GitHub keeps only the current PR description,
  so one edited after the first review is left out (`body_source`).
- Sending code out needs an explicit flag (`--allow-external <provider>`) and an allowlist:
  `external_allowlist.json` names, per provider, the exact repositories whose code may be sent.
  Anything not listed stays local. Every case sent is logged to a `.sent.jsonl` file. Secret
  redaction covers common key formats only; treat it as a safety net.
- Each case records `written_by` for the revision being classified: an AI coding agent named
  in its commit trailers, `mixed` when only some commits are marked or agents differ,
  `unmarked` when there is no marker (which does not prove a human wrote it), or `unknown`
  when the revision's commits could not be determined. `written_by_evidence` lists the markers.
- A change with no readable diff at all (binary-only, for example) is kept as an incomplete
  case with no units, so it appears in coverage and escalation counts.
- Labels are `label_source: "inferred"`: the teacher's judgement, not human-validated.

## 2. Train Laya on the DGX Spark

Needs Docker with the NVIDIA Container Toolkit. `spark/run.sh` clones Laya at a pinned revision
and uses its own ARM64 / CUDA 13.0 container (`compose.spark.yaml`), adding mounts for `data/`
and `out/`.

```bash
spark/run.sh smoke     # GPU inference check
spark/run.sh dryrun    # validate data/train.jsonl, print the optimizer budget
spark/run.sh train     # fine-tune + calibrate + before/after eval on data/val.jsonl
```

The checkpoint is written to `out/hb-review/`, the log to `out/hb-review.train.log`.
Tune settings against `val.jsonl`; `test.jsonl` is read only by `predict` in step 3.
Settings are environment variables, for example:

```bash
BASE=multilingual NAME=hb-review-ml TRAIN_ARGS="--epochs 8 --loss soft-ce" spark/run.sh train
```

## 3. Evaluate and compare with Jev

```bash
spark/run.sh predict                    # tuned Laya   -> out/pred.laya.jsonl
spark/run.sh predict english base       # untuned Laya -> out/pred.base.jsonl

export TYPESAFE_API_KEY=...
python evaluate.py predict --backend jev --allow-external typesafe --out out/pred.jev.jsonl

python evaluate.py report laya=out/pred.laya.jsonl base=out/pred.base.jsonl jev=out/pred.jev.jsonl
```

The report (`out/report.md`, `out/report.json`) gives, per question and backend: unit accuracy
and calibration error, change-level accuracy, under- and over-calls, and how many changes that
needed escalation were missed. A change-level answer is the highest answer among its units.

Backends are compared only on changes that are complete, that every backend answered, and whose
inputs no backend truncated. Changes with an incomplete diff or a truncated input are counted
separately as requiring escalation and are never scored as classified. When the test set mixes
AI-written and unmarked changes, the report also breaks the results down by author.

### Screening the teacher for potential bias

An LLM judge can favour output from its own model family, and the first teacher is Claude. As a
screen, label the AI-attributed changes and a group of unmarked ones again with an OpenAI model
and compare:

```bash
pip install openai
export OPENAI_API_KEY=...
python label_second_teacher.py --allow-external openai --model <openai-model-id>
python evaluate.py teachers claude=data/labels.jsonl openai=data/labels.openai.jsonl
```

If Claude answers lower than the second teacher more often on Claude-written changes than on
unmarked ones, that is a reason to look closer. It is not proof of self-preference: the gap can
come from ordinary model differences, and the two groups are not matched by repository, date,
size or risk. Confirming it needs matched cases and independently reviewed labels.

## Tests

```bash
python -m unittest discover -s tests
```
