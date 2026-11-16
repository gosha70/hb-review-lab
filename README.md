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
python repo_audit.py --repo django/django --repo grafana/grafana --max-prs 500 --max-commits 1000 --out audit_out

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

For a public repository always set both limits. `--max-commits 0` means the whole history,
which for an old project includes thousands of commits from before it used pull requests. Runs
for different repositories add to the same `audit_out/`; a repository already collected is not
fetched again, so use a new `--out` to change its limits.

Every step resumes if interrupted. Results land in `audit_out/` and `data/`, both git-ignored
because they contain repository content.

How the data is shaped:

- A change is cut into **units** so each fits Laya's context window. A unit is the change
  header plus one piece of one file's diff, at most 2,250 characters in total, so a longer
  header leaves less room for the diff. Nothing is dropped silently: a change is marked
  `units_complete: false`, with the reason, when it needs more than 60 units, when a changed
  file has no readable diff (too large, or binary), or when not all files of the retrieved
  revision were fetched.
- Every unit of a change carries the same **header** (at most 950 characters): repository,
  title, the start of the description, a summary of the whole change, and a list of up to 8
  files. The summary gives files added/deleted/renamed/modified; files and lines per kind
  (source, test, docs); test lines per source line; files per path stratum (security,
  migration, infra/CI, dependencies, UI); the top directories; and files without a readable
  diff. The file list puts stratum files first, then source files, by lines changed. All of it
  comes from the prediction view of the revision being classified. The teacher is shown the
  same header and labels each unit for its own part in the change, read with that context.
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
  `unmarked` when there is no marker, or `unknown`
  when the revision's commits could not be determined. `written_by_evidence` lists the markers.
  Many agent-written commits carry no marker at all, so `unmarked` is not "human": in a
  repository built with an AI agent it mostly means "agent, without a marker". The by-author
  tables compare marked with unmarked changes; they cannot be read as AI versus human.
- Labels record a fingerprint of `questions.json`. If the questions change after labeling,
  `build_dataset.py` refuses the old labels instead of training on answers to different
  questions.
- A change with no readable diff at all (binary-only, for example) is kept as an incomplete
  case with no units, so it appears in coverage and escalation counts.
- Labels are `label_source: "inferred"`: the teacher's judgement, not human-validated.

## 2. Train Laya on the DGX Spark

`spark/run.sh` reads `data/` and writes `out/` inside this repository's checkout on the Spark.
Either run step 1 on the Spark as well, or copy the dataset over and the results back:

```bash
rsync -av data/train.jsonl data/val.jsonl data/test.jsonl data/test_changes.jsonl data/majority.json <spark>:hb-review-lab/data/
rsync -av <spark>:hb-review-lab/out/ out/     # after step 3, for the report
```

The predictions and `evaluate.py report` need `data/test.jsonl`, `data/test_changes.jsonl` and
`data/majority.json` on whichever machine writes the report.

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

Each table also has a `majority` row: a classifier that always gives the train split's most
common answer (`data/majority.json`, written by `build_dataset.py`). On a skewed question a
backend can score high by doing the same, so a backend that does not beat `majority` there has
learned nothing useful. The report also lists how the teacher's labels are distributed.

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

## First run: a pilot through every step

None of the external paths (GitHub, the teacher APIs, the Spark, Jev) is covered by the offline
tests. Before the full run, take about 20 changes through **every** step, including training,
prediction and the report. The numbers will mean nothing at this size; the point is to find
integration failures while they are cheap.

```bash
python repo_audit.py --owner <login> --out audit_out
python scrape_changes.py --max-per-repo 5 --out data/cases.jsonl
python label_changes.py --allow-external anthropic --limit 20
python build_dataset.py --out data
spark/run.sh smoke
spark/run.sh dryrun
TRAIN_ARGS="--epochs 1" NAME=pilot spark/run.sh train
NAME=pilot spark/run.sh predict
spark/run.sh predict english base
python evaluate.py predict --backend jev --allow-external typesafe --out out/pred.jev.jsonl
python evaluate.py report laya=out/pred.laya.jsonl base=out/pred.base.jsonl jev=out/pred.jev.jsonl
```

Expect `laya-train` to warn that calibration was not fitted on so few rows. The pilot passes
when every command finishes, the report lists a row per backend, and a few labels read against
their diffs look sensible. Then remove the limits and re-run steps 1b to 1d; they resume from
what the pilot already fetched and labeled. Delete `out/pilot` and `out/pred.*.jsonl` before the
real run so pilot predictions are not mixed into it.

## Tests

```bash
python -m unittest discover -s tests
```
