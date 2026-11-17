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
- Every unit of a change carries the same **header** (at most 950 characters): the title, the
  start of the description (the repository name is left out, so the classifier cannot learn a
  prior per repository), a summary of the whole change, and a list of up to 8
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
- Labels record a fingerprint of `questions.json` and of every unit the teacher was shown. If
  the questions, the header or the unit code change after labeling, `build_dataset.py` refuses
  the affected labels instead of training on answers to different inputs. Settle
  `questions.json` before the first labeling run.
- Lines in a description or commit message that report how review went ("DeepSeek: PASS, notes
  only", "second and third review rounds") are hindsight. They are removed from the prediction
  view (`body_source: stripped_review_loop`) and kept in the outcome view, where the teacher
  sees them. `build_dataset.py` warns if any unit header still contains one.
- The teacher runs at `--effort high` by default; the model and effort are recorded with every
  label. A case that fails twice is not sent again (`--max-attempts`).
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

## 3. Evaluate

```bash
spark/run.sh predict                    # tuned Laya   -> out/pred.laya.jsonl
spark/run.sh predict english base       # untuned Laya -> out/pred.base.jsonl

python evaluate.py report laya=out/pred.laya.jsonl base=out/pred.base.jsonl
```

Jev is optional. It sends the test diffs to TypeSafe, so it needs a key, `--allow-external
typesafe` and a `typesafe` entry in the allowlist:

```bash
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

Each backend is scored on every complete change it answered in full without truncating an
input, and the report shows that coverage. A backend that covers fewer repositories, such as
Jev behind its allowlist, therefore does not shrink the others' scores. A separate head-to-head
table compares all backends on the changes they have in common. Changes with an incomplete diff
or a truncated input are counted as requiring escalation and are never scored as classified.
When the test set mixes marked and unmarked authorship, the report also breaks the results down
by author.

### Second teacher

The second teacher is a model on your own hardware: Qwen served by vLLM on the Spark, reached
over its OpenAI-compatible chat-completions API. It labels a random 20% of the changes plus the
ones where Claude's label conflicts with an observed signal (changes requested, CI failed or a
revert, on a change Claude rated as needing no human review or as risk 0 to 1). Nothing leaves
your network, so this mode has no external gate; it still logs every case to a `.sent.jsonl`.

```bash
pip install openai
python label_second_teacher.py --base-url http://<spark>:8001/v1 --model qwen38-27b
python evaluate.py teachers claude=data/labels.jsonl qwen=data/labels.second.jsonl
```

In the pilot, run it with `--limit 3` first to confirm the server accepts the strict JSON schema
and returns the answer in `message.content`.

This is a disagreement screen. It cannot establish bias or correctness: a gap can come from
ordinary differences between the models. `--select authors` gives the older comparison by
attributed author, and `--provider openai` uses a hosted model behind the external gate.

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
python evaluate.py report laya=out/pred.laya.jsonl base=out/pred.base.jsonl
python label_second_teacher.py --base-url http://<spark>:8001/v1 --model qwen38-27b --limit 3
```

`--limit` on the labeling step takes a seeded sample across repositories, so the token totals it
prints are a fair basis for estimating the full run.

Expect `laya-train` to warn that calibration was not fitted on so few rows. The pilot passes
when every command finishes, the report lists a row per backend, and a few labels read against
their diffs look sensible. Then remove the limits and re-run steps 1b to 1d; they resume from
what the pilot already fetched and labeled. Delete `out/pilot` and `out/pred.*.jsonl` before the
real run so pilot predictions are not mixed into it.

## Tests

```bash
python -m unittest discover -s tests
```
