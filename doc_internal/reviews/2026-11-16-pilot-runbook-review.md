---
title: "Code review: hb-review-lab pipeline and pilot runbook at c3b35ac"
date: 2026-11-16
reviewer: Claude Code (Opus 5.5)
subject: branch claude/elegant-archimedes-0b8l9f at c3b35ac, and the pilot runbook written for it
companion: 2026-11-16-code-review-addendum.md (owner decisions and data the pipeline doesn't use yet)
verdict: changes requested before the pilot
rereviewed: 99cdcd9 (2026-11-16): blockers fixed; W1 partly fixed; W2–W4 open
rereviewed_2: b847917 (2026-11-16): W1–W4 fixed; new W2b and W3b open; owner keeps criticality
---

# Code review: pipeline and pilot runbook at `c3b35ac`

**Verdict: changes requested before the pilot.**

**Classification: on-target for what it builds.** It distils a build-stage change classifier from
Claude labels and compares it with untuned Laya, a majority baseline and Jev. The pilot-first plan
is right, the code is lean, and the runbook's closing caveat is accurate: the result measures
agreement with the teacher, not correctness. The owner decided things today that the builder
couldn't know; those gaps are in the companion addendum.

Severities follow the review-loop scale (blocking / warning / note). File:line references are at
`c3b35ac`.

**Verified:**

- 15 of 15 tests pass on Python 3.13.
- The pinned Laya ref `b1d6569` is v0.4.35 (released 2026-11-16).
- Laya's container entrypoint `exec`s its arguments, so the `spark/run.sh` invocations are sound.
- Read-only GitHub checks of the owner's repositories and of django/django and grafana/grafana
  produced the numbers below.

## Re-review of `b847917` (latest)

What I checked:

- Read the whole diff; 18 of 18 tests pass.
- Ran `strip_review_loop()` over 238 real merged PR descriptions and titles from wiki-studio,
  code-copilot-team, ai-atlas and memkernel, and over probe sentences.
- Read issues #5–#8.

**Verdict.** Everything this review required before labelling is addressed, and the issues reflect
the addendum faithfully. Before the full labelling run, two things remain:

- **W2b:** the stripper removes too much.
- **W3b:** the disagreement signal is nearly empty on this data.

The owner has decided to keep `criticality` (see below).

| Item | Status | Notes |
|---|---|---|
| W1 drift guard | Fixed | Each label stores a `unit_sha` per unit, `build_dataset.py` refuses any mismatch, and `questions_sha` is kept. |
| W2 description leak | Fixed, but it over-strips | Leaks into the unit header fell from 8 to 1 of 238. Some removed lines describe the change itself (W2b). |
| W3 second teacher | Fixed | Local provider by default; a random 20% plus disagreements; the hosted mode stays behind the gate; the endpoint is logged. The disagreement signal is weak here (W3b). |
| W4 Jev scoring | Fixed | Each backend is scored on its own changes, with coverage. A head-to-head table (majority included) uses the common set. Jev is out of the pilot, and the example allowlist no longer names Django. |
| B1 gap | Fixed | Unfinished repositories are logged. |
| N1, N2, N5, N6, N7 | Fixed | Seeded `--limit` sample; per-answer counts; effort set inside `output_config` (the correct place) and recorded; at most 2 attempts per case; repository name out of the header. |
| N3, N4 | Filed in #8 | Fine to defer. |

### W2b. The stripper removes descriptions of the change, including security evidence (warning)

- **Where:** `common.py` `_ROUND`, `_REVIEWER`, `_RESULT` and `strip_review_loop()`.
- **Evidence:**
  - It removed 120 lines from 75 of the 238 descriptions.
  - 8 of 12 sampled code-copilot-team lines describe the change rather than report a review
    outcome. #418 is one: "A gating reviewer's finding that a named test fails counted
    toward the breaker…" (the bug being fixed). #425 and #433 are others.
  - code-copilot-team, with 320 PRs, is itself review tooling, so its subject matter matches the
    patterns.
- **Probes that should be kept but are stripped:**
  - "Hash passwords with bcrypt at 12 rounds."
  - "Derive keys with PBKDF2, 600000 rounds."
  - "Retry the upload up to 3 rounds before it fails."
  - "Round 2 of the migration moves the remaining tables."
  - "Add a reviewer field; approved requests now show who signed off."

  The two crypto lines are exactly the security evidence the classifier needs.
- **Fix:**
  - Count a round only when "review" is part of the phrase ("second review round", "review rounds",
    "rounds of review"). Drop the generic `\brounds? \d` and `\b\d+ rounds?\b` alternatives, and the
    bare `\breviewer\b`.
  - Require the reviewer and the result to sit together: "DeepSeek: PASS", "Codex review: 2
    findings", "the second review found".
  - Drop whole sections under a heading such as "## Review rounds".
  - Add the probes above as must-keep tests, and the lines from #126, #127, #132, #443 and #449 as
    must-strip tests.
  - Do it before the full labelling run. A new stripper means re-scraping, and the drift guard will
    then refuse labels made from the old headers.

### W3b. The disagreement signal is nearly empty on this data (warning)

- **Where:** `label_second_teacher.py` `conflicts()` uses `changes_requested`, `ci_failed` and
  `reverted`. `ci_failed` reads only the final head commit (`repo_audit.py:75` and `:308`).
- **On the owner's repositories:**
  - Changes were requested on 3 PRs, and nothing was reverted.
  - Final-state CI failures cluster in mileroot and hookatlas, whose CI stays red: 56 and 13 PRs
    with failures, 2 recovered in each. That's noise.
- **The strongest observed signal here is CI going red, then green before merge:** 116 of 828
  human PRs.
  - Capturing it is a single change at `repo_audit.py:75`: `commits(last:1)` becomes
    `commits(first:100){…statusCheckRollup{state}}`, which yields `observed.ci_red_then_green`.
  - #8 files it as a follow-up. Do it before the full labelling run instead: the outcome view is
    what the teacher reads, so labels made without it miss the most common observed signal here.

### The builder's decisions

- **Teacher effort `high`.** Keep it for now, but in the pilot also label the same 20 cases at
  `medium`. The agreement between the two doubles as a teacher self-agreement check, and the token
  totals show the cost difference. If the change-level answers agree on at least 90% of cases, use
  `medium` for the full run.
- **One-directional disagreement.** Agree. An observed problem on a change rated low is evidence
  of an under-call. A clean outcome on a change rated high is not evidence of an over-call.
- **Stripping review-round lines without a result word.** Agreed in principle, provided the
  pattern requires "review" (W2b).

### `criticality`: owner decision

**Owner decision (2026-11-16): keep `criticality` in `questions.json`.** It is settled; label with
it.

### Merge readiness

- **Open the pull request now,** so this review is attached to it.
- **Before merging,** land W2b and, preferably, W3b. Both change what the teacher sees.
- **Then run the pilot from `main`.**

## Re-review of `99cdcd9`

I read the whole diff; 16 of 16 tests pass. Line references in this section are at `99cdcd9`;
those in the original findings below stay at `c3b35ac`.

| Finding | Status | Evidence, and what's left |
|---|---|---|
| B1 audit overwrite | Fixed | `repo_audit.py:497-507` re-analyses repositories from earlier runs, and a regression test covers it. Small gap: a repository whose earlier run stopped part-way (PRs or commits unfinished) is left out of the outputs silently. Log it. |
| B2 `--max-commits 0` | Fixed in the docs | The README example now sets `--max-commits 1000`, and the help text explains that 0 means all. There's no code guard, which is acceptable. |
| B3 pilot | Fixed, with one new problem | The README pilot now runs train, predict and report. But it runs Jev unconditionally (`README.md:173-174`): without `TYPESAFE_API_KEY` it stops, and with `jev=` in the report every backend is scored on Jev's subset (W4). Make the Jev lines optional and drop `jev=` from the default report. |
| W1 stale labels | Partly fixed | `questions_sha` now guards `questions.json`. Changes to the header and unit construction (`change_header()`, `_chunks()`) still aren't detected. Store a hash of each unit's state in the label row, and have `build_dataset.py` compare it with the rebuilt unit. |
| W2 description leak | Open | Answer below. Fix before the full labelling run: the header is part of what the teacher labels, so changing it later means labelling again. |
| W3 second teacher | Open | Answer below. |
| W4 Jev scoring | Open | Answer below. |
| W5 data hand-off | Fixed | The README now has the rsync section. |
| N1–N7 | Open | See Notes. |

### Answers to the builder's questions

**W2: the specific case.**

- **Why the guard never fires here.** `scrape_changes.py:94-97` drops a description only when it was
  edited after the first *human* review (`first`, at `:77`, is the first review by someone other
  than the author). Only 12 of the owner's 1,023 PRs have one.
- **Where the leaking text comes from.** The owner's review loop. The agent writes its results into
  the description:
  - memkernel #126, never edited: "DeepSeek, two rounds on the spec: round 1's points applied;
    round 2 one warning …"
  - memkernel #132: "DeepSeek: PASS, notes only."
  - memkernel #127: "DeepSeek on A1 and on the second-review fixes: PASS, notes only. DeepSeek on
    the build: PASS …"
  - memkernel #124: "second and third review rounds".
- **How common it is.** In wiki-studio, code-copilot-team, ai-atlas and memkernel, 67 of 238 recent
  merged descriptions match `review (round|loop)|peer review|codex|deepseek|verdict`. 8 of them match
  inside the first 300 characters, which reach every unit header (`common.py:175`).
- **Why "use the first-written description" isn't enough.** #126 was never edited, so it was
  written with the results already in it.
- **Fix.** Before a description or commit-message body enters the prediction view, drop the lines
  that name a reviewer or a review round together with a result. Keep the original in the outcome
  view, record `body_source: "stripped_review_loop"`, and log the stripped lines.
  - Don't strip on the bare word "Codex": the owner's codex-evaluator repository uses it as a
    subject.

**W3: the decision, and the endpoint.**

- **Decision (owner, 2026-11-16).** The second teacher is Qwen3.8-27B, served by vLLM on the Spark.
  It labels a random slice (suggest 20%) plus disagreements: cases where Claude's label conflicts
  with an observed signal, such as changes requested, CI going red then green, or a review-loop FAIL
  or blocking finding. OpenAI isn't used.
- **Endpoint.**
  - It is OpenAI-compatible chat completions at `http://<spark>:8001/v1`, model id `qwen38-27b`,
    with a 262,144-token context.
  - The host and launch flags are in the public repository gosha70/code-copilot-team:
    `docs/dgx-spark/runbook-qwen38.md` and `docs/dgx-spark/vllm-qwen38.md`.
  - vLLM ignores the API key, but the client needs one (`OPENAI_API_KEY=dummy`).
  - It runs with `--reasoning-parser qwen3`, so reasoning arrives separately and `message.content`
    should be the JSON. Verify one strict-schema request in the pilot.
  - Stay on chat completions: the runbook records problems with this server's Responses API.
- **Code.**
  - Setting `OPENAI_BASE_URL` reaches it. But `--allow-external openai` and an "openai" allowlist
    entry would then be gating a local endpoint. Add a local-provider mode that skips the external
    gate and keeps the `.sent.jsonl` log.
  - Add `--select random --fraction 0.2` and `--select disagreements` alongside the current
    author-based selection.

**W4: recommendation.**

- **Scoring.** Score each backend on every change it answered in full, and show its coverage. Keep
  the all-backends intersection only as a separate paired head-to-head table.
- **Make Jev opt-in.** It's in neither the proposal nor the owner's decisions, it sends test diffs to
  a fourth party, and its endpoint has never been exercised. In the pilot, run the Jev lines only
  when a key is set.

## Blocking

### B1. A second audit run wipes the first run's candidates

- **Where:** `repo_audit.py:475-501`. `all_cands` holds only the repositories named in the current
  run, `candidates.jsonl` is opened with `"w"` at :498, and `summary.md` is rewritten at :501. The
  runbook's Phase 1 (and `README.md:17-18`) runs two audits into the same `--out`.
- **Failure:** run `--owner gosha70`, then `--repo django/django --repo grafana/grafana`.
  `candidates.jsonl` now lists only Django and Grafana, so `scrape_changes.py` never sees the
  owner's repositories, and `summary.md` describes only those two.
- **Fix:** build candidates and the summary from every repository already in the store (`raw/` and
  `state.json` keep them), or merge with the existing file. Add a regression test that runs two
  audits into one `--out`.

### B2. `--max-commits 0` means all commits, so public repositories flood candidates with pre-PR history

- **Where:**
  - `repo_audit.py:455-456`: the help text says "0 = all"; the stop test is at :247.
  - `README.md:18` passes `--max-commits 0`, apparently intending "none". The runbook's
    public-repo command omits the flag, which has the same effect.
  - `scrape_changes.py:165` takes every candidate by default.
  - `external_allowlist.example.json:2` lists `django/django` for Anthropic.
- **Failure:**
  - Django has 35,126 commits and Grafana 76,192, so the audit pages through all of them.
  - Django's 11,948 commits before May 2012 predate its GitHub pull requests. Nearly every one
    qualifies as a direct commit with no review signal.
  - Phase 3's unlimited scrape-and-label then makes 2005–2012 Django the bulk of the dataset, at a
    cost of hundreds of dollars in Opus labels.
- **Fix:** add a real way to skip commits (`--no-commits`, or a negative value), plus a per-repo cap
  or date floor for public repositories in `scrape_changes.py`. Until both B1 and B2 are fixed, skip
  public repositories.

### B3. The pilot doesn't run every step (runbook)

- **Where:** runbook Phase 2 stops at `spark/run.sh dryrun`.
- **Failure:** the following first execute after the full labelling spend:
  - `train`;
  - `predict`: `evaluate.py` inside the container through the read-only `/hb` mount, and
    `laya.load("english")` for the base run;
  - `report`;
  - Jev.

  A failure there wastes that spend or forces a fix that changes units and so a relabel.
- **Fix:** end the pilot with

  ```bash
  TRAIN_ARGS="--epochs 1" NAME=pilot spark/run.sh train
  NAME=pilot spark/run.sh predict
  spark/run.sh predict english base
  python evaluate.py report laya=out/pred.laya.jsonl base=out/pred.base.jsonl
  ```

  and run Jev on a handful of units if it stays (W4).

## Warnings

### W1. Rubric changes invalidate labels, and nothing guards against it

- **Where:** `label_changes.py:210-213` stores the teacher, `unit_chars` and `max_units`, but no
  rubric or header version. `build_dataset.py:62-68` rebuilds units with whatever header code is
  current.
- **Failure:** if `questions.json` or `change_header()` changes after labelling, the training states
  and expected answers no longer match what the teacher saw, and nothing fails.
- **Fix:** store a hash of `questions.json` plus a header/unit version on every label row, and make
  `build_dataset.py` refuse a mismatch. Freeze `questions.json` before the pilot; the addendum
  explains why it may change (§1). Consider dropping `criticality` (`questions.json:13-22`): it
  asks about business impact the teacher can't see.

### W2. Review outcomes leak through PR descriptions

- **Where:**
  - `scrape_changes.py:94-97` drops a description only when it was edited after the first *human*
    review.
  - `common.py:175` puts the description's first 300 characters into every unit header.
- **Evidence:**
  - The owner's agent-written PR descriptions record review-loop results, e.g. "DeepSeek: PASS,
    notes only" and "four review rounds on #124".
  - In wiki-studio, code-copilot-team, ai-atlas and memkernel, 67 of 238 recent merged
    descriptions mention the review loop, 8 of them within the first 300 characters.
  - These repositories have no human reviews, so the guard never fires.
- **Fix:** strip review-loop lines (rounds, verdicts, reviewer names) from descriptions and
  commit-message bodies in the prediction view. Add a pilot check that greps `train.jsonl` for them.

### W3. The second teacher differs from the owner's decision

- **Where:** `label_second_teacher.py:32-37` selects every AI-attributed change plus the 200 newest
  unmarked ones; `:73-74` is the OpenAI client.
- **Owner decision (addendum §1):** local Qwen3.8-27B on the Spark labels a random slice plus
  disagreements.
- **Fix:**
  - The script uses chat completions with a strict JSON schema, so pointing `OPENAI_BASE_URL` at the
    Spark's vLLM endpoint may work unchanged; verify vLLM accepts the schema.
  - Add a random-slice mode, and a disagreement mode for cases where Claude's label conflicts with
    observed signals.
  - Record option log-probabilities where the server returns them.
  - The author screen it exists for has a contaminated control group in these repositories. Claude
    markers appear on 0 to 42 of the last 100 default-branch commits, depending on the repository
    (wiki-studio 0, memkernel 0, code-copilot-team 1, ai-atlas 8, mapatlas 37, mileroot 42). Yet
    every review-loop run that names its builder names Claude (`subject_provider: claude`),
    including 61 runs in wiki-studio. So "unmarked" here mostly means Claude without a trailer.

### W4. Jev narrows every backend's score and is an unrequested destination for code

- **Where:** `evaluate.py:290-291` scores only changes that every backend answered.
  `evaluate.py:39` (the endpoint) and `:370` (the model id) have never been exercised.
- **Failure:** if Jev is allowlisted for fewer repositories than Laya, the Laya and base numbers are
  silently restricted to Jev's subset.
- **Fix:** report Laya, base and majority on the full test set, and give Jev its own comparison on
  its subset. Neither the proposal nor the owner's decisions call for Jev, so make it optional.

### W5. Two machines, no data hand-off, missing setup lines (runbook)

- **Where:** runbook, "Before you start".
- **Problem:**
  - It says steps 1–3 run on a laptop and steps 4–5 on the Spark, but Phase 2 already runs
    `smoke` and `dryrun` on the Spark.
  - Nothing copies `data/` into the Spark checkout that `spark/run.sh` mounts
    (`compose.hb.yaml:6`).
  - It omits `pip install anthropic`, `ANTHROPIC_API_KEY` and `TYPESAFE_API_KEY`, which appear only
    in the README.
- **Fix:** run everything on the Spark, or add an explicit copy step; add the missing lines.

## Notes

- **N1. The pilot sample skews the cost estimate.** `label_changes.py:232-233` takes the first N
  cases in file order, which is the alphabetically first repositories. Sample the pilot across
  repositories and diff sizes before multiplying up.
- **N2. Effective size is changes, not units.** The runbook's "at least 1,000 training units" can be
  met by about 100 changes, and units of one change share a header and labels. Also print distinct
  changes and the minority-answer count per question.
- **N3. Taking the maximum over units inflates over-calls on large changes** (`evaluate.py:185-187`).
  At a 5% per-unit over-call rate, a 20-unit change is over-called 64% of the time. Consider adding a
  header-only row per change (state = header, expected = change label), both as a training item and
  as an alternative change-level predictor.
- **N4. Laya's calibration slice includes sibling units.** Laya draws its calibration slice at random
  from training items, so units of changes it trains on sit in the slice, and the fitted
  temperatures come out over-confident. Trust the report's test-set ECE; optionally refit the
  temperature on `val.jsonl`.
- **N5. Teacher settings aren't recorded.** `label_changes.py:106-115` sets no `effort`, so Opus 5.5
  defaults to medium. Set it explicitly and record it with the model id.
- **N6. Deterministic failures are retried, and billed, on every run.** Failures are logged at
  `label_changes.py:207`, and `:227` retries anything unlabelled, so a case that hits `max_tokens`
  (`:110`) fails the same way each time. Cap retries per case.
- **N7. The repository name sits in every header** (`common.py:173`). It lets the student learn
  per-repository priors. That's harmless with time splits, but misleading with `--holdout-repo`
  (`build_dataset.py:79`).

## Agree with

- Pilot first.
- Merge `c3b35ac` before labelling, because header changes force a relabel.
- Tune on `val.jsonl`, read `test.jsonl` once.
- Run two or three seeds.
- Check the `majority` row first.
- The statement that a go/no-go still needs a human-validated gold set plus rule, feature and
  direct-LLM baselines.
