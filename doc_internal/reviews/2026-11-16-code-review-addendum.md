---
title: "Code review addendum: owner decisions and data the pipeline doesn't use yet"
date: 2026-11-16
reviewer: Claude Code (Opus 5.5)
companion: 2026-11-16-pilot-runbook-review.md (code review of c3b35ac and the pilot runbook)
---

# Addendum: owner decisions and data the pipeline doesn't use yet

This is context the builder session didn't have. Decisions are the owner's answers given on
2026-11-16; numbers come from read-only audits run that day, against GitHub and against the owner's
local clones under `~/dev/repo`.

## 1. Owner decisions (2026-11-16)

| Topic | Owner's answer | Pipeline at `c3b35ac` | Gap |
|---|---|---|---|
| Stages | "The plan/build classifier must be used in different stages of AI SDLC: during the planning (identify critical tasks - required explicit review of human and/or carries security, ux, performance concerns); during the build - based on implementation to evaluate the changes, etc." | Build stage only (code changes) | Add plan-stage task cases (§2.3) |
| Targets | "Both targets": change-level routing, plus triage of AI review findings (actionable or not) | Change-level only | Add finding cases (§2.2) |
| Teachers | "Claude + local Qwen": Claude Opus 5.5 labels everything through Batch, private code after secret scanning; Qwen3.8-27B on the Spark labels a random slice plus disagreements | Claude, plus an OpenAI author-bias screen | Swap the second teacher (review W3) |
| Validation | "~40 h, ~400 cases" of owner adjudication | None ("labels are teacher-inferred") | Add a validated set, and score against it |

The plan-stage wording maps onto four existing questions in `questions.json`: `human_review`,
`security_review`, `ux_review` and `performance_review`. Settle any change to `questions.json`
before the pilot (review W1).

## 2. Data facts that change the plan

### 2.1 Human review signal is near zero in the owner's repositories

| Signal | Count |
|---|---:|
| Non-fork repositories / without any PR | 29 / 16 |
| PRs (Dependabot / owner / others) | 1,023 (189 / 821 / 13) |
| PRs reviewed by someone other than the owner | 12 |
| PRs with "changes requested" / with inline review threads | 3 / 28 |
| Revert PRs | 0 |
| PRs with CodeQL review alerts | 17 |
| Human PRs with a failing check on some commit / fixed red-to-green before merge | 217 / 116 of 828 |

`mileroot` and `hookatlas` have chronically red CI (56 and 13 PRs with failures, 2 recovered in
each), so a failure there carries no signal.

**Implication:** on these repositories, Claude's labels are almost entirely inference, and the
outcome view is mostly CI. The observed signal is in §2.2.

### 2.2 Local review-loop logs: the richest observed signal, and the scraper can't see them

**Where:** untracked `.cct/` directories in the local clones of ai-atlas, wiki-studio, rlmkit and
code-copilot-team. They aren't committed, so they aren't on GitHub, and they exist on one disk:
**back them up first.**

| | |
|---|---|
| Review runs / rounds | 96 / 161 (wiki-studio 61 runs, ai-atlas 19, code-copilot-team 8, rlmkit 8) |
| Unique findings | 1,731: blocking 53, warning 821, note 857 |
| Categories | correctness 1,012 · testing 264 · documentation 249 · design 157 · style 29 · security 12 · performance 8 |
| Round verdicts | PASS 98 · FAIL 56 · INCONCLUSIVE 7 |
| Reviewer, by round | DeepSeek 134 · Codex 25 · Spark 2 |
| Findings with a final disposition | 228: fixed 140 (138 name the fix commit) · disputed 43 · not-applicable 33 · deferred 12 |

File shapes:

- `findings-round-N.json`: `round`, `verdict`, `reviewer_provider`, `invocation_cost_usd`,
  `raw_output`, and `findings[]`, where each finding has `id`, `category`, `severity`, `file`,
  `line_hint`, `description`, `suggested_fix`, `disposition` and `first_seen_round`.
- `resolution-round-N.json`: `round`, and `resolutions[]` with `id`, `disposition`, `rationale` and
  `commit_ref`.
- `state.json`: `feature_id`, `phase`, `subject_provider`, `peer_provider`, `review_scope`,
  `review_specialization`, `target_ref` (the branch), `last_verdict` and per-finding history.

How to use them:

- **Labels for finding triage.** `fixed` means actionable; `disputed` and `not-applicable` mean not
  actionable; leave out `deferred`.
- **Evaluation.** That gives about 216 labelled findings. That is small, so use cross-validation
  grouped by review run; a frozen-encoder head (Laya `vegaml=True` with `fit()`) may suit better
  than a fine-tune.
- **Provenance.** The builder agent sets the dispositions, so keep these labels in their own
  AI-observed tier, separate from teacher labels and from human validation.
- **Reconstructing the reviewed revision.** `state.json` names the branch and resolutions name the
  fix commit, but I didn't see a per-round base/head SHA. Verify how to recover the exact diff each
  round reviewed before relying on it.
- **Phase coverage.** Every review run is build-phase; none reviews a plan.

### 2.3 Plan-stage inputs already exist

| SDD artifact in the owner's local clones | Count |
|---|---|
| `specs/*/plan.md` with frontmatter | 183 |
| … `risk_category` | integration 58, feature 32, security 17, bugfix 14, ui 9, infra 7, privacy 7, schema 5, data 4, other 13, missing 17 |
| … `spec_mode` / `status` | full 132, lightweight 40, none 9 / approved 110, draft 71 |
| Task rows in `specs/*/tasks.md` tables | 1,092 (962 name a file path), plus checklist-style tasks in mileroot tagged `[gate]`, `[manual+CI]` and similar |

How to use them:

- **Outcome links.** Tasks name files, and review findings name files and carry `feature_id`.
  Within one feature, that gives task-level outcome evidence: findings on the task's files, and CI
  on the implementing PRs.
- **Prediction view.** `tasks.md` and `plan.md` get edited after implementation (done marks,
  notes). Take each file's version from git history as of before its feature's first implementation
  commit.
- **Fit.** Task text is short, so a plan-stage case fits Laya's 512-token English checkpoint whole,
  with no units.

### 2.4 Diff sizes

Over 45 sampled merged PRs (code-copilot-team, memkernel, wiki-studio, mapatlas; tokens ≈ characters
/ 4), the median diff is about 6,000 tokens, p75 about 14,900 and p90 about 19,600. The unit design
is the right response. Confirm in the dry run that units at 2,250 characters plus the question head
fit in `MAX_LEN=1024` without truncation.

### 2.5 The second-teacher infrastructure is already running

The Spark serves Qwen3.8-27B (NVFP4) and Qwen3-Coder-Next through vLLM, and the review loop has
used it as a reviewer ("spark" above). The setup is documented in the public repository
gosha70/code-copilot-team, under `docs/dgx-spark/runbook-qwen38.md` and `vllm-qwen38.md`.

| Setting | Value |
|---|---|
| API | OpenAI-compatible chat completions at `http://<spark>:8001/v1`; avoid the Responses API on this server |
| Model id | `qwen38-27b` |
| Context | 262,144 tokens |
| API key | Ignored by vLLM, but required by clients: use a dummy value |
| Reasoning | `--reasoning-parser qwen3`: reasoning is returned separately, and `message.content` holds the answer |

`label_second_teacher.py` already speaks chat completions with a strict JSON schema; code review W3
lists the changes it needs.

### 2.6 Authorship markers under-count agent-written changes

`written_by` keys on commit trailers, but the owner's repositories often have none. Of the last 100
default-branch commits, these carry a Claude marker: wiki-studio 0, memkernel 0, code-copilot-team
1, ai-atlas 8, mapatlas 37, mileroot 42. Meanwhile every review-loop run that names its builder
names Claude: 96 runs, 61 of them in wiki-studio.

So `unmarked` mostly means "Claude without a trailer". Don't read the by-author slices of the report
or the teacher screen as AI vs human. For these repositories, the review-loop `subject_provider` is
the better authorship source.

### 2.7 Privacy

Anthropic's Batch API isn't covered by zero data retention (it keeps batch data for 29 days). The
owner accepted Batch for private repositories after secret scanning. `common.py:76-95` redacts
common key formats only, as the README already says; keep treating it as a safety net.

## 3. Suggested order

1. **Before labelling.** B1–B3 are fixed in `99cdcd9`. Still to do: freeze `questions.json` and
   guard unit drift (W1); strip review-loop text from prediction views (W2); and make the Jev lines
   of the pilot optional (re-review, B3).
2. **Pilot end to end** on your own repositories, including train, predict and report.
3. **New case kinds.** Add plan-stage task cases and finding cases, sharing `questions.json` where
   the questions apply.
4. **Second teacher.** Point it at local Qwen, using a random slice plus disagreements.
5. **Validation and baselines.** Build the ~400-case validated set across build, plan and finding
   cases, and add rule, feature and direct-LLM baselines.
