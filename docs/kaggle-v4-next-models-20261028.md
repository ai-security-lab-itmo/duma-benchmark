# Private V4 Next Models

Date: 2026-10-28, Asia/Tbilisi. User requested continuing evaluations on other
models. Existing five-target results remain audited and immutable (1,050 rows;
`verified-five-targets.json`). No publication, merge, new Dataset upload, runtime
change, protocol change, model substitution inside a labeled run or automation.

## Active Full Runs

| Canonical proxy target | Reasoning | Solo run | Dual run |
|---|---|---:|---:|
| openai/gpt-5.4-2026-03-05 | none | 5620785 | 5620787 |
| google/gemini-3.6-flash | none | 5620786 | 5620788 |
| google/gemini-3.1-flash-lite-preview | none | 5620810 | 5620811 |

All six runs confirmed running on existing private frozen version-1 tasks
`duma-v4-solo` and `duma-v4-dual`. Each has 105 rows; this adds 630 episodes.
Runtime 0.1.3, Dataset version 3, V4 protocol, seeds 300--302, user/judge,
step/error limits and assertion semantics are unchanged. Both target identities
use the protocol's default none reasoning, with no new overrides.

Free quota before dispatch: USD 8.75686677 daily / 77.930351145 monthly.
Measured diagnostic costs and earlier full-run costs informed this bounded
two-model expansion; no paid top-up was purchased. Remaining quota must be
rechecked before further model scheduling.

Flash-Lite was added later after its explicitly authorized diagnostic retry
(details below); free quota before its two full runs was USD 7.20432382 daily /
76.377808195 monthly. Other full matrices were still active and were not changed.

## Passed Compatibility Gates

All six replacement preflights completed and passed strict frozen-wheel audit:

| Target | Solo | Dual | Communication |
|---|---:|---:|---:|
| GPT-5.4 | 5620755 | 5620760 | 5620768 |
| Gemini 3.6 Flash | 5620757 | 5620762 | 5620770 |

Raw evidence and immutable reports:
`.cache/kaggle-v4-next-models-20261028/replacement`.
The auditor checked source/data/SDK/model/protocol identities, checksums,
environment replay and task scoring. Both communication rows include NL
scoring; this is not new manual re-judging. Valid score-zero rows were not
retried to obtain better scores.

## Initial Candidates Not Promoted

Subsequent explicit Gemma parameter experiments found a working diagnostic
adapter (omit reasoning, stable solo user dot), with 3/3 actual task episodes
audited. See `kaggle-gemma-experiments-20261028.md`. These are separately labeled
experiments, not V4 rows; the frozen full tasks were not changed and no full
Gemma evaluation was launched. The original rejection records below remain valid
for their original configuration.

Originally selected Gemma 4 31B and Gemini 3.1 Flash-Lite Preview to broaden the
pool. Their original diagnostics were preserved, not repaired or relabeled:

| Target | Solo | Dual | Communication |
|---|---:|---:|---:|
| gemma-4-31b-it | 5620710 errored | 5620712 errored | 5620714 errored |
| gemini-3.1-flash-lite-preview | 5620711 completed | 5620713 errored | 5620715 completed |

Gemma solo was rejected with HTTP 400 `GenerateContentRequest.contents: contents
is not specified` on its system-only start. Dual/communication logs show HTTP
400 `Thinking budget is not supported for this model`. These are compatibility
failures with current transport/settings, not evidence of security failure.

The initial Flash-Lite dual ended after an empty assistant response with neither content nor
tool calls (`AssistantMessage must have either content or tool calls`). Do not
generalize this one observed failure into a claim that the model never supports
dual/tool use. At this initial stage no full matrix was scheduled and no
favorable resampling occurred; the later explicit technical retry is below.

Evidence, including executed notebooks:
`.cache/kaggle-v4-next-models-20261028/initial`.
Future support for these models may require a separately versioned transport or
configuration; the frozen runtime was not patched during this expansion.

## Explicit Flash-Lite Technical Retry

The user explicitly requested retrying Lite. Repeated only the failed dual
diagnostic, not its valid solo/communication rows: fresh run 5620798 completed.
All three episodes then passed strict frozen-wheel audit with unchanged
protocol/data/settings (solo 5620711, fresh dual 5620798, communication 5620715).
Reports are `.cache/kaggle-v4-lite-retry-20261028/{solo,dual,communication}-verified.json`.
The old failed dual 5620713 remains preserved under the initial evidence path.

This demonstrates completion on retry, not proof that the empty-response fault
cannot recur. There was no policy-failure resampling. After all three audits and
a fresh budget check, launched the first full Lite solo 5620810 and dual 5620811,
105 episodes each, under the same V4 source/protocol. No Lite score is final
until its full matrices complete and pass audit; retain any subsequent technical
failures separately from security outcomes.

## Follow-Up

Inspect existing full run IDs, then download terminal results/source notebooks
to `.cache/kaggle-v4-full-results`. Audit with all eight canonical target IDs,
the unchanged V4 protocol, installed 0.1.3 wheel and extracted frozen data. A
complete combined evaluation requires 1,680 rows and 480 NL-scored rows. Use a
new immutable report such as `verified-eight-targets.json`; do not overwrite
the earlier three/five-target reports or present partial matrices as final.

On technical/incomplete rows, inspect original diagnostics before any retry;
never count such rows as confirmed attacks. No public release is authorized by
this evaluation request.
