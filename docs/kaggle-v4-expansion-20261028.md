# Private V4 Target Expansion

Date: 2026-10-28, Asia/Tbilisi. User requested additional models and another
Chinese-model availability check. Everything remains private; no publication,
merge, protocol/source/Dataset changes, paid-capacity purchase or automation.

## New Full Runs

### Completion And Audit

All four expansion runs completed. Combined frozen-wheel audit of all five
targets passed 1,050/1,050 episodes, including 300 NL-scored rows. All ten
matrices have complete coverage and no technical/incomplete episodes. Protocol,
source/data/SDK identities, checksums, environment replay and independent versus
hosted aggregation matched. No model/judge calls were made during the audit;
it is not a manual re-judging of all communication assertions.

| Target | Solo compliance | Dual compliance | Solo / dual failed rows |
|---|---:|---:|---:|
| Gemini 3.8 Flash | 99/105 (94.29%) | 101/105 (96.19%) | 6 / 4 |
| Claude Haiku 5.5 | 99/105 (94.29%) | 94/105 (89.52%) | 6 / 11 |
| Claude Haiku 4.5 | 99/105 (94.29%) | 93/105 (88.57%) | 6 / 12 |
| GPT-5.4 mini | 98/105 (93.33%) | 91/105 (86.67%) | 7 / 14 |
| GPT-5.4 nano | 89/105 (84.76%) | 82/105 (78.10%) | 16 / 23 |

Higher compliance is better. Its complement is assertion failure, not uniformly
confirmed attack success. One-episode differences are not evidence of a
statistically meaningful model improvement. Do not pool V4 with V3/paper runs.

Report: `.cache/kaggle-v4-full-results/verified-five-targets.json`, checksum
`a825c9ad145cd0b82cba562339f0fa180278cb87192b952f2b84c64726250715`.
Preserved all earlier immutable reports. All-role reported evaluation cost is
USD 8.169262785 for five targets, excluding compatibility checks and V3 runs.
Expansion alone cost USD 1.11624886; new daily quota used including diagnostics
was USD 1.13526818 (USD 8.86473182 daily remaining at the final snapshot).

No V4 full run remains active. All resources remain private, and no public
publication, merge or model fallback was performed. Before public release,
review failed trajectories and update the benchmark card/descriptions from
this audited five-target report, then reconcile live visibility/artifacts.

The original scheduling notes below are historical; completion above supersedes
their in-progress status and pending-follow-up instructions.

Selected two inexpensive targets present in the refreshed Kaggle catalog:

| Canonical proxy identity | Reasoning | Solo run | Dual run |
|---|---|---:|---:|
| openai/gpt-5.4-nano-2026-03-17 | none | 5558102 | 5558104 |
| anthropic/claude-haiku-5-5 | none | 5558103 | 5558105 |

All four full runs were confirmed running. Each has 35 tasks x three repeats =
105 episodes; expansion adds 420 episodes. Existing frozen version-1 tasks are
`duma-v4-solo` and `duma-v4-dual`; private Dataset version 3, runtime 0.1.3 and
`protocol.full.v4.json` are unchanged. User/judge and source/data identities are
the same as the 630 audited baseline episodes described in
`kaggle-v4-preparation.md`. No baseline was rerun or relabeled.

Do not add partial expansion rows to the final ranking. Five-target evaluation
requires 1,050 complete, independently audited rows, including 300 NL-scored
rows, not counting any preflights or routing checks.

## Compatibility Gates

All six new-target diagnostics completed and passed strict frozen-wheel audits:

| Target | Solo | Dual | Communication |
|---|---:|---:|---:|
| GPT-5.4 nano | 5557588 | 5557604 | 5557613 |
| Haiku 5.5 | 5557589 | 5557605 | 5557621 |

Reports: `.cache/kaggle-expansion-20261028/preflights/*-verified.json`.
The auditor checked exact model, protocol, source, data, SDK and scoring
identities and replayed environment assertions. Both communication rows include
NL scoring. This is not a new human re-judging exercise. Completed policy-failure
rows were not retried to obtain better scores.

Typed quota API confirmed USD 10 daily / 79.173484375 monthly remaining at the
start. Zero daily use was omitted by JSON serialization, not unknown; it was
read explicitly as `quota_used == 0.0` from the typed response. After diagnostics,
before full dispatch: USD 9.98098068 daily / 79.154465055 monthly remaining.
Diagnostic spend was USD 0.01901932. Unknown usage is never assumed to be free.

## Chinese Model Recheck

Fresh one-tool probes on the original private compatibility task:

| Model | Run | Outcome |
|---|---:|---|
| Qwen3 235B A22B Instruct | 5557636 | HTTP 404, model may no longer be available |
| GLM-5 | 5557637 | HTTP 404, model may no longer be available |
| Qwen3 Next 80B Instruct | 5557645 | HTTP 404, model may no longer be available |
| Qwen3 Next 80B Thinking | 5557646 | HTTP 404, model may no longer be available |
| Qwen3 Coder 480B | 5557647 | HTTP 404, model may no longer be available |
| DeepSeek-R1 0528 | 5557649 | HTTP 400, tool calling not supported |

These used minimal reasoning with the existing side-effect-free ping tool, not
DUMA tasks. Evidence: `.cache/kaggle-expansion-20261028/catalog-chinese`.
The old Gemini validation also appears in downloads; it is not a fresh retry.

Also reran the current-provider-ID routing probe, run 5557648 on
`duma-current-chinese-model-id-probe/1`. All six calls returned HTTP 404 with
explicit model-not-found messages:

- `qwen/qwen3.8-flash`, `qwen3.8-flash`
- `zai/glm-5.3`, `glm-5.3`
- `deepseek-ai/deepseek-flash`, `deepseek-flash`

It directly uses the hosted Kaggle proxy, omits reasoning/temperature, includes
a ping tool, caps output at 128 tokens and disables retries. Its automatic
Gemini task label is only orchestration, never a Gemini or Chinese security
score. Evidence: `.cache/kaggle-expansion-20261028/current-ids`, including
`current-model-id-probe.json`. No external paid provider API was called.

No Chinese model is promoted to full evaluation or assigned a DUMA score.
The old R1 tool restriction is not a claim about newer DeepSeek models.

## Follow-Up

Inspect the four existing full run IDs above, without duplicating them. When
terminal, download outputs and source notebooks to `.cache/kaggle-v4-full-results`
and run the existing auditor with all five canonical target identities, the
frozen 0.1.3 wheel and data. Use a new immutable report name such as
`verified-five-targets.json`; retain all older reports. Inspect technical or
unfinished rows before any retry and never treat them as confirmed attacks.
