# Private V4 Target Expansion

Date: 2026-10-28, Asia/Tbilisi. User requested additional models and another
Chinese-model availability check. Everything remains private; no publication,
merge, protocol/source/Dataset changes, paid-capacity purchase or automation.

## New Full Runs

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
