# Private Candidate Checks: Sonnet 5.5 And GPT-5.5

Date: 2026-10-29, Asia/Tbilisi. The user requested additional models.
Refreshed the live Kaggle catalog and selected two previously unevaluated
targets without changing any existing model labels or results.

| Candidate | Canonical proxy identity | Solo | Dual | Communication |
|---|---|---:|---:|---:|
| Claude Sonnet 5.5 | anthropic/claude-sonnet-5-5@default | 5674221 | 5674234 | 5674238 |
| GPT-5.5 | openai/gpt-5.5-2026-04-23 | 5674223 | 5674235 | 5674239 |

All six single-episode checks were accepted and confirmed Running on existing
private version-1 tasks `duma-v4-r013-preflight-{solo,dual,communication}`.
Frozen runtime 0.1.3, protocol V4 default target reasoning `none`, pinned
user/judge, data/assertions, seed and step/error limits are unchanged.

## Quota Gate

The typed proxy API reported USD 0.38048223 daily and 69.553966605 monthly
remaining before dispatch. This is sufficient for a bounded compatibility
check, not a justified budget for two complete 210-episode model evaluations.
**No full run was scheduled in this turn.** No paid capacity, new Dataset,
public publication, main merge, settings override or monitoring automation.
There is no automatic launch after quota refill; a subsequent active turn must
check readiness and quota before scheduling anything further.

## Follow-Up

Inspect these run IDs, download terminal outputs and executed notebooks to
`.cache/kaggle-sonnet55-gpt55-preflights-20261029/{solo,dual,communication}`.
Verify every candidate's three rows with `verify_results.py`, original V4 and
the frozen wheel/data, including environment replay and communication judge
schema/call counts. Preserve valid policy failures without favorable retries.
Only compatible candidates may be promoted after a fresh quota check. A
successful diagnostic execution is not evidence of a successful security score.

Then schedule affordable full matrices on unchanged `duma-v4-solo` and
`duma-v4-dual`, recording exact run IDs. Keep original incomplete matrices,
post-hoc adjudications, completed-only estimates and the separate Gemma profile
distinct from original complete-matrix scores.
