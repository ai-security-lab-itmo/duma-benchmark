# DUMA-Bench migration to Kaggle Benchmarks

Status: implementation started on `codex/kaggle-benchmarks`.

The original benchmark and paper results remain available. Kaggle evaluations
will use a separately versioned protocol, initially private. All implementation
changes belong on this branch and reach `main` through review.

## Decisions

- Keep DUMA's environments, policies, assertions and interaction orchestrator.
- Evaluate Kaggle's selected target model; pin a separate fast user model and
  a separate fast communication judge, selected from the available catalog.
- Do not silently replace unavailable service models or omit communication checks.
- Start with the Model Proxy; GPU-hosted open-weight models are a later extension.
- Keep solo and dual-control results separate, with identical task/repeat manifests.
- Preserve original rewards and report technical failures and coverage separately.
- Publish only after completing the private pilot and reviewing its artifacts.

## Milestones and acceptance criteria

| Stage | Status | Deliverable and exit criterion |
| --- | --- | --- |
| 1. Versioned protocol | Draft implemented | Manifest covers all 35 task identifiers, seeds, limits and data fingerprints. Pin live user/judge models and freeze the release configuration after catalog access is available. |
| 2. Single-task adapter | Offline validated | The pilot runs as an actual SDK Task with scripted model responses and real DUMA environments; DUMA controls execution and actor histories. Hosted model/leaderboard registration remains unverified. |
| 3. Model/judge configuration | Implemented | Separate target, user and judge backends; optional Kaggle dependencies; existing LiteLLM defaults retained. Missing, duplicate, foreign or non-Boolean judge decisions raise. |
| 4. Portable data and artifacts | Partially implemented | Original trajectories, assertion results, hashes and per-role usage are saved. Dataset bundling, hosted mounts, atomic writes and resume remain pending. |
| 5. Offline validation | Passed | 151 tests passed, 16 live/inapplicable tests skipped with PyPI SDK 0.6.1. All 10 SDK integration tests also passed against the upstream ci snapshot. |
| 6. Private live pilot | Pending | Three representative tasks, two targets, both regimes, three trials: up to 36 episodes. Inspect trajectories and measure spend before expanding. |
| 7. Full private evaluation | Pending | Cover all eight domains on 2-3 targets, review task compatibility and aggregation weights, and retain per-domain and per-repeat results. |
| 8. Public release | Pending | Review Dataset, notebook, task and collection visibility separately; publish Dual-Control and Solo collections, protocol card, reproduction guide and initial leaderboard. |

## Implementation boundaries

The integration entry points will live under `src/duma/integrations/kaggle/` and
`benchmarks/kaggle/`. Changes to the existing runner and evaluators must be
backward compatible and covered by focused tests. Do not replace DUMA's tool loop
with the SDK's automatic loop without establishing behavioral equivalence.

The current evaluator imports a global judge model. Make the judge configuration
explicit for Kaggle runs, retaining the existing defaults for other callers.

## Evaluation and cost policy

Use the fraction of episodes passing all original checks as the primary score.
Retain ASR/failure-rate reporting with explicit denominators, termination reasons,
and technical-error counts. Never label an API failure as a confirmed attack.
Do not drop failed runs silently or compare scores with different coverage.

Freeze tasks, user/judge models, prompts, decoding settings, repetitions, limits,
code version and dependency versions before adding target models over time.
Store protocol and data hashes with results. Distinct trials must not share a
cached model response. Provider seeds are best-effort, not a determinism guarantee.

Measure pilot cost for agent, user and judge calls. Use bounded concurrency,
checkpoint completed episodes, and resume without rerunning them. Check AI quota
before a live batch. Public leaderboard scoring must match the documented
per-task weighting rather than accidentally equally weighting unequal domains.

## First implementation slice

1. Validate the manifest and resolve pinned user/judge models without guessing slugs.
2. Implement the smallest SDK-compatible adapter supported by the actual SDK.
3. Run a synthetic end-to-end episode and replay its state assertions offline.
4. Export a private-task entry point and instructions for the first live pilot.
5. Record unresolved platform limitations before scheduling or publishing runs.

## Progress: 2026-10-19

- Added `benchmarks/kaggle/protocol.v1.draft.json`, a private pilot entry point and
  setup documentation. Service-model slots remain null rather than guessed.
- Added scoped generation backends, explicit judge configuration and strict
  decision validation. No changes were made to domain policies or assertions.
- Validated both regimes, actual environment tool execution and assertion replay,
  actor-history isolation, communication judging, SDK Task execution and error
  artifacts using scripted model responses. No paid inference was performed.
- Validated wire-format tool calls in PyPI 0.6.1 and typed calls in upstream
  `Kaggle/kaggle-benchmarks` commit
  `43dc1682a41d2679136a368414b0a9971d2e38fd`. Both advertise SDK version 0.6.1.
  Native GenAI tests cover token-limit/tool-choice translation and tool thought
  signature round trips.
- Full offline regression: `151 passed, 16 skipped`; Ruff and `git diff --check`
  passed. One existing dependency deprecation warning is unrelated to scoring.
- Updated `uv.lock` for the optional SDK; `uv lock --check --offline` passed.
  Existing dependency versions were retained, with an additional compatible
  protobuf resolution required by the SDK. The Kaggle extra is Python 3.11+.
- The read-only `kaggle benchmarks tasks models` request timed out after 40
  seconds. Model availability and quota could not be verified; no Dataset or
  hosted task was uploaded or published.

Next: restore catalog access, pin the two service models, prepare private data
inputs and run one hosted episode before scheduling the 36-episode pilot.

## Sources

- [DUMA-Bench preprint](https://arxiv.org/abs/2609.24662)
- [Kaggle Benchmarks SDK guide](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/user_guide.md)
- [Kaggle Benchmarks CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md)
- [Kaggle Benchmarks documentation](https://www.kaggle.com/docs/benchmarks)
