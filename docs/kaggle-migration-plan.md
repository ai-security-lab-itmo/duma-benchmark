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
  hosted task had been uploaded or published at that point.

## Private hosted draft: 2026-10-19

- Created version 1 of the private task through the official CLI:
  https://www.kaggle.com/benchmarks/tasks/ivanblch/duma-pilot/1.
  The status endpoint confirmed `Public: False`.
- The initial task-list request returned HTTP 500, but the subsequent push and
  status requests succeeded. Avoid treating the earlier timeout as an
  authentication failure.
- A push automatically started a validation run on `gemini-3.8-flash`
  (run 4950707), without a separate `tasks run` command. Future pushes must be
  treated as potentially executing notebook code, not just saving a draft.
- Hosted validation failed at import with `ModuleNotFoundError: No module named
  'duma'`, before DUMA inference. This is not a completed pilot or a score.
  Package installation, data attachment and pinned user/judge configuration
  remain required before the next hosted version. No task was published.

## Portable runtime prepared: 2026-10-19

- Retrieved the account model catalog successfully. Pinned the compatibility
  pilot user to `gemini-2.5-flash` and judge to `gemini-3-flash-preview` in a
  separate three-task protocol; the full 35-task draft is unchanged. Hosted
  access and suitability still need validation.
- Added a deterministic allowlist-based bundle builder and hash-pinned hosted
  bootstrap. The 516 KiB archive includes source code, original task data,
  package metadata, dependency lock and pilot protocol, excluding credentials,
  caches and experiment results. Bootstrap installs DUMA before importing it
  and sets the explicit data/protocol paths.
- Added bundle reproducibility, file-integrity, credential-exclusion and pilot
  manifest tests. Full offline regression: `153 passed, 16 skipped`; Ruff passed.
- Private Dataset upload was denied by the environment pending explicit approval
  for uploading this code/data payload to Kaggle. Asked the user to approve
  `ivanblch/duma-bench-private-pilot-runtime`. No Dataset was created, no new task
  version was pushed, and no new inference run was requested in this step.

## Private Dataset and hosted validation: 2026-10-19

- Received explicit user approval for the code/data upload and created the
  [private runtime Dataset](https://www.kaggle.com/datasets/ivanblch/duma-bench-private-pilot-runtime).
  The Dataset status endpoint returned `ready`.
- Attached Dataset version 1 to pilot task version 2. Kaggle extracts uploaded
  ZIPs into a `duma-pilot/` directory, so validation run 4951075 stopped before
  inference because the initial bootstrap expected an archive.
- Added extracted-Dataset support with a task-pinned manifest hash, per-file
  verification and allowlisted copying. The original archive hash remains
  `4fed5a17ef60b4de91d8efa7a409e8d66f47adfb795aa7678575328e25ce4858`;
  the Dataset contents did not change.
- Pushed corrected task version 3; automatic validation run 4951322 loaded DUMA
  and all eight domain registries. Downloaded artifacts show an explicit
  pre-inference error: CLI model names omit provider prefixes required by SDK
  dictionary keys. No episode score was produced.
- Full offline regression after both asset-layout tests: `155 passed, 16 skipped`.
  Ruff passed. No task or Dataset was made public.

- Server metadata confirmed Dataset `isPrivate: true`. Its license metadata
  currently reads `unknown`, despite the included MIT LICENSE; reconcile this
  without changing the original licensing before any public release.
- The initial broad pip installation upgraded protobuf outside the host SDK's
  declared range. Updated bootstrap constraints to use single-version pins
  from the repository lock while preserving installed host versions, include
  the SDK in dependency resolution, and require `pip check` before inference.
  Save the installed version snapshot alongside outputs.
- Uploaded Dataset version 2 with protocol v2 using verified SDK model IDs
  `google/gemini-2.5-flash` and `google/gemini-3-flash-preview`. Bundle SHA256:
  `63fde0e43c96fb8daebaef45e646edccdec5654e520b3a6a00bb80fcbbc5517a`.
  Task version 4 is under automatic validation. Full offline regression:
  `156 passed, 16 skipped`.

Next: verify hosted installation, inspect the first episode and role-level
spend, then validate communication judging and both regimes before scheduling
the 36-episode pilot.

## Sources

- [DUMA-Bench preprint](https://arxiv.org/abs/2609.24662)
- [Kaggle Benchmarks SDK guide](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/user_guide.md)
- [Kaggle Benchmarks CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md)
- [Kaggle Benchmarks documentation](https://www.kaggle.com/docs/benchmarks)
