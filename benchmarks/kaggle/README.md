# Minimal private Kaggle runtime

Work stays on `codex/kaggle-benchmarks`; see the [migration plan](../../docs/kaggle-migration-plan.md).
Private validation does not imply publication or merging.

## Hardened revision (2026-10-24)

Domain hardening from `73c439c` and scoring/policy fixes are now in `main`
(`09401e1`, `da2723a`, plus negation regression fix `fd6972d`) and merged into
this branch. No experiment dumps or
legacy runner/provider changes were imported. Use `protocol.full.v3.json`
or the three-task `protocol.pilot.v6.json` for these sources. The task count
remains 35 across eight domains; five task IDs changed and ten tasks now have
communication scoring. The pinned user/judge and generation settings are
unchanged from full-v2. Full-v3 is 105 episodes per target and regime.

V3 pins every task/data/prompt file hash and adds a bundle revision manifest.
The builder rejects historical protocols with hardened data. Historical v1/v2
results, protocols and fixtures are preserved; reproducing their original runs
requires the frozen checkout/bundle, not this branch's new domain classes.
`tests/fixtures/kaggle-legacy-data.zip` is a historical data test fixture from
`39bb64b`, not a replacement for that original runtime.

Private full and pilot V3 Datasets have been uploaded with MIT terms and upstream
NOTICE. Hosted readiness passed (run 5442610, SDK 0.6.2). The final package passed
dual, solo and communication-judge preflights (runs 5444068, 5444177 and 5444232);
their artifacts passed offline replay and assertion checks. The private 36-episode
pilot is in progress. Full evaluation remains gated on its audit. Do not reuse
v1/v2 checkpoints or append new results to their leaderboards.
Infra over-quota checks include blocked attempts, not just executed harm.

Offline validation: research environment with Kaggle SDK, 281 passed and 16
live tests skipped; installed minimal wheel without SDK/full DUMA, 141 passed,
one optional build-tool test skipped and one legacy-runner comparison deselected.
All 35 bundled tasks load in both modes, with ten communication-scored tasks.
Clean SDK installation validation passed on the hosted runtime.
See [release preparation](../../docs/kaggle-v3-release-preparation.md),
[benchmark card draft](BENCHMARK_CARD.md) and [publication checklist](PUBLISH_CHECKLIST.md).

## Architecture

`duma-kaggle-runtime` contains shared domain sources, orchestration, models,
evaluators and the Kaggle adapter. It excludes the research runner, CLI, servers,
analytics, LiteLLM and Langfuse. Its four direct dependencies are Pydantic,
docstring-parser, typing-extensions and Loguru. Do not install alongside full
`duma`: both distributions contain the same Python namespace.

The host supplies Kaggle SDK >=0.6.1,<0.7. Bootstrap checks compatibility and
installs the wheel and missing small dependencies with `--no-deps`; it never
upgrades SDK/provider packages. An incompatible host fails explicitly.
Restart the notebook kernel before loading another bundle. Bootstrap refuses
an already imported DUMA namespace and reinstalls only the selected wheel.

## Build

Install `pdm-backend` in the local build environment only:

```bash
python benchmarks/kaggle/build_bundle.py --owner ivanblch \
  --output .cache/kaggle-v3-check --evaluation check \
  --protocol benchmarks/kaggle/protocol.full.v3.json
```

The bundle includes a deterministic wheel, original JSON/Markdown, protocol,
LICENSE and a hash-pinned task. Credentials, `.env`, caches and experimental
results are excluded. Neither the research pyproject nor its lock is deployed.
Both archived and Kaggle-extracted assets are verified before installation.

The CLI defaults to `check`: `duma-v3-runtime-check` validates installation and
assets without model inference. Its result is readiness, not a security score.

For authorized private updates after creating the new private v3 Dataset:

```bash
kaggle datasets version -p .cache/kaggle-v3-check -t \
  -m 'Minimal runtime readiness bundle'
kaggle benchmarks tasks push duma-v3-runtime-check \
  -f .cache/kaggle-v3-check/hosted_pilot.py \
  -d ivanblch/duma-bench-private-v3-runtime
```

**Each task push automatically starts validation.** Never push an inference
profile until model access and quota have been checked. Do not use `--public`
or `tasks publish` during private development.

## Profiles and protocol

- `check`: installation/assets only, no model calls.
- `smoke`: `duma-pilot`, one dual collab episode.
- `solo` / `dual`: separate tasks executing the full protocol row matrix
  sequentially; a complete-matrix check gates the returned leaderboard score.

The full draft has 35 tasks, seeds 300--302, 100 steps and 10 errors. Its service
models must be pinned before inference. The diagnostic pilot has three tasks and
shorter limits, a different protocol. SDK model keys include provider prefixes.
Do not silently substitute unavailable services or omit communication checks.

The earlier private pilot used `protocol.pilot.v4.json`: three tasks, two regimes,
seeds 300--302, 100 steps and 3 errors. Earlier v3 used 40 steps. Target reasoning
is `none` for both Gemini 3.8 Flash and GPT-5.4 mini; user/judge settings are
unchanged from `protocol.pilot.v1.json` (protocol version v2). Keep protocols and
source hashes separate; old results are not overwritten by the new pilot.
OpenAI function tools with `reasoning=low` were rejected by the hosted Chat
Completions endpoint; v3 was explicitly approved instead of silently changing
generation settings. The earlier Google region-access error no longer prevents
the tested calls. Catalog presence alone is still not proof of inference access.

Task pushes execute the full selected profile on Kaggle's default target as
validation. An incomplete matrix deliberately raises without a leaderboard score,
which may leave the task in `ERRORED`. CLI 2.2.4 refuses to schedule other models
in that state; the official `batch_schedule_benchmark_task_runs` API can schedule
an independent target run without repeating validation. This does not change
the incomplete matrix or authorize assigning it a score.

## Artifacts and scoring

Kaggle-only v2 preparation uses `protocol.full.v2.json` / `protocol.pilot.v5.json`.
The builder applies a hash-manifested revision to generated assets only; v1 data
and assertion semantics remain intact. Revised infra policy, field-aware identifier
checks and explicit marketing campaign IDs are documented in
[v2 preparation](../../docs/kaggle-v2-preparation.md). Private v2 hosted readiness
passed (run 5401810, SDK 0.6.2) without model calls. Full-v2 subsequently completed
all 420 rows: see [verified compliance results and interpretation limits](../../docs/kaggle-full-v2-results.md).
The readiness task's 1.0 is only an installation indicator. Dataset API metadata
still reports license `unknown` despite MIT in the uploaded metadata and LICENSE;
resolve before any public release.

**Full-v1 release hold:** offline replay matched all 420 selected completed
episodes, but audit found two false secret-output failures and infra policy/scoring
gaps. Preserve raw results; do not promote them as a validated public safety ranking.
See [audit and release gates](../../docs/kaggle-full-v1-audit.md) and
[frozen run identities/results](../../docs/kaggle-full-v1-results.md).
`audit_full_v1.py` verifies private artifacts without inference and can stage local
resume inputs. It does not publish, upload or repair scores.

Rows retain target/services, protocol/data/source hashes, SDK version, seed,
trajectory, assertions, outcome and per-role usage. Completed success and policy
failure are distinct from technical errors and incomplete episodes. Unknown
cost stays null, not zero.

The unscored trajectory is captured before evaluation, so a judge parse/schema
error retains the completed conversation and the last raw judge response in the
technical-error checkpoint. Judge parsing accepts an exact outer Markdown JSON
fence, but does not repair JSON or relax schema, boolean or assertion checks.

Checkpoints are checksummed and atomically replaced under exclusive locks.
Resume reuses verified completed episodes only; retried failures are archived in
`attempts/`, reports in `summaries/`. Corrupt checkpoints fail closed. A crash may
leave `.lock`: verify no runner is active before removing it manually; there is
no automatic stale-lock takeover.

Scoring weights tasks equally after averaging repeats, not domains equally.
Incomplete matrices have no final score; coverage and errors remain visible.
`pass_power_k` is the existing pass^k, not at-least-one-success pass@k.

Private resources: [Dataset](https://www.kaggle.com/datasets/ivanblch/duma-bench-private-pilot-runtime),
[original pilot](https://www.kaggle.com/benchmarks/tasks/ivanblch/duma-pilot).
