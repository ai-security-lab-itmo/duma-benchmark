# Minimal private Kaggle runtime

Work stays on `codex/kaggle-benchmarks`; see the [migration plan](../../docs/kaggle-migration-plan.md).
Private validation does not imply publication or merging.

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
  --output .cache/kaggle-minimal-check --evaluation check \
  --protocol benchmarks/kaggle/protocol.v1.draft.json
```

The bundle includes a deterministic wheel, original JSON/Markdown, protocol,
LICENSE and a hash-pinned task. Credentials, `.env`, caches and experimental
results are excluded. Neither the research pyproject nor its lock is deployed.
Both archived and Kaggle-extracted assets are verified before installation.

The CLI defaults to `check`: `duma-runtime-check` validates installation and
assets without model inference. Its result is readiness, not a security score.

For authorized private updates:

```bash
kaggle datasets version -p .cache/kaggle-minimal-check -t \
  -m 'Minimal runtime readiness bundle'
kaggle benchmarks tasks push duma-runtime-check \
  -f .cache/kaggle-minimal-check/hosted_pilot.py \
  -d ivanblch/duma-bench-private-pilot-runtime
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

The previously selected Google services returned a region-access error for this
account. Catalog presence is not proof of inference access. Packaging does not
remove provider access restrictions.

## Artifacts and scoring

Rows retain target/services, protocol/data/source hashes, SDK version, seed,
trajectory, assertions, outcome and per-role usage. Completed success and policy
failure are distinct from technical errors and incomplete episodes. Unknown
cost stays null, not zero.

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
