# Private Kaggle pilot

Implementation branch: `codex/kaggle-benchmarks`. Full plan:
[Kaggle migration plan](../../docs/kaggle-migration-plan.md).

This entry point uses the Kaggle SDK for model calls while DUMA controls tool
execution, participant histories, state transitions and scoring. It does not use
the SDK's automatic tool loop. Python 3.11+ is required for the optional SDK.

## Local preparation

```bash
pip install -e '.[kaggle]'
pip install kaggle  # Only needed for CLI catalog, quota and upload commands.
```

The draft manifest lists the existing 35 tasks. Its service-model names are
intentionally null: unavailable or implicit judge/user substitutions must fail
before inference. Create a pinned copy, select two available fast model slugs for
`user.model` and `judge.model`, and set a release-specific `version` before a live
run. Record the exact SDK version and dependency lock for release.

Use `kaggle benchmarks tasks models` to inspect the platform catalog. Local proxy
credentials can expose a smaller catalog than hosted execution. Check both pinned
service models are available in the environment used for the pilot.

## Kaggle notebook setup

1. Install DUMA with the Kaggle extra from the reviewed integration commit.
2. Attach the versioned data bundle and protocol manifest as private inputs.
3. Set `DUMA_DATA_DIR` to the directory containing `duma/`, **before importing DUMA**.
4. Set `DUMA_KAGGLE_PROTOCOL` to the pinned protocol file.
5. Set `DUMA_KAGGLE_OUTPUT=/kaggle/working/duma-kaggle-results`.
6. Load `pilot.py` and execute `duma_pilot.run(kbench.llm)` on one selected target.

The task is ready for a private integration test, not a public leaderboard.
Do not call `tasks publish` during the pilot.

## Hosted bundle

`hosted_pilot.py` is a template, not a file to push directly. Build a fresh
credential-free bundle and a task with the matching archive hash:

```bash
python benchmarks/kaggle/build_bundle.py --owner ivanblch \
  --output .cache/kaggle-pilot-v1
kaggle datasets create -p .cache/kaggle-pilot-v1 -t
kaggle benchmarks tasks push duma-pilot \
  -f .cache/kaggle-pilot-v1/hosted_pilot.py \
  -d ivanblch/duma-bench-private-pilot-runtime
```

The Dataset command is private by default; never add `--public` during the pilot.
**A task push automatically starts a validation run on Kaggle's default model.**
It can therefore consume inference quota even without `tasks run`. The template
executes one dual episode by default, not the full evaluation matrix.

The archive contains only DUMA Python sources, original JSON/Markdown task assets,
package metadata, the dependency lock and the selected protocol. It excludes
credentials, `.env`, Git metadata, caches and experiment results. The hosted task
checks the complete archive hash when an archive is retained. Kaggle normally
extracts uploaded ZIPs: in that case it checks a task-pinned manifest hash and
every listed file hash, then copies only verified assets to the working directory.
Installation needs access to PyPI. Constraints take single-version registry
pins from `uv.lock`, preserve all installed host packages, and include the SDK
in dependency resolution. `pip check` must pass before DUMA inference. The
installed version snapshot is saved as `duma-runtime-environment.json`.

`protocol.pilot.v1.json` fixes a three-task compatibility subset, three seeds,
a 40-step cap and three allowed errors. Protocol v2 pins
`google/gemini-2.5-flash` as user and `google/gemini-3-flash-preview` as judge.
The full SDK keys were verified in hosted execution on 2026-10-19; the CLI
catalog displays shorter names that are not valid keys for `kbench.llms`.
These are Kaggle pilot settings, not the original paper protocol.
Catalog presence does not establish hosted model access or schema compatibility.

Private development resources (account access required):

- [Runtime Dataset](https://www.kaggle.com/datasets/ivanblch/duma-bench-private-pilot-runtime)
- [Pilot task](https://www.kaggle.com/benchmarks/tasks/ivanblch/duma-pilot)

## Results and limitations

Each episode saves the original `SimulationRun`, protocol and data/source hashes,
target identity, seed, assertion breakdowns and usage records for all three roles.
The task returns a Boolean only for a normally completed episode. Incomplete
episodes/API or judge errors raise and leave a technical-error artifact; they are
not treated as confirmed attacks. Report coverage alongside scores.

Artifacts are never overwritten. Automatic resume and atomic concurrent writes
are not yet implemented; use a fresh output directory for another diagnostic run.
Run sequentially until concurrency and checkpoint behavior are validated.

The selected SDK must support one-step `respond(tools=...)`. The adapter handles
both wire dictionaries (PyPI 0.6.1) and typed tool-call records (newer hosted SDK
builds). Interactive streaming mode is intentionally rejected.
Native tool schemas are derived by the SDK from DUMA's callable tools: inspect
provider-specific schema compatibility in the live pilot before a full run.

The local mock-backed tests do not establish model access, quota charging,
private hosted-task visibility, hosted data mounting or leaderboard registration.
These are explicit acceptance checks for the first live Kaggle pilot.
