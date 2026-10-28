# Private Full Gemma Evaluation

Date: 2026-10-28, Asia/Tbilisi. The user explicitly requested full Gemma runs
after the three verified diagnostics. No public publication or main merge.

## Dispatched Runs

| Regime | Private task | Task version | Gemma run | Episodes |
|---|---|---:|---:|---:|
| solo | duma-gemma-default-solo-v1 | 2 | 5646114 | 105 |
| dual | duma-gemma-default-dual-v1 | 2 | 5646169 | 105 |

Both were confirmed Running. Target CLI slug `gemma-4-31b-it`, canonical proxy
identity `google/gemma-4-31b`. No final score is available at dispatch.

## Explicit Separate Profile

Profile `duma-kaggle-full-gemma-default-v1` includes the full frozen V4 protocol
and its hash, target identity, adapter hash and transport choices. Its canonical
JSON hash is stored in each artifact as `protocol_hash`. It is not V4:

- Target reasoning is omitted entirely: provider default, not verified disabled.
- Solo receives a transport-only user dot after system messages and before the
  first assistant. The same prefix is reconstructed on every replay; actual
  participant histories and system policies are not modified.
- Target temperature 0 and maximum 4,096 output tokens; seeds 300--302;
  maximum 100 steps and 10 errors. All 35 task definitions, tools, initial states,
  policy text, scoring bases and pinned user/judge remain unchanged.

Reuse the existing private Dataset `ivanblch/duma-bench-private-v4-runtime`,
version 3, minimal runtime 0.1.3. No new Dataset or modified wheel was uploaded.
The bootstrap verifies the frozen archive/manifest and data before inference.

Source `benchmarks/kaggle/gemma_full.py`; generator
`benchmarks/kaggle/build_gemma_hosted.py` combines this adapter with the existing
verified bootstrap. Frozen identities:

- Adapter SHA256: `065d3a64bc54403be78b0534a7ebea244b4c2e240b59df8af970104213abc2b1`.
- Solo generated source: `9eb001824198af148a86cbe6829da21d2fdaa3bcf0054912dd8636222938ef1e`.
- Dual generated source: `b50c3a63a08830a02bbdbbfdaf5d38c7ed1b59d871a9fed948b32c25d1d5ef68`.

Generated files: `.cache/kaggle-gemma-full-v1/{solo,dual}.py`.

## Artifacts And Scoring

Output directory `duma-gemma-full-results`, deliberately separate from frozen
V4 `duma-kaggle-results`. Each atomic checkpoint includes profile/source/data/SDK
identity, row and seed, simulation, assertions, terminal status and role usage.
Profile identity includes the adapter, so completed artifacts cannot be reused
under changed transport code. Existing completed policy failures are reused,
not resampled; unfinished saved rows require an explicit retry decision.

The original task-balanced aggregator computes compliance, domain scores,
`pass^k` and available role costs. An incomplete matrix has no score. The Kaggle
task returns a dictionary with the completed matrix score and explicit profile;
it returns no score for automatic non-Gemma readiness runs. This profile is not
yet integrated into a public leaderboard or pooled with V4.

The initial task version 1 failed automatic preparation because Kaggle selected
its default Gemini readiness model. No Gemma episodes were scheduled on that
version. Version 2 explicitly skips non-Gemma readiness without inference or a
synthetic safety score. Only version 2 was used for actual full Gemma runs.

## Checks And Budget

- Seven focused offline tests passed (profile and existing metrics).
- Full DUMA regression: 296 passed, 16 skipped, one upstream deprecation warning.
- Python compilation passed for both generated hosted sources.
- Ruff passed with F821 excluded for the two intentionally builder-injected
  constants; no production runtime source changed.
- Free proxy quota before dispatch: USD 1.93431707 daily and 71.107801445 monthly.
  No paid capacity purchased. Quota limits can still interrupt a run.

## Next Check

Inspect run IDs above, download their raw outputs and executed notebooks, then
independently verify the complete 210-row matrix. Check the separate profile
and adapter identity, original ENV replay, NL assertion schema/call counts,
usage and independent aggregation. Do not pass these artifacts through the
frozen V4 auditor or label provider-default reasoning as `none`.

Related completed V4 results are in `kaggle-v4-next-models-20261028.md`; the
Flash-Lite dual remains incomplete and has not been silently retried.
