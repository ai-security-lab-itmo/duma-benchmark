# Private GPT-OSS Full Dual Dispatch

Date: 2026-10-31, Europe/Moscow. The user authorized launching GPT-OSS after
the compatibility repair was independently validated on a complete XSS episode.

## Fixed Scope

- Target: `openai/gpt-oss-120b`, CLI `gpt-oss-120b`.
- Dual only, all 35 hardened tasks, seeds 300/301/302: **105 fresh episodes**.
- Protocol `duma-gptoss-low-named-descriptions-v1`, fingerprint
  `7792ba790baee00c290b213c431203b4f969241e25e174a44764326c53dc0315`.
- Target reasoning low, temperature 0, maximum 4096 output tokens; only empty
  tool descriptions are replaced with their existing function names.
- Original user/judge, policies, tasks, 100-step/10-error limits and assertions.
- Separate from frozen V4 and from any provider-aware post-hoc scoring view.
- No solo run: the required-tool API is unsupported. No public publication,
  paid top-up or main-branch merge.

The preflight is validation only, not copied into this evaluation. This is a
new full-matrix phase, not a selective replacement of preflight outcomes.
No full-matrix result is claimed before downloading and independently auditing
all expected artifacts.

## Private Task And Provenance

Task `duma-gptoss-compatible-dual-v1/1` attaches only the existing private
`ivanblch/duma-bench-private-v4-runtime` Dataset, unchanged version 3/runtime
0.1.3. No new Dataset or raw trajectory upload.

Generated source `.cache/kaggle-openweights-20261031/gptoss-full-dual.py`, SHA256
`736876fc6e44c76f66118329bb990024d372a044f9d1ced1969918bbf38969e1`.
The builder includes unchanged compatibility definitions, verified against
adapter hash `2014678aa72d6a856f0540ef158b19bd4660744c24de30d0e062000143bd62ca`.
The new matrix wrapper does not alter the measured generation profile.

## Execution And Verification

`gptoss_full.py` uses the existing `run_episode` and `aggregate`. It records
original/effective target-request hashes and finish reasons per episode,
retains original role usage/cost, preserves completed score-0 checkpoints, and
restores the client/backend in finally blocks. On HTTP 401/403/404/429 observed
from any generation role, subsequent episodes are not attempted. An unfinished
matrix has no final score. Existing unfinished checkpoints require an explicit
retry selection, not automatic resampling. Existing wire logs survive resume
of completed rows.

Two offline tests cover all 105 rows, completed failures, no extra API calls on
resume, wire-log preservation, quota early stop, client/backend restoration,
and rejection of implicit unfinished-row retries.

The live typed quota before launch was daily USD **9.995414345**, monthly
USD **55.710584385**. Availability was checked from the API rather than assumed
from a calendar reset.

## Dispatched

Actual gpt-oss run **5763534** was submitted and confirmed **RUNNING** by the
typed Kaggle API. Automatic Gemini readiness skipped without inference/score.
The task is private, version 1. No full results have been downloaded or scored
at this dispatch checkpoint. Final regression: **344 passed, 16 skipped**;
Ruff passed for the new wrapper/tests (builder-injected globals excluded).
