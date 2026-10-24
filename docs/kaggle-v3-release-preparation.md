# Hardened V3: Private Validation And Publication Gates

## Authorization And Scope

The user authorized autonomous private preparation up to publication on
2026-10-24. Do not publish tasks, notebooks, datasets or collections yet.
All changes remain on `codex/kaggle-benchmarks`; the domain hardening and
scoring fixes were already merged into main separately.

Order: hosted readiness, one hosted dual episode, solo/communication checks,
pilot-v6 (3 tasks x 2 regimes x 3 seeds x 2 targets = 36 episodes), artifact and
scoring audit, then full-v3 (35 tasks x 2 x 3 x 2 = 420 episodes) and release audit.
The full run starts only after the pilot passes execution/scoring validation.
Valid policy failures are results, not grounds to change tasks or retry.

Models: targets Gemini 3.8 Flash and GPT-5.4 mini (2026-03-17), reasoning none;
user Gemini 2.5 Flash (none); judge Gemini 3 Flash Preview (low). No automatic
model substitution. Maximum 100 steps, 10 errors, seeds 300--302.
Full-v3 and pilot-v6 pin the same source task assets. Historical v1/v2 runs and
checkpoints remain separate. A pilot score is not benchmark-wide ASR.

Quota before validation: daily used $1.10744015 of $10; monthly used $9.0103096
of $100. Check again before full runs. No purchased inference or external keys.
If quota is insufficient, stop scheduling rather than silently changing models,
limits or task coverage. Unknown cost is not zero.

## Technical Retry Policy (Declared Before Hosted V3 Runs)

Select original completed episodes without rescoring or model calls.
At most one additional attempt per technical-error/incomplete episode is allowed,
under identical protocol/source/data/model/settings, with original attempts
preserved and reported. Do not rerun a completed score-zero episode. Never copy
pilot/preflight results into full evaluation. Do not retry an entire matrix if it
would repeat completed episodes. Any retry upload must contain only the necessary
verified resume artifacts and remain private. A second technical failure blocks
release of that target/regime until investigated; missing rows have no final score.

## Dataset License And Visibility

The repository license remains MIT, and the bundle includes its unchanged text.
Kaggle's supported license catalog has no MIT entry. Use `other` with an explicit
MIT statement and reference to LICENSE in the description, not an invented MIT
catalog slug or a different substantive license. Always include `isPrivate: true`
in update metadata: the installed CLI otherwise defaults an omitted value to false.
Verify returned visibility and license after creation and each metadata update.

Primary metadata specification:
https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md

## Release Gates

- Full coverage, checksums, identities, aggregation and ENV replay verified.
- NL-scored rows use exactly the expected assertions; malformed judgments remain
  technical errors. No judge reruns during audit. Contextual manual audit reports
  only actually inspected examples, not agreement statistics without annotation.
- Compliance score is named explicitly; missing escalation, communication-only
  rubric failures, blocked attempts and executed harm are distinguished.
- Benchmark description documents 35 tasks/eight domains, ten NL-scored tasks,
  protocol/model limits, pass^k, service models, costs and limitations.
- Solo and dual stay separate scored tasks; they may be grouped into one benchmark
  collection without averaging the regimes into an unexplained score.
- Privacy, licensing, credential scan and visibility of all linked resources checked.
- Draft release content and exact publish commands prepared, not executed.

## Execution Log

Private full Dataset created: `ivanblch/duma-bench-private-v3-runtime`, ID 12829012.
Server metadata confirms `isPrivate: true`, license `other` and the explicit MIT
description. Full bundle SHA-256:
`0c6e00cbe3322e8ce48a2e4a410a6bd26df3c2d6d1de60056622e5690b06304e`.
Readiness task `duma-v3-runtime-check/1` pushed; hosted validation pending.
Offline checks already passed:
259 research/SDK tests (16 live skips); 141 installed minimal-wheel tests.

### Final Packaging

Added the unchanged upstream Sierra Research MIT terms in `NOTICE`, copied into
both wheel and Dataset. This is an attribution/packaging change, not a task or
runtime-Python change. Tests now include the generic offline result auditor.
Current local regression: 267 passed, 16 skipped, including the end-to-end
auditor test; all eight auditor tests pass. Installed minimal wheel: 141 passed,
one optional build-tool skip and one legacy-runner deselection.

Final full bundle SHA-256:
`7ae44b1ba0c0a515581ebe89e370ad475f827267b2d482284fc3c1fce38d25b3`.
Pilot-v6 bundle SHA-256:
`9cf2e2baf562a52339016f09d8fdf7f0f8312aca38ccb8758b85d6a72feb0de8`.
Full protocol hash:
`8315b8cdcf882fa83f20d2224e78d573d8ea2f924aeead5351bad76704878d03`.
Runtime source hash:
`df4a672c8615c9cd974cf5044bb287a9f337ada8e3238b78156a63dcd999a70b`.
Data hash including revision marker:
`815b04be70557438697a1b7b7bfd018e07f2478087682d08d38fd1146be81fd0`.

Full Dataset updated with final packaging. Separate pilot Dataset created:
`ivanblch/duma-bench-private-v3-pilot-runtime`, ID 12829103. Both server responses
confirm private visibility, `other` license and explicit MIT terms. Known local
environment-secret values were checked against both archives and nested wheels;
no matches. This check does not certify that every synthetic-looking example is
necessarily non-sensitive; review the release assets before publication.

Readiness run 5442610 (initial package) remains Running; the source notebook
`ivanblch/new-benchmark-task-f9c57` also reports Running. No execution log is
available from either the benchmark or kernel log endpoint yet. The CLI refuses
to replace a task while creation is pending. One bounded `--wait 600` push waits
for that run before validating the final NOTICE-containing package. No model
preflight, pilot or full V3 run has been launched before this gate passes.

Prepared wrappers (not yet pushed): `.cache/kaggle-v3-release-preflight-{dual,solo,communication}`
and `.cache/kaggle-v3-pilot-{solo,dual}`. All preflight archives match the final
full bundle; both pilot archives match each other.

Release documentation draft: `benchmarks/kaggle/BENCHMARK_CARD.md`.
Auditor: `benchmarks/kaggle/verify_results.py`; use the bundle's installed wheel
and extracted data, never the full source checkout for a frozen-run audit.

### External Readiness Blocker (2026-10-24 19:21 UTC)

Readiness 5442610 / `duma-v3-runtime-check/1` is still Running after about
31 minutes. Kernel status also reports Running. The benchmark log endpoint
returned 400, the ordinary kernel log was empty, and the kernel log-stream
endpoint returned HTTP 500. The downloaded source notebook has no persisted
execution outputs; this alone does not establish whether execution started.

The 600-second CLI push wait timed out, then tried creating a version anyway;
the server refused because creation was still pending. No final-package task
version was created. A further bounded 600-second read-only wait with increasing
poll intervals also ended pending. No model preflight/pilot/full runs were
scheduled and no public visibility was enabled. Do not infer a security result,
successful installation or a DUMA code failure from this state.

All local preparation is committed as `87c3ce2`. Current regression is 267 passed,
16 live skips; minimal-wheel regression remains 141 passed. No recurring monitor
was created. The explicit private-upload approval covers the two named V3 Datasets
and related private validation tasks, not public publication.

Resume with a status check. Once creation is terminal, inspect any failure before
retrying; then push `.cache/kaggle-v3-release-readiness/hosted_pilot.py` with the
same full Dataset attached. That final NOTICE-containing bundle must pass
readiness before pushing the already prepared preflights and pilot wrappers.
Avoid `push --wait` on an already pending task: installed CLI 2.2.4 ignores its
initial wait timeout and still attempts a new version. Use read-only waiting
and a separate push after observing terminal state.

### Readiness Passed; Resume (2026-10-24)

Run 5442610 / `duma-v3-runtime-check/1` completed at 19:51:59 UTC
(23:51:59 Asia/Tbilisi). Downloaded outputs confirm `No broken requirements found`,
Kaggle SDK 0.6.2, minimal runtime 0.1.0, no full DUMA distribution, and successful
protocol/data validation for all 35 tasks. The 1.0 result is installation readiness,
not a security measurement. Artifacts: `.cache/kaggle-v3-initial-readiness-results`.

The initial and final bundles' `duma/` Python files and all data files were compared
byte-for-byte and match. Only LICENSE/NOTICE packaging and distribution metadata
changed. To avoid repeating an otherwise identical standalone readiness run,
final-package installation is checked in the first dual preflight bootstrap:
manifest verification, clean-host/SDK compatibility, wheel installation, `pip check`
and full protocol/data validation must succeed before any inference. No task,
generation, scoring or retry settings change. Solo and communication preflights
remain gated on this first episode. This supersedes the separate zero-model
final-package readiness push described above.

Quota before preflight is unchanged: daily $1.10744015 / $10 used, monthly
$9.0103096 / $100 used. No readiness inference cost was observed.

### Preflights Passed; Pilot Started (2026-10-25 Asia/Tbilisi)

The final NOTICE-containing package passed all three Gemini preflights:
dual collab run 5444068, solo collab run 5444177, and communication run 5444232
(`output_handling/output_xss_block`). Each completed row was audited using the
frozen installed wheel: integrity hashes, tool-response/state replay, environment
assertions and judge-call/schema consistency. Communication scoring was present
in exactly the expected preflight. Reports are under
`.cache/kaggle-v3-preflight-results/{dual,solo,communication}`.
These one-row diagnostics are not full benchmark scores.

Private pilot tasks `duma-v3-pilot-solo/1` and `duma-v3-pilot-dual/1` were pushed
with the approved pilot Dataset. Gemini validation runs are 5444250 and 5444255.
The local foreground driver `benchmarks/kaggle/private_pilot.py` adopts existing
tasks/runs, audits Gemini before scheduling GPT, checks both free quota periods,
and stops on technical failure or duplicate attempts. It does not publish, update
Datasets, retry failed episodes or install any recurring automation.

The auditor permits only finite aggregate-float roundoff (1e-12 tolerance), because
artifact enumeration can alter summation order. Counts, coverage, per-episode
scores, assertions, identities and hashes retain strict checks. No frozen runtime,
task data or protocol settings were changed.

Local regression after adding preflight auditing and the foreground driver:
281 passed, 16 live skips; Ruff passed. Full solo/dual wrappers are prepared in
`.cache/kaggle-v3-full-{solo,dual}`; both archive hashes equal the final full
bundle hash above. They are not pushed until the pilot is audited.
