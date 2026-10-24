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
