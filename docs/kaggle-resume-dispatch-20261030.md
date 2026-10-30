# Private Interrupted-Only Continuation

Date: 2026-10-30, Asia/Tbilisi. After the daily refill, the live typed API
confirmed USD 10 daily and USD 59.378813805 monthly remaining. The user
authorized continuing the private evaluations. No publication or paid top-up.

## Guarded Resume Tasks

Both tasks are version 1, private, and attach the existing
`ivanblch/duma-bench-private-v4-runtime` Dataset. No new Dataset or original
trajectory upload was needed. Only row identities and audit hashes are embedded.
Automatic Gemini readiness returns a skip without inference or a safety score.

| Task | Retry rows | Preserved rows | Generated source SHA256 |
|---|---:|---:|---|
| `duma-gpt55-dual-resume-20261030` | 15 | 90 | `55c67312cd0dd93389d1fec618236bec8a7b5fc7561835d5829f8d9088f8ed94` |
| `duma-sonnet55-dual-resume-20261030` | 77 | 28 | `200d38ac131c3b4c59da0710ddc758546022cc3ba987e10b84a8a56c517669f4` |

GPT-5.5 continuation run **5742878** was dispatched first. Sonnet inference
must follow GPT completion and a fresh quota check; task readiness alone is
not a model evaluation. The frozen V4 and Sonnet transport-v2 tasks/profiles
remain unchanged and separate.

## Integrity And Scoring

`build_hosted_resume.py` uses the frozen 0.1.3 bootstrap and checks the pinned
selection checksum. Sonnet definitions must match the original adapter bytes.
`hosted_resume.py` checks exact protocol, code, data, SDK and service identities
before inference. Only declared technical/incomplete rows are selected.
Completed score-0 episodes are preserved, not retried.

The worker stops selecting further rows after HTTP 401/403/404/429. It never
reduces limits or silently substitutes models. New checkpoints are stored in
`duma-resume-results`, separate from original artifacts. A subset report is
not a full leaderboard score. No original checkpoint is overwritten.

`audit_resumed_matrix.py` independently checks every original hash, rejects
undeclared or duplicate retry attempts, replays completed assertions, and
aggregates the original-plus-declared-retry matrix. Missing/failed retries
leave the full score null. Costs describe selected attempts only; superseded
technical-attempt costs remain additional and may be unknown.

## Validation

- Twelve focused resume/merge tests passed in the frozen minimal environment.
- Full regression: 321 passed, 16 skipped; one upstream deprecation warning.
- Ruff checks passed for builder, auditor and tests.
- Original Sonnet data were independently replayed through the merge auditor
  without retries: 105 rows verified, 28 completed, 77 technical errors, score
  null. This confirms that missing retries do not produce a ranking.

Local selection hashes and original audit identities are recorded in
`docs/kaggle-resume-preparation-20261030.md`. Final continuation outcomes must
be added only after downloading and auditing the actual retry artifacts.
