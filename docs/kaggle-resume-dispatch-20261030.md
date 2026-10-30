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

## GPT-5.5 Completed And Independently Audited

Run 5742878 completed all 15 declared retries: 14 passed, one policy failure,
no technical errors. The 90 original completed episodes, including all six
original failures, were retained unchanged. The merged matrix has **105/105
completed, 98 passed, seven policy failures: compliance 93.33%**, failure rate
6.67%. All 105 selected artifacts and 30 communication-scored rows were
independently verified. `pass^3` is 88.57%.

Immutable merged audit:
`.cache/kaggle-resume-20261030/gpt55-merged-verified.json`, checksum
`15a71b99cf627f4d0aa95efda6c3583d63b57ac493ca605f516e061aa4659723`.
Domain compliance: auth, CRM, infrastructure, mail, marketing and output 100%;
collaboration 66.67%; tool shadowing 88.89%. Original solo remains 97/105,
92.38%. These are Kaggle protocol results, not reproduced paper numbers.

After GPT completion, the live quota reported daily remaining USD 9.068017
and monthly USD 58.446830805. Its daily spend was USD 0.931983. This includes
continuation consumption and is distinct from selected-matrix cost totals.
Only then was Sonnet continuation run **5742900** submitted and confirmed
running. Its result is pending; no
full Sonnet score is claimed yet.

## Sonnet Continuation Audited

Checked on 2026-10-30 (Europe/Moscow). Run 5742900 completed at 08:32:45 UTC.
All 77 declared retries produced artifacts: 61 completed (58 passed, three
policy failures), 16 technical errors (`RuntimeError`). The terminal log
contains repeated Model Proxy `choices[0].message=None` warnings for Sonnet.
No quota-denial early stop occurred. Failed-row domains: marketing eight,
collaboration four, output two, mail two.

Preserving the 28 original completed rows gives 89/105 completed, 83 passed,
six policy failures and 16 technical errors. The full compliance score remains
null, not a leaderboard result. No failures were converted into successes or
resampled in this status-check turn. All 105 selected artifacts and 28
completed communication-scored rows were independently verified.

Immutable audit: `.cache/kaggle-resume-20261030/sonnet55-merged-verified.json`,
checksum `c8ca23bbf228ebd971ae199aa018f3ee9ddd8eb3fa50ddc83d9439b6af663ffa`.
Quota after completion: daily remaining USD 6.4690513, monthly USD 55.847865105.
The continuation consumed approximately USD 2.5989657 since the previous
quota check; target per-role cost remains unknown in stored artifacts.
Both tasks remain private. Further work should diagnose the empty-response
transport behavior before spending quota on another retry selection.
