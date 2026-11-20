# Audited Resume Selection And Quota Gate

Date: 2026-10-30, Asia/Tbilisi. The user requested continuing the interrupted
evaluations and stated that quota had refreshed. The live typed Kaggle API,
checked twice, still reported daily used USD 9.9805148 and remaining
**USD 0.0194852**. Monthly remaining was USD 59.378813805.

The API's daily refill timestamp is `2026-10-30 07:42:07.810 UTC`, which is
**2026-10-30 11:42:07.810 Asia/Tbilisi**. The clock at verification was
`2026-10-30 06:37:22 UTC` (10:37:22 Tbilisi), before that refill.
Do not infer quota availability merely from the client calendar day changing.

No model run, model probe, Dataset upload, paid top-up or automatic follow-up
was scheduled in this turn. The earlier GPT reservation exceeded USD 0.12 per
call and Sonnet reservations exceeded USD 0.04; the actual remaining balance
cannot support these resumed episodes.

## Prepared Locally Without Inference

| Matrix | Retry rows | Preserved completed rows | Original run |
|---|---:|---:|---:|
| GPT-5.5 V4 dual | 15 | 90 | 5699821 |
| Sonnet 5.5 transport-v2 dual | 77 | 28 | 5700040 |

Generator `benchmarks/kaggle/build_resume_manifest.py` verifies each original
checkpoint against an explicitly pinned immutable audit, exact 105-row
coverage, target/regime and identical code/data/protocol/SDK/services. It
selects only technical/incomplete rows with no safety score. Completed score-0
episodes are preserved just like successful episodes. All original artifacts
remain unchanged. Four focused tests and Ruff passed.

Immutable local selections:

- `.cache/kaggle-resume-20261030/gpt55-manifest.json`, checksum
  `4260f1a611f106d94f29e93775a628d32f1a7d28848d719b35cf9a5630cf2eb1`.
- `.cache/kaggle-resume-20261030/sonnet55-manifest.json`, checksum
  `60aeb6bfffa1741afa3024a3f5660d4bf3dd17ecc9af7544031f5d9ad03fbc1c`.

These are selection manifests, **not uploaded/running retry tasks** and not
final scores. They include original metadata identities, row/seed identifiers,
original artifact hashes and preserved scores; no trajectories or API keys.

## Next Active Turn

Recheck the live typed daily quota after the stated refill. Generate retry-only
hosted tasks from these selections, attaching the existing private runtime and
preserving the exact original GPT V4 and Sonnet v2 identities. No new Dataset
is necessary: the small selection manifests can be part of the private task
source, without re-uploading original trajectories or full completed artifacts.

Run GPT's 15 interrupted episodes first. Then inspect remaining quota before
Sonnet's 77. Never resample the 90/28 completed original rows. Retain every old
technical attempt, verify that resumed metadata hashes match original rows,
and stop dispatching new rows promptly on quota denial instead of producing
dozens of additional known-failing requests. Do not reduce generation limits
or tool constraints and label it the original protocol.

Afterward independently merge original completed rows and declared new
attempts for scoring; keep the two profiles separate. Do not publish a full
score until the selected original-plus-resumed matrix has complete coverage.
