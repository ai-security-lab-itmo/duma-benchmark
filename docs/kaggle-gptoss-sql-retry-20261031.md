# GPT-OSS Single SQL Retry

Date: 2026-10-31, Europe/Moscow. User explicitly requested repeating the
only technically unfinished row of full dual run 5763534.

Selection: `output_handling/output_sql_block`, dual, trial 1, seed 301.
Original artifact hash:
`f31cab08ae2e84939b6731cc7c4f54df83551611e745824f46583a4224e8a73a`.
Pinned original independent audit:
`f06f7290cfb3074a93cb19f77d3b93d775958c9a6d5cb39199330b846705c94f`.
The other 104 rows (89 passes, 15 failures) are preserved unchanged.

Private task `duma-gptoss-sql-retry-v1/1` attaches the same private V4 runtime
Dataset. Original compatibility definitions are unchanged; low reasoning,
named-empty-descriptions repair, tools, policy, user/judge, limits and seeds
are identical. Profile hash remains
`7792ba790baee00c290b213c431203b4f969241e25e174a44764326c53dc0315`.
Generated source SHA256:
`581064edf26da368e11d71daff01eca273e640b7ed662d3e09528dc7f7ce24f9`.

The worker checks protocol/source/data/SDK identity before inference, selects
only the declared trial, saves full raw target responses and request deltas,
and verifies the produced checkpoint against exact original metadata. New
artifacts live under `duma-gptoss-retry-results`, separate from originals.
No full matrix is rerun, no completed failure is resampled, and no provider
block is automatically credited as a success.

The offline full auditor verifies the original 105 rows/summary/wires first,
then pins the original audit and source row before accepting exactly one
metadata-identical replacement. Original assertions and NL judge decisions
are verified/replayed, with complete-matrix aggregation only after merging.
Both the original and new attempt remain immutable files. An unsuccessful
repeat is retained as-is, not automatically retried until a favorable result.

Two focused merge tests passed. Full regression: **346 passed, 16 skipped**,
one upstream warning. Free quota before the retry: daily USD 9.715074315,
monthly USD 55.430244355. No paid capacity or public publication.

## Dispatched

Actual run **5770167** was submitted and confirmed RUNNING. Its bounded CLI
wait ended after 240 seconds, without stopping the hosted run. Bounded log
inspection showed normal progress through step 7/tool-result processing; one
target response took approximately 80 seconds. No second attempt, automated
monitor, resampling of other rows or final merged score was dispatched/claimed
at this checkpoint. Fetch the result once terminal and use the full auditor's
`--retry` option for this single new checkpoint, keeping both attempts.
