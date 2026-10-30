# Sonnet Empty-Response Investigation

Date: 2026-10-30, Europe/Moscow. This diagnostic does not alter or rescore
the original Sonnet transport-v2 matrix.

## Confirmed Offline

The 16 remaining technical failures in run 5742900 all end at
`Failed to generate/validate agent message: AssistantMessage must have either
content or tool calls`. The orchestrator marks generation errors fatal;
`too_many_errors` does not mean ten API calls were made. Retrying the full
matrix would unnecessarily resample completed episodes.

Fifteen failed episodes have zero output tokens in their final target usage
record. One (`collab_poisoning_partial_delete`, seed 302) has 104 output tokens
but still no usable assistant content/tool call. Thus it is not a separate
tool-loop failure, and raising the 4096-token generation budget is unsupported
by the observed evidence. Failed histories include both initial user requests
and tool-result continuations, so a parallel-tool replay bug alone cannot
explain every case.

The cached SDK source `actors/proxy_openai.py` explicitly returns an empty
`LLMResponse` for `choices[0].message=None` (and for no choices). It preserves
usage but discards the original response's finish reason and other fields.
That empty response then fails DUMA's validation. The original artifacts do
not preserve raw API responses, so they cannot establish whether the upstream
cause is filtering, malformed proxy output, or another provider condition.

Kaggle independently documented this symptom in
https://github.com/Kaggle/kaggle-benchmarks/issues/191 and merged the guard in
https://github.com/Kaggle/kaggle-benchmarks/pull/192. The guard prevents an
SDK AttributeError; it does not recover missing model output. Installing the
same guard again would not fix our incomplete episodes.

## Bounded Raw-Response Probe

Private task `duma-sonnet55-raw-response-probe/1` attaches only the existing
private V4 runtime Dataset. Generated source SHA256:
`c7f6b3ac4506d94259808e2720bc07354d0dd716a824d3469f207cf5a6ccf240`.

Two failed snapshots are embedded: `output_xss_block` seed 301 (initial user
request) and `mktg_phishing_kpi_pressure` seed 300 (after three tool results).
Each exact request is repeated twice, with original history, system prompt,
seed, tools and transport-v2 generation parameters. No tool is executed, no
user/judge is called, and no safety score is emitted. Snapshots carry original
checksums. Automatic readiness skips the non-Sonnet model without inference.

`sonnet_raw_probe.py` temporarily observes the client completion method,
saving the original response before SDK normalization plus parsed output or
error, and restores the client afterward. Diagnostic outputs are separate
from benchmark checkpoints. Mocked tests cover valid and empty responses,
exact four-call bounds, unchanged generation limits and restoration on
errors. Full regression: 323 passed, 16 skipped, one upstream warning.

Do not fill missing episodes with successes, strip attack content or remove
thinking carriers to force a score. Changes to transport/generation policy
need a separately versioned profile and explicit compatibility validation.

## Hosted Probe Outcome: New Regional Block

Diagnostic run **5744538** completed. All four requests were rejected before
a model response with `User location "RU" is not supported for this model/API`.
No raw completion was available, so the old response's finish reason remains
unknown. This is a new access failure, not evidence that the prior 16 empty
responses were regional errors. Further model calls were stopped; no attempt
was made to circumvent regional restrictions or substitute a different model.

Downloaded diagnostic report:
`.cache/kaggle-sonnet55-raw-probe-results/duma-sonnet55-raw-response-probe/1/claude-sonnet-5-5-default/5744538/sonnet-raw-probe.json`,
checksum `96092f40a0670f292faff6587dbab23e5bb325158ed025f6d7e993b695e4f31c`.

Next steps: resolve legitimate model availability with Kaggle support/account
configuration, then rerun the bounded probe before any 16-row retry. Inspect
the raw response for explicit filtering/refusal, a malformed completion, or
an intermittent empty response. A real textual refusal is a valid response;
an absent message alone is not evidence of a safe completed episode. If the
provider returns valid responses intermittently, use a documented bounded
transport retry policy with all attempts retained, separately versioned from
the frozen results. If it fails consistently, report the reproducible wire
response to Kaggle instead of repeatedly re-running full episodes.

Existing Sonnet matrix remains 89/105 completed, 83 passed, six policy failures,
16 technical errors, full score null. Runtime, prompts, adapter and benchmark
artifacts were not modified. Only standalone diagnostic code/tests were added.
