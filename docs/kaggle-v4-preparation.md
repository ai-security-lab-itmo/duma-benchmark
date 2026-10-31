# Unified V4 Preparation

Updated: 2026-10-28. Private five-target evaluation complete and audited:
1,050/1,050 episodes, including 300 NL-scored rows. Further GPT-5.4, Gemini
3.6 Flash and Flash-Lite full evaluations are running; current IDs and compatibility evidence:
`kaggle-v4-next-models-20261028.md`. See `kaggle-v4-expansion-20261028.md` for
the audited five-target results and Chinese-model probes. Earlier history below
is preserved. Not published.
Work remains on `codex/kaggle-benchmarks`; no merge/publication.

## User Decision And Protocol

The user first approved minimal reasoning for Qwen/GLM, then requested that all
targets be displayed in one profile. This supersedes the proposed separate
none/minimal profiles: one `duma-kaggle-private-full-v4` protocol now defines
provider-compatible target settings explicitly.

| Target model proxy identity | Target reasoning |
|---|---|
| google/gemini-3.8-flash | none |
| openai/gpt-5.4-mini-2026-03-17 | none |
| anthropic/claude-haiku-4-5@20251001 | none |
| qwen/qwen3-235b-a22b-instruct-2507 | minimal |
| zai/glm-5 | minimal |

Default target reasoning is none; exact Qwen/GLM identities have pinned minimal
overrides. Every V4 episode records its effective target generation arguments;
the offline auditor rejects mismatches. This is one versioned, provider-compatible
comparison protocol, not proof of equal reasoning compute. Do not universally
equate minimal with disabled reasoning. Show settings next to model results.

All 35 tasks, policies, prompts, tools, assertion semantics, seeds 300--302,
100-step/10-error limits and pinned user/judge settings are unchanged from V3.
The hardened task-data bytes, including generated revision marker, are identical.
Runtime 0.1.1 adds the tested Anthropic system-only solo newline transport fix and
the versioned target override dispatch. Existing Gemini/OpenAI transport paths
are unchanged. The two V3 baseline results remain historical; a common V4 table
requires new V4 matrices, not relabeling or pooling old-source checkpoints.

Full five-target evaluation is 1,050 episodes: 35 tasks x 3 repeats x 2 regimes x
5 targets. Stage inference against actual free quota and diagnosed compatibility;
do not schedule the entire matrix blindly or buy paid API capacity.

## Frozen Candidate

### Active Full Evaluation

Final three-target check, 2026-10-26: Haiku solo 5487724 and dual 5487762 both
completed. The user explicitly requested dual to run alongside solo after the
initial staged launch; no automatic favorable retries were performed.

Strict combined audit passed 630/630 episodes across all six matrices, including
180 NL-scored rows. All matrices have 100% coverage with no technical errors or
unfinished episodes. Frozen identities/checksums, environment replay and hosted
versus independent aggregation matched. This is not an independent manual audit
or fresh judge scoring of all communication decisions.

| Target | Solo compliance | Dual compliance | Solo / dual failed rows |
|---|---:|---:|---:|
| Gemini 3.8 Flash | 99/105 (94.29%) | 101/105 (96.19%) | 6 / 4 |
| GPT-5.4 mini | 98/105 (93.33%) | 91/105 (86.67%) | 7 / 14 |
| Claude Haiku 4.5 | 99/105 (94.29%) | 93/105 (88.57%) | 6 / 12 |

Higher compliance is better; its complement is assertion failure, not uniformly
confirmed attack success. These V4 rows remain separate from V3 and paper results.
Haiku dual failures concentrate in collab (9/18 rows) and tool_shadow_poison
(3/9); solo failures are six collab rows. Do not infer statistical significance
from small score differences across only three repeats per task.

Final report: `.cache/kaggle-v4-full-results/verified-three-targets.json`, checksum
`bb4540770767ce685668aaad6ec5b357c04c35c83613a86e51767b2c581d1566`.
All-role evaluation costs: Gemini USD 3.089688225, GPT USD 1.2789853, Haiku USD
2.6843404; total USD 7.053013925, excluding compatibility diagnostics and V3.
Latest free quota snapshot: USD 2.729526525 daily / 79.173484375 monthly.

No full run remains active. No public release or merge was performed. Chinese
model availability remains unresolved; they have no V4 scores. Before public
release, review the V4 failed trajectories, refresh release descriptions/cards
from this audited report, and reconcile live privacy/artifact visibility again.

Everything below is historical preparation/progress evidence; final state above
supersedes earlier pending/active run notes and staged scheduling instructions.

Latest completion check, 2026-10-26: GPT solo 5487055 and dual 5487056 both
completed. Combined strict audit passed 420/420 Gemini/GPT episodes, including
120 NL-scored rows; all four matrices have complete coverage and no technical
or incomplete episodes. No new model/judge calls were made during this audit;
it is not an independent human re-judging of communication assertions.

| Target | Solo compliance | Dual compliance | Solo / dual failed rows |
|---|---:|---:|---:|
| Gemini 3.8 Flash | 99/105 (94.29%) | 101/105 (96.19%) | 6 / 4 |
| GPT-5.4 mini | 98/105 (93.33%) | 91/105 (86.67%) | 7 / 14 |

Audit report: `.cache/kaggle-v4-full-results/verified-gemini-gpt.json`, checksum
`a24a28525061f0b650e5e3538a1751f4e269f3e17ff0094e55b5d23ba6b5b975`.
All-role costs: Gemini USD 3.089688225; GPT USD 1.2789853; combined USD
4.368673525 (excludes diagnostics and earlier V3 experiments).

Free quota at this check: USD 5.413866925 daily / 81.857824775 monthly. Started
only Haiku solo on the existing frozen `duma-v4-solo/1`: run 5487724, 105 rows.
Its preflights already passed strict audits. Haiku dual is not scheduled yet:
wait for solo completion, inspect cost/coverage, then reread quota before dual.
Do not include partial Haiku results in a full two-regime ranking or change
the existing full task versions/protocol to accommodate unavailable models.

The entries below retain earlier investigation/scheduling history; current
completion/scheduling state is the paragraph above.

Current provider-ID investigation, explicitly requested on 2026-10-26:
the prior retries only covered Kaggle's advertised, old model identities.
Official provider documentation now lists Qwen3.8 Flash, GLM-5.3 and DeepSeek
V4.1 Flash (`deepseek-flash`). New generations support tool calling; the old
DeepSeek-R1 endpoint rejection must not be generalized to current DeepSeek.

To test actual routing rather than only the picker, created private diagnostic
`duma-current-chinese-model-id-probe/1`, run 5487426. It uses Kaggle's hosted
OpenAI-compatible proxy directly, omits reasoning/temperature, caps output at
128 tokens, supplies a side-effect-free ping tool and disables retries. It
does not call provider endpoints directly, change DUMA or produce a model score.
The automatic task label is Gemini, but the report explicitly records each
requested Chinese model ID; never interpret that label as a Gemini evaluation.

All six real requests returned HTTP 404 with explicit `model ... was not found`:

- `qwen/qwen3.8-flash`, `qwen3.8-flash`
- `zai/glm-5.3`, `glm-5.3`
- `deepseek-ai/deepseek-flash`, `deepseek-flash`

The proxy's OpenAI-compatible models-list endpoint also returned 404. A temporary
local proxy token's models-list attempt returned 404 too; local tokens have a
restricted model subset, so the hosted calls are the relevant routing evidence.
No token was printed or persisted. No external paid provider API was called.
These results establish that tested current IDs are not available through the
accessible Kaggle proxy; they do not establish global/provider unavailability
or whether the root cause is routing configuration versus access/catalog policy.

Evidence: `.cache/kaggle-current-model-id-probe-results`, including exact HTTP
messages in `current-model-id-probe.json` and executed source notebook. Standalone
probe source: `.cache/kaggle-current-model-id-probe/probe.py`; no DUMA Dataset
attached, no runtime/provider dependencies added, no active full run modified.

Primary documentation checked:

- https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen3-8-flash
- https://docs.z.ai/guides/llm/glm-5.3
- https://api-docs.deepseek.com/quick_start/agent_integrations/
- https://api-docs.deepseek.com/quick_start/pricing/
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md

Chinese-model recheck requested explicitly by the user on 2026-10-26: refreshed
the advertised catalog and repeated one tool-only API probe per Chinese model
on `duma-v4-r013-model-compatibility/1`, reasoning minimal, max_tokens 512.
All six fresh attempts failed before demonstrating tool compatibility:

| Model | Fresh run | Result |
|---|---:|---|
| Qwen3 235B A22B Instruct 2507 | 5487300 | HTTP 404, model may no longer be available |
| GLM-5 | 5487301 | HTTP 404, model may no longer be available |
| Qwen3 Next 80B A3B Instruct | 5487302 | HTTP 404, model may no longer be available |
| Qwen3 Next 80B A3B Thinking | 5487303 | HTTP 404, model may no longer be available |
| Qwen3 Coder 480B A35B Instruct | 5487304 | HTTP 404, model may no longer be available |
| DeepSeek-R1 0528 | 5487305 | HTTP 400, tool calling not supported |

Fresh outputs/source notebooks: `.cache/kaggle-v4-chinese-recheck-20261026`;
previous diagnostic directories remain untouched. The previously completed
Gemini validation 5484237 is also downloaded there, but is not a fresh attempt.
These are compatibility diagnostics, never DUMA security scores. No replacement
has been pinned; no task, protocol or ongoing GPT full run was changed.

Latest follow-up, 2026-10-26: Gemini full runs 5484273 (solo) / 5484274 (dual)
both completed. Strict frozen-wheel audit passed 210/210 rows, 60 NL-scored
rows, no technical/incomplete episodes. Hosted aggregation matched independent
recomputation and environment assertions replayed exactly. This is not a new
human audit or LLM re-judging of all communication assertions.

| Gemini 3.8 Flash V4 | Compliance | Assertion failures |
|---|---:|---:|
| Solo | 99/105 (94.29%) | 6/105 (5.71%) |
| Dual | 101/105 (96.19%) | 4/105 (3.81%) |

Solo failed rows are all collab; dual has three collab and one CRM failure.
Assertion failure is not uniformly confirmed attack success. Cost across all
roles: USD 3.089688225. These are fresh V4 results; do not pool them with V3.

Audit: `.cache/kaggle-v4-full-results/verified-gemini.json`, checksum
`d5df737cfbd437533b3f8824a67c276436b3037cd40c5edf716933c7a7342aef`.

GPT full evaluation is now running on the existing frozen version-1 tasks:
solo 5487055 and dual 5487056, 105 episodes each. Original GPT preflights were
already audited. Quota before dispatch: USD 6.692852225 daily / 83.136810075
monthly remaining. Haiku full remains unscheduled until GPT audit and a fresh
budget check; do not start all remaining targets without accounting for cost.

Historical in-progress snapshots follow; the completion/audit above supersedes
their status and pending-dispatch notes.

Manual follow-up, 2026-10-26: both original Gemini full runs remain running.
Bounded live-log snapshots at approximately 620 elapsed seconds contained 44
distinct solo episode-start lines and 17 dual starts, with zero generation-error
lines. These counters are progress diagnostics, not completed coverage or valid
scores. Do not infer final ASR, retry rows, or launch duplicate full tasks.
Latest quota snapshot: USD 9.39492835 daily / 85.8388862 monthly remaining.

The installed 0.1.3 wheel passed 30 protocol/metrics/revision tests (one optional
build-tool skip) plus five runtime tests (one legacy comparison deselected),
using `python -m pytest --noconftest` and the unchanged source data. Initial
collection via the full repository conftest imported intentionally excluded
`duma.run`; disabling that research-only conftest resolved collection without
adding a runtime dependency or changing the tests. Historical full-research
regression remains 293 passed / 16 skipped.

When both original runs are terminal, download them with source notebooks into
`.cache/kaggle-v4-full-results`, then audit using the installed frozen wheel:

```sh
env DUMA_DATA_DIR="$PWD/.cache/kaggle-v4-r013-assets/data" \
  .cache/kaggle-v4-minimal-env/bin/python benchmarks/kaggle/verify_results.py \
  --results .cache/kaggle-v4-full-results \
  --bundle .cache/kaggle-v4-r013-runtime/duma-pilot.zip \
  --protocol benchmarks/kaggle/protocol.full.v4.json \
  --target google/gemini-3.8-flash \
  --output .cache/kaggle-v4-full-results/verified-gemini.json
```

Require 210/210 complete, all scoring checks and matching identities before
next-target dispatch. Inspect original errored rows if incomplete; never
relabel technical failures as attacks. Re-read free quota, then schedule GPT
and Haiku in budget-aware stages on existing `duma-v4-solo`/`duma-v4-dual`
version-1 tasks; inspect existing run IDs first. No watcher was installed.

All three final-runtime Gemini preflights passed strict offline audits; reports:
`.cache/kaggle-v4-r013-preflight-results/{solo,dual,communication}-gemini-verified.json`.
Haiku solo 5484252 also passed strict audit:
`.cache/kaggle-v4-r013-haiku-results/solo-verified.json`.

Full private V4 tasks use runtime 0.1.3, Dataset version 3 and the protocol hash
listed below. Gemini validation runs are active: `duma-v4-solo/1` 5484273 and
`duma-v4-dual/1` 5484274, 105 episodes each. No other full target is scheduled.
Free quota before dispatch: USD 9.8312372 daily / 86.27519505 monthly. Full
Gemini results must be downloaded and audited before further target scheduling.

Final-runtime GPT solo 5484272, dual 5484269 and communication 5484271 completed
and passed strict offline audits. Haiku dual 5484268 and communication 5484270
also completed and passed, alongside its audited solo. Reports:
`.cache/kaggle-v4-r013-service-preflights/*-verified.json`. All nine diagnostic
rows across Gemini/GPT/Haiku are audited, including three NL-scored rows. GPT
and Haiku full evaluation remains unscheduled pending full Gemini audit/quota.

Alternative-model tool probes: Qwen Next 80B Instruct 5484248 and Qwen Coder
480B 5484249 returned HTTP 404; DeepSeek-R1 5484250 explicitly rejects tool
calling. Last remaining Qwen Next Thinking probe 5484267 also returned HTTP 404. No
compatible Chinese replacement has been demonstrated or pinned. Their failure
must not block private evaluation of supported models or become a scored row.
Adding an eventual model-specific override requires a new frozen protocol
identity; never silently amend the already-dispatched full task versions.

All resources remain private; no V3 updates, publication, merging, paid capacity
purchases, or monitoring automation. Remote runs continue on Kaggle; next
manual follow-up should inspect existing run IDs, not recreate them.

### Latest Runtime And Model Gate

Runtime 0.1.3 replaces Anthropic's whitespace-only solo transport prefix with
punctuation (`.`); the provider rejected whitespace-only text with HTTP 400 in
5484207. This transport-only prefix is replayed consistently and is not inserted
into real-user dual histories. Gemini/OpenAI paths and task assets are unchanged.
Full regression: 293 passed, 16 skipped; Ruff passed.

- Runtime Python: `9722a4029bbdc5af9e62dfe47304cffa8077e804805d1e400ce7cb97498b2462`
- Bundle: `9f5f1cc07f397f81c6c435af6600271aa2b7f475bb44572e18eeabbbd206c611`
- Dataset version 3; `.cache/kaggle-v4-r013-runtime`, extracted `-r013-assets`.
- `duma-v4-r013-runtime-check/1` passed; fresh r013 solo/dual/communication tasks
  are validating this runtime. The installed audit environment is now 0.1.3.

All three 0.1.2 Gemini diagnostics passed strict frozen-wheel audits (one NL
row). Reports are in `.cache/kaggle-v4-r012-preflight-results`. Haiku dual
5484214 completed with score zero; no favorable retry was scheduled.

Qwen dual 5484215 ended with too_many_errors after endpoint HTTP 404, not a
security failure. GLM solo 5484211 returned the same model-not-found response.
Both identities remain in Kaggle's advertised catalog, which does not guarantee
actual endpoint availability. Original diagnostics remain in
`.cache/kaggle-v4-r012-target-probes`.

The user explicitly approved testing alternative available Chinese models and
pinning compatible replacements. A private `duma-v4-r013-model-compatibility/1`
single-tool probe is prepared/running (not a DUMA security score). It uses the
selected Kaggle target, with minimal for candidate non-Google targets and none
for automatic Gemini validation. No replacement is yet pinned; supported-model
full evaluation proceeds with the already-frozen protocol, without replacements.

### Earlier Candidates

The initial 0.1.1 candidate below is retained for provenance only. Hosted Gemini
readiness passed (5484134); initial dual/solo/communication diagnostics completed
(5484156/5484157/5484158). Strict offline audit rejected solo: the agent mutates
its generation dictionary by adding seed, which also changed saved metadata.
Do not use these initial diagnostics as certified V4 ranking results.

Runtime 0.1.2 passes a copy of generation arguments into the engine, preserving
metadata/checkpoint identity. The regression covers both regimes and resume
without new target calls. Full tests: 293 passed, 16 skipped; focused tests: 42
passed; Ruff passed. Direct pytest initially failed collection because tests/
was absent from its import path; python -m pytest passed without source changes.

Corrected candidate (same protocol and byte-identical task data):

- Runtime Python: `b07d80d8664d14b3e87fa0ca8cd90a54ff9360390cd4740e17ab5a905a8d85f8`
- Bundle SHA-256: `024fe81c8e76d15e15e9f63321e76e275420ac15213c03aef3dfcb3fd31b892e`
- Staging: `.cache/kaggle-v4-r012-runtime`; extracted: `.cache/kaggle-v4-r012-assets`.
- Private Dataset version 2 uploaded; corrected readiness task:
  `duma-v4-r012-runtime-check/1`. Await its completion before corrected preflights.
- Installed audit environment now contains 0.1.2, not the original 0.1.1 wheel.

Corrected readiness passed. The first r012 preflight wrappers retained the old
manifest hash and failed closed during bootstrap, before inference. Their
diagnostics remain untouched; wrapper manifest pins were corrected to
`7ec6e5da8b6f796fd96883e93a3d7ddd1697f03dac4beee82cb4c3b59f2a9247`.
Fresh `duma-v4-r012b-preflight-{dual,solo,communication}/1` tasks now validate the
same 0.1.2 bundle. No integrity checks were bypassed or relaxed.

Original candidate identities:

- Protocol JSON: `benchmarks/kaggle/protocol.full.v4.json`
- Protocol fingerprint: `cc5dedcad4c423156f17d257129a77c846fb418680f23da7d97e00480e6e7325`
- Runtime Python: `bac845c6c388258ffc38b7147ec409948845639ec4858d60f1ed0d57535510bc`
- Data: `815b04be70557438697a1b7b7bfd018e07f2478087682d08d38fd1146be81fd0`
- Bundle SHA-256: `5089946d19ddec1af196516f3cafc921913c5bf239b444000baeff81d8ad4738`

Dataset staging folder: `.cache/kaggle-v4-runtime`; intended new private Dataset:
`ivanblch/duma-bench-private-v4-runtime`. Includes MIT/NOTICE and the data/safety
notice in its prepared metadata. No real credential file is included.

Prepared task wrappers, all using the identical bundle:

- `.cache/kaggle-v4-runtime`: `duma-v4-runtime-check`, installation only.
- `.cache/kaggle-v4-preflight-dual`: one dual collab episode.
- `.cache/kaggle-v4-preflight-solo`: one solo collab episode.
- `.cache/kaggle-v4-preflight-communication`: one dual output-handling episode.
- `.cache/kaggle-v4-full-solo`: full solo matrix.
- `.cache/kaggle-v4-full-dual`: full dual matrix.

Readiness and preflight scores are not full ranking measurements. Task push
automatically starts validation on Kaggle's default Gemini target, which uses
none under this protocol. Its full validation is a legitimate V4 baseline, not
a readiness score or an implicit Haiku/Qwen/GLM result. Explicitly schedule each
additional target only after its compatibility gates; never hardcode a different
target into a run labeled as Gemini.

## Local Validation

- Research/SDK tests/: 292 passed, 16 live skips; Ruff passed.
- New clean installed-wheel environment, no full DUMA or provider SDKs:
  36 protocol/metrics/revision tests passed, two optional build-tool skips;
  five runtime tests passed, one old-runner equivalence comparison deselected.
- All 35 task contracts match the original hardened fixture. Running that source
  fixture check directly on extracted bundle data initially reported the extra
  generated revision marker. Independent comparison proved no changed/missing
  assets and identical task contracts; rerun against identical source data passed.
  No fixture expectations were relaxed.
- Updated auditor with the original frozen V3 wheel rechecked 420/420 episodes
  and 120 NL-scored rows successfully. Historical fingerprint remains unchanged.
- New archive including nested wheel: 127 scanned elements; no configured-token
  match or tested secret patterns, and no credential/Git/SSH files.

Isolated new audit runtime: `.cache/kaggle-v4-minimal-env/bin/python`.
Extracted candidate: `.cache/kaggle-v4-unified-assets`; use its data for V4 audits.
The earlier none/minimal candidate archives in `.cache/` were local-only drafts;
they are superseded and must not be uploaded or confused with unified V4.

Two tiny local proxy transport probes of Qwen/GLM with minimal returned model-not-
found 404 for that temporary local token. They are not benchmark/model scores
and do not establish hosted model availability; hosted access must be validated.
No proxy token was persisted or printed.

## Upload Authorization Gate

Resolved: the user explicitly approved the exact V4 private upload and five-model
quota-funded checks/evaluation. Dataset `ivanblch/duma-bench-private-v4-runtime`
was created successfully (ID 12853003, version 1, server-confirmed private and
ready). The private `duma-v4-runtime-check/1` installation-only task was pushed
with only this Dataset attached; validation is running. No V3 resource changed.
At scheduling, free quota was USD 9.9682838 daily / 86.41224165 monthly.

The first private V4 Dataset-create command was rejected by the approval reviewer
before execution: prior explicit upload approval named the V3 destinations,
not this new V4 destination/payload. No V4 bytes were uploaded. Do not work around
the rejection by mutating V3 or using a different route.

Requested explicit user approval for code, all 35 tasks/test secrets and the V4
protocol in `ivanblch/duma-bench-private-v4-runtime`, followed by private checks
and five-model evaluation using Kaggle quota, without public publication. That
approval was subsequently received; the original rejection was not bypassed.

After approval: create private Dataset, verify server metadata, hosted readiness,
then one-episode dual/solo/judge compatibility and frozen-wheel audit. Only then
schedule complete matrices in quota-aware stages. New failures/settings changes
must retain original diagnostics; do not silently retry valid score-zero rows.
