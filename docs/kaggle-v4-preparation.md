# Unified V4 Preparation

Date: 2026-10-26. Private full Gemini V4 matrix running; other-target checks in progress.
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

All three final-runtime Gemini preflights passed strict offline audits; reports:
`.cache/kaggle-v4-r013-preflight-results/{solo,dual,communication}-gemini-verified.json`.
Haiku solo 5484252 also passed strict audit:
`.cache/kaggle-v4-r013-haiku-results/solo-verified.json`.

Full private V4 tasks use runtime 0.1.3, Dataset version 3 and the protocol hash
listed below. Gemini validation runs are active: `duma-v4-solo/1` 5484273 and
`duma-v4-dual/1` 5484274, 105 episodes each. No other full target is scheduled.
Free quota before dispatch: USD 9.8312372 daily / 86.27519505 monthly. Full
Gemini results must be downloaded and audited before further target scheduling.

Final-runtime GPT solo 5484272 completed; dual 5484269 and communication 5484271
are running. Haiku dual 5484268 and communication 5484270 are running. Audit
these original artifacts before promoting either model to full evaluation.

Alternative-model tool probes: Qwen Next 80B Instruct 5484248 and Qwen Coder
480B 5484249 returned HTTP 404; DeepSeek-R1 5484250 explicitly rejects tool
calling. Last remaining Qwen Next Thinking probe 5484267 is running. No
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
