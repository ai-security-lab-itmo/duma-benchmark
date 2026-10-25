# Unified V4 Preparation

Date: 2026-10-26. Local preparation complete; no V4 resources uploaded or run.
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

The first private V4 Dataset-create command was rejected by the approval reviewer
before execution: prior explicit upload approval named the V3 destinations,
not this new V4 destination/payload. No V4 bytes were uploaded. Do not work around
the rejection by mutating V3 or using a different route.

Requested explicit user approval for code, all 35 tasks/test secrets and the V4
protocol in `ivanblch/duma-bench-private-v4-runtime`, followed by private checks
and five-model evaluation using Kaggle quota, without public publication. Wait
for that approval before retrying creation or any hosted V4 push.

After approval: create private Dataset, verify server metadata, hosted readiness,
then one-episode dual/solo/judge compatibility and frozen-wheel audit. Only then
schedule complete matrices in quota-aware stages. New failures/settings changes
must retain original diagnostics; do not silently retry valid score-zero rows.
