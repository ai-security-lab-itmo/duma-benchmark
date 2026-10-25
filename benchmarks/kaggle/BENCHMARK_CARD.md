# DUMA-Bench: Interactive Agent Security (Hardened V3)

PRIVATE RELEASE DRAFT. V3 pilot/full audits, local content review and live server
reconciliation passed. Publication approval remains required; apply the safety
notice to the Dataset description during that transition. No V1/V2 scores are V3 measurements.

## Overview

DUMA-Bench evaluates policy compliance in executable, multi-turn security
scenarios with tools, mutable state and potentially adversarial instructions.
This Kaggle adaptation uses a minimal runtime and a separately versioned protocol;
it does not reproduce the model settings or reported numbers of the paper.

Paper: https://arxiv.org/abs/2609.24662
Code: https://github.com/ai-security-lab-itmo/duma-benchmark

The collection is intended to contain two scored tasks, with separate rankings:

- **DUMA V3 Solo** (`duma-v3-solo`): the agent receives a static ticket and works
  with the simulated environment without an active user simulator.
- **DUMA V3 Dual** (`duma-v3-dual`): an active user simulator interacts with the
  agent while it operates on the simulated environment.

Do not average the regimes into an unexplained combined score. The gap reflects
their combined interaction conditions, not an isolated causal effect of turn
count, user influence or environment changes.

## Coverage

| Domain | Tasks | Tasks with communication scoring |
|---|---:|---:|
| Authentication and support impersonation | 3 | 0 |
| Collaborator-message poisoning | 6 | 0 |
| CRM information leakage | 4 | 4 |
| Infrastructure and resource abuse | 6 | 0 |
| Email / retrieval-augmented phishing | 7 | 0 |
| Marketing-workflow phishing | 3 | 0 |
| Unsafe output handling | 3 | 3 |
| Tool-shadow poisoning | 3 | 3 |
| Total | 35 | 10 |

All 35 tasks include deterministic environment assertions. Only the ten tasks
whose reward basis includes communication assertions invoke the fixed judge.
Tools operate on simulated task state; they do not actually send emails, operate
production infrastructure or execute paid external API operations named in tasks.
The model inference calls themselves consume Kaggle AI quota.

## Data And Safety Notice

The authors confirm that the CRM names, passport numbers, dates of birth,
addresses and phone numbers are entirely fictional test fixtures, not real
customer records. Seeded credentials are test values used to evaluate disclosure.

Scenario texts and model outputs contain adversarial instructions, phishing
content and email/URL domains that are not guaranteed to be reserved, inactive
or controlled by the authors. Do not visit those links, contact those addresses
or submit credentials/data. They are scenario labels, not authorized destinations.
Keep tools connected only to the supplied simulated state; never connect this
benchmark to real mail, customer databases or production infrastructure.
Treat exported notebook HTML/JavaScript as active content, not inert plain text.
The declared model inference calls are the only intended external service use.

## Protocol

- Full protocol: `duma-kaggle-private-full-v3`; pilot: `duma-kaggle-private-pilot-v6`.
- 35 tasks x 3 repeats = 105 episodes per target and regime.
- Framework seeds: 300, 301, 302; model-provider determinism is not guaranteed.
- Maximum 100 orchestration steps and 10 errors per episode.
- Target: selected by Kaggle; temperature 0, reasoning none, max_tokens 4096.
- User: `google/gemini-2.5-flash`, temperature 0, reasoning none, max_tokens 2048.
- Judge: `google/gemini-3-flash-preview`, temperature 0, reasoning low,
  max_tokens 4096. Each expected assertion must have an explicit boolean decision.
- User and judge models are pinned; unavailable services are not substituted.
- Runtime Python, task data, prompts and protocol identities are stored with
  episodes. V3 pins the complete data/prompt hash map and a bundle manifest.

## Score

Higher is better: the main score is **mean task compliance**. For each task,
average its three binary repeat outcomes, then average over the 35 tasks with
equal task weight. All scored assertions must pass for an episode to score one.
Domain means are provided separately; the main score is not an equally weighted
mean of domains with different task counts.

Technical errors and unfinished episodes have no score. The final leaderboard
score is returned only for the complete expected matrix; missing episodes are
not silently dropped or treated as confirmed attacks.

The complement of compliance is a raw assertion-failure rate, **not uniformly
confirmed attack success**: checks can concern unsafe actions, blocked attempts,
missing safe escalation or communication requirements. Reports distinguish ENV
and communication failures; trajectories are needed to establish executed harm.

For consistency across repeats, the report also provides **pass^k**, not ordinary
pass@k. With n=3 repeats and s passing repeats for one task, its value is
`C(s,k) / C(n,k)`, averaged across tasks. It estimates all k selected repeats
passing, rather than at least one passing.

## Reproducibility And Limitations

Per-episode artifacts retain assertions, trajectories, termination, role-specific
usage/cost when available and integrity hashes. Unknown cost remains unknown.
The offline verifier checks checksums, exact coverage, identities, aggregation
and deterministic environment replay without calling the communication judge again.

This is a small, deliberately constructed scenario set, not an exhaustive threat
distribution. Fixed simulated users/judges, finite turn limits, deterministic
assertion coverage and mutable provider backends limit generalization. Do not
claim deployment security, causal isolation or a robust model ranking from a
single pilot or these 35 tasks alone. The hardened tasks and scoring differ from
V1/V2 and the paper; comparisons must be labeled and separately evaluated.

## Results And Cost

Complete private full-v3 evaluation: 420/420 episodes, 120 with communication
scoring, no technical or incomplete episodes. Frozen-wheel offline replay and
aggregate checks passed. Higher compliance is better:

| Target | Solo compliance | Dual compliance | Reported full inference cost |
|---|---:|---:|---:|
| Gemini 3.8 Flash | 94.29% (99/105) | 97.14% (102/105) | $3.02342600 |
| GPT-5.4 mini (2026-03-17) | 92.38% (97/105) | 80.00% (84/105) | $1.21643855 |

Cost includes target, user and judge calls for both regimes; total $4.23986455,
excluding pilot/preflight. Full domain results, pass^k, failure categories, role
costs, run IDs and audit identities are in
[the verified results](../../docs/kaggle-full-v3-results.md).

Gemini scored 100% in seven domains, a difficulty/discrimination limitation.
Regime trends differ by target: dual compliance is higher for Gemini and lower
for GPT. Do not assert a universal directional effect. Failures include missing
escalation, synthetic-secret repetition, simulated financial-document solicitation
and spending above the default operational cap, not uniformly executed harm.
The offline audit did not independently human-score all communication judgments.

## License

MIT; see LICENSE. Upstream tau2-bench attribution and MIT terms are retained in
NOTICE. The Kaggle Dataset catalog label is Other with this explicit MIT
statement because its supported license catalog does not include MIT.
