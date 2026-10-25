# V3 Publication Checklist (Public Transition Not Executed)

Keep this stage private until explicitly approving the final publication action.
Readiness is not a security score. Never publish an incomplete matrix or present
historical V1/V2 measurements as hardened V3 results.

## Current Gate Status (2026-10-25)

- Passed: hosted readiness, three preflights, 36-row pilot, full 420-row audit,
  exact coverage/ENV replay and stored judge/aggregate consistency.
- Reviewed: all 38 failed action/output records; categories and interpretation
  limits recorded in `docs/kaggle-full-v3-results.md`.
- Prepared: benchmark card results, domain/pass^k/cost tables and frozen hashes.
- Confirmed private: full Dataset version 2, pilot Dataset version 1; both Ready,
  with Other catalog label and explicit MIT terms. Solo/dual tasks are version 1.
- Pending: final public-content/privacy review, exact notebook/version inventory,
  anonymous-access checks after transition and explicit publication approval.
- Not executed: Dataset/task/notebook publication, public collection, branch merge.

Audit checksum: `3e260d93f8dd64ab2087dd899bf5fc37aae7cb9e3504b181fe2e72ab4a87643a`.
Offline checks do not independently establish the semantic correctness of all
LLM judge decisions. No claim of a full human communication audit is made.

## Validation

- Final full/pilot bundles have matching runtime Python and task-data identities.
- Final hosted installation succeeds with supplied Kaggle SDK and `pip check`.
- Dual, solo and communication preflights finish with valid saved artifacts.
- Pilot-v6 covers 36 episodes on the two named targets, with technical attempts
  retained and no retries of valid score-zero results.
- Full-v3 covers 420 episodes with identical frozen settings; complete matrices
  have verified summaries and independent environment replay.
- Judge assertions and role-specific cost/unknown-cost fields are audited.
- Failure categories are reviewed; compliance failures are not all called ASR.

## Release Materials

- Fill BENCHMARK_CARD.md results only from the complete checksummed audit report.
- Retain protocol/source/data/bundle/SDK identities and reproduction commands.
- Include LICENSE and NOTICE in each distributed bundle and wheel.
- Keep Kaggle's Other catalog label paired with explicit MIT terms.
- Check for credentials, private outputs, unintended dataset attachments and
  links that an unauthenticated reader cannot access.
- Do not publish check/smoke/preflight/pilot tasks as benchmark rankings.
- Do not merge the Kaggle runtime PR merely because private runs finish.

## Public Transition

1. Review exact full-v3 solo/dual task versions, Dataset versions and backing
   notebooks. Freeze them; do not rely on an implicit latest version.
2. Update the full Dataset's public-facing title/description and explicitly set
   public visibility only at the approved publication step. Keep MIT terms and
   both notices. Do not publish the diagnostic pilot Dataset by default.
3. Publish only duma-v3-solo and duma-v3-dual and their reviewed backing notebooks.
   Kaggle's task publish command normally publishes its backing notebook too.
4. Group the two scored tasks into one DUMA-Bench collection through the Kaggle
   web interface, preserving separate regime scores and the stated main metric.
5. Verify every public link and source/data asset without author-only access.
6. Record the published URLs, versions and final audit checksum in release notes.

These are instructions for the later publication decision, not authorization to
execute visibility changes. No publish command belongs in an unattended monitor.
