# Hardened task revision (2026-10-24)

Domain changes were selectively imported from `73c439c`; experimental dumps,
legacy runner configuration and provider changes were not imported.

The inventory remains 35 tasks across eight domains. Five former control/task
IDs were replaced with adversarial scenarios. All tasks retain environment
scoring; ten (previously nine) also use communication assertions.

A separate follow-up fixes campaign identity visibility, aligns infrastructure
policy with its budget checks, removes fixed refusal phrases from solo tickets,
and selects field-aware identifier and financial-exfiltration assertions. Legacy
assertion methods remain available for historical replay. Low-entropy labels
such as HIGH are covered by communication assertions, not substring matching.
The mail assertion is a bounded deterministic check, not a general phishing
classifier: it inspects the specified finance recipients, unsafe destination
hosts and explicit requests for the specified document categories.

The infrastructure over-quota assertion records an attempted action, even when
the tool rejects it; it must not be described as successful resource consumption.

These changes alter policies, scenarios and scoring. Existing paper and Kaggle
v1/v2 results must retain their original source/data/protocol hashes. Run this
revision under a new protocol and do not mix it into historical leaderboards.
