"""Complete-matrix, task-balanced scoring using only the standard library."""

from collections import defaultdict
import math

from duma.integrations.kaggle.artifacts import checksum


def evaluation_rows(protocol, regime):
    if regime not in protocol.regimes:
        raise ValueError("Unknown evaluation regime")
    return [
        dict(domain=domain, task_id=task, regime=regime, trial=trial, seed=seed)
        for domain, tasks in protocol.tasks.items()
        for task in tasks
        for trial, seed in enumerate(protocol.trial_seeds)
    ]


def aggregate(artifacts, protocol, target_model, regime):
    expected = {
        (r["domain"], r["task_id"], r["trial"])
        for r in evaluation_rows(protocol, regime)
    }
    seen = set()
    completed = defaultdict(list)
    statuses = defaultdict(int)
    identity = None
    costs = defaultdict(list)
    for item in artifacts:
        meta = item["metadata"]
        if item.get("artifact_hash") != checksum(item):
            raise ValueError("Artifact checksum mismatch")
        if (
            meta["protocol_hash"] != protocol.fingerprint
            or meta["target_model"] != target_model
            or meta["regime"] != regime
        ):
            raise ValueError("Mixed protocols, targets or regimes")
        current = (meta["data_hash"], meta["source_hash"], meta["sdk_version"])
        if identity is not None and identity != current:
            raise ValueError("Mixed code, data or SDK versions")
        identity = current
        key = (meta["domain"], meta["task_id"], meta["trial"])
        if (
            key not in expected
            or key in seen
            or meta["seed"] != protocol.trial_seeds[meta["trial"]]
        ):
            raise ValueError("Unexpected, duplicate or mismatched evaluation row")
        seen.add(key)
        status = item["status"]
        if status not in {"completed", "technical_error", "incomplete"}:
            raise ValueError("Unknown episode status")
        statuses[status] += 1
        if status == "completed":
            score = item["score"]
            if (
                score not in (0.0, 1.0)
                or item["simulation"]["reward_info"]["reward"] != score
            ):
                raise ValueError("Invalid episode score")
            completed[key[:2]].append(score)
        for role, calls in item.get("usage_by_role", {}).items():
            for call in calls:
                parts = [
                    call.get("input_tokens_cost_nanodollars"),
                    call.get("output_tokens_cost_nanodollars"),
                ]
                costs[role].append(
                    sum(parts) / 1e9 if all(p is not None for p in parts) else None
                )
    complete = len(completed) == sum(
        len(v) for v in protocol.tasks.values()
    ) and statuses["completed"] == len(expected)
    by_domain = {}
    pass_ks = {}
    score = None
    if complete:
        task_rates = {
            key: sum(values) / len(values) for key, values in completed.items()
        }
        score = sum(task_rates.values()) / len(task_rates)
        for domain in protocol.tasks:
            rates = [rate for (name, _), rate in task_rates.items() if name == domain]
            by_domain[domain] = sum(rates) / len(rates)
        for k in range(1, len(protocol.trial_seeds) + 1):
            pass_ks[k] = sum(
                math.comb(int(sum(v)), k) / math.comb(len(v), k)
                for v in completed.values()
            ) / len(completed)
    return {
        "complete": complete,
        "expected": len(expected),
        "observed": len(seen),
        "completed": statuses["completed"],
        "coverage": statuses["completed"] / len(expected),
        "statuses": dict(statuses),
        "score": score,
        "failure_rate": None if score is None else 1 - score,
        "by_domain": by_domain,
        "pass_power_k": pass_ks,
        "cost_usd_by_role": {
            role: sum(values) if all(v is not None for v in values) else None
            for role, values in costs.items()
        },
    }
