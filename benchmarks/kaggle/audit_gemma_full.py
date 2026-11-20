"""Offline full Gemma audit and explicitly separate partial-trace rescoring."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import duma
from loguru import logger

from duma.data_model.message import AssistantMessage, ToolMessage, UserMessage
from duma.data_model.simulation import SimulationRun
from duma.data_model.tasks import RewardType
from duma.evaluator.evaluator_env import EnvironmentEvaluator
from duma.integrations.kaggle.artifacts import read_checkpoint, write_checkpoint
from duma.integrations.kaggle.metrics import aggregate, evaluation_rows
from duma.integrations.kaggle.protocol import fingerprint_files, load_protocol
from duma.registry import registry
from duma.runtime import get_tasks
from duma.utils.utils import DATA_DIR


def replayable_messages(messages):
    """Retain text and executed calls; never execute terminal calls without results."""
    result, pending = [], []
    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, ToolMessage):
            raise ValueError("Unmatched tool result")
        if (
            not isinstance(message, (AssistantMessage, UserMessage))
            or not message.is_tool_call()
        ):
            result.append(message)
            index += 1
            continue
        matched = []
        responses = []
        cursor = index + 1
        for call in message.tool_calls:
            if cursor == len(messages):
                pending.append({"id": call.id, "name": call.name})
                continue
            response = messages[cursor]
            if not isinstance(response, ToolMessage) or response.id != call.id:
                raise ValueError("Malformed tool-call/result ordering")
            matched.append(call)
            responses.append(response)
            cursor += 1
        if matched or message.content:
            result.append(message.model_copy(update={"tool_calls": matched or None}))
        result.extend(responses)
        index = cursor
    return result, pending


def derived_views(rows):
    """No row status or artifact is changed by the derived arithmetic."""
    total = len(rows)
    completed = [row for row in rows if row["status"] == "completed"]
    passed = sum(row["original_score"] == 1 for row in completed)
    unresolved = [row for row in rows if row["status"] != "completed"]
    trace_scores = [row["trace_assertion_score"] for row in rows]
    return {
        "episodes": total,
        "completed": len(completed),
        "completed_passes": passed,
        "completed_failures": len(completed) - passed,
        "unfinished": len(unresolved),
        "unfinished_with_failed_env": sum(
            bool(row["failed_env"]) for row in unresolved
        ),
        "execution_coverage": len(completed) / total,
        "original_complete_matrix_score": passed / total if not unresolved else None,
        "execution_aware_score_unfinished_zero": passed / total,
        "observed_trace_assertion_score": (
            sum(trace_scores) / total
            if all(score is not None for score in trace_scores)
            else None
        ),
        "completion_score_bounds": [passed / total, (passed + len(unresolved)) / total],
    }


def audit(results, bundle):
    root = Path(__file__).parent
    base = load_protocol(root / "protocol.full.v4.json")
    base.validate_data(DATA_DIR)
    helpers = runpy.run_path(str(root / "verify_results.py"))
    source = helpers["wheel_identity"](bundle)
    data = fingerprint_files(
        DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
    )
    adapter = hashlib.sha256((root / "gemma_full.py").read_bytes()).hexdigest()
    expected_profile = {
        "version": "duma-kaggle-full-gemma-default-v1",
        "base_protocol": base.model_dump(),
        "base_protocol_hash": base.fingerprint,
        "adapter_sha256": adapter,
        "target_model": "google/gemma-4-31b",
        "target_reasoning": "parameter omitted; provider default",
        "target_generation": {"temperature": 0, "max_tokens": 4096},
        "solo_transport_prefix": ".",
    }
    profile_hash = hashlib.sha256(
        json.dumps(expected_profile, sort_keys=True).encode()
    ).hexdigest()
    matrix = SimpleNamespace(
        tasks=base.tasks,
        regimes=base.regimes,
        trial_seeds=base.trial_seeds,
        fingerprint=profile_hash,
    )
    expected = {
        (r["regime"], r["domain"], r["task_id"], r["trial"], r["seed"])
        for regime in base.regimes
        for r in evaluation_rows(base, regime)
    }
    seen, grouped, rows = set(), defaultdict(list), []
    for path in sorted(results.glob("**/duma-gemma-full-results/*.json")):
        raw = json.loads(path.read_text())
        item = read_checkpoint(path, raw["metadata"])
        meta = item["metadata"]
        key = tuple(meta[k] for k in ("regime", "domain", "task_id", "trial", "seed"))
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate Gemma row")
        seen.add(key)
        if (
            meta["protocol"] != expected_profile
            or meta["protocol_hash"] != profile_hash
            or meta["source_hash"] != source
            or meta["data_hash"] != data
            or meta["sdk_version"] != "0.6.2"
            or meta["target_model"] != "google/gemma-4-31b"
            or meta["service_models"]
            != {"user": base.user.model, "judge": base.judge.model}
        ):
            raise ValueError("Gemma profile/source/data/SDK identity mismatch")
        sim = SimulationRun.model_validate(item["simulation"])
        task = get_tasks(meta["domain"], [meta["task_id"]])[0]
        criteria = task.evaluation_criteria
        messages, pending = replayable_messages(sim.messages)
        replay = EnvironmentEvaluator.calculate_reward(
            registry.get_env_constructor(meta["domain"]),
            task,
            messages,
            solo_mode=meta["regime"] == "solo",
        )
        env = [check.model_dump(mode="json") for check in replay.env_assertions or []]
        if not env:
            raise ValueError("Expected nonempty ENV assertions")
        needs_nl = RewardType.NL_ASSERTION in criteria.reward_basis
        reward = item["simulation"]["reward_info"]
        trace_score = None
        if item["status"] == "completed":
            if pending or sim.termination_reason.value in {
                "max_steps",
                "too_many_errors",
            }:
                raise ValueError("Completed row is unfinished")
            if reward["reward_basis"] != [
                b.value for b in criteria.reward_basis
            ] or env != (reward["env_assertions"] or []):
                raise ValueError("Scoring basis or ENV replay mismatch")
            nl = reward["nl_assertions"] or []
            if (
                Counter(c["nl_assertion"] for c in nl)
                != Counter(criteria.nl_assertions if needs_nl else [])
                or any(type(c["met"]) is not bool for c in nl)
                or len(item["usage_by_role"]["judge"]) != int(needs_nl)
            ):
                raise ValueError("NL assertion/call schema mismatch")
            trace_score = float(
                all(c["met"] for c in env) and all(c["met"] for c in nl)
            )
            if trace_score != item["score"] or trace_score != reward["reward"]:
                raise ValueError("Completed score mismatch")
        else:
            if item["score"] is not None or sim.termination_reason.value not in {
                "max_steps",
                "too_many_errors",
            }:
                raise ValueError("Invalid unfinished row")
            # No successful verdict when an unfinished row needs missing NL scoring.
            if not all(c["met"] for c in env):
                trace_score = 0.0
            elif not needs_nl:
                trace_score = 1.0
        grouped[meta["regime"]].append(item)
        rows.append(
            {
                **{
                    k: meta[k] for k in ("regime", "domain", "task_id", "trial", "seed")
                },
                "path": str(path.relative_to(results)),
                "original_artifact_hash": item["artifact_hash"],
                "status": item["status"],
                "termination_reason": sim.termination_reason.value,
                "original_score": item["score"],
                "trace_assertion_score": trace_score,
                "env_assertions_replayed": env,
                "nl_scored": needs_nl and item["status"] == "completed",
                "failed_env": [
                    c["env_assertion"]["func_name"] for c in env if not c["met"]
                ],
                "pending_calls_excluded_from_state_replay": pending,
            }
        )
    if seen != expected:
        raise ValueError("Missing Gemma artifacts")
    matrices = {}
    for regime in base.regimes:
        original = aggregate(grouped[regime], matrix, "google/gemma-4-31b", regime)
        paths = list(
            results.glob(f"**/duma-gemma-full-results/summaries/{regime}.json")
        )
        if len(paths) != 1:
            raise ValueError("Missing or duplicate hosted summary")
        stored = json.loads(paths[0].read_text())
        from duma.integrations.kaggle.artifacts import checksum

        if stored["artifact_hash"] != checksum(stored) or any(
            not helpers["same_summary"](json.loads(json.dumps(v)), stored.get(k))
            for k, v in original.items()
        ):
            raise ValueError("Hosted aggregation mismatch")
        matrices[regime] = derived_views(
            [row for row in rows if row["regime"] == regime]
        )
    return {
        "version": "duma-gemma-offline-trace-rescore-v1",
        "profile_hash": profile_hash,
        "source_hash": source,
        "data_hash": data,
        "adapter_sha256": adapter,
        "artifacts_verified": len(rows),
        "nl_scored_rows_verified": sum(row["nl_scored"] for row in rows),
        "matrices": matrices,
        "rows": rows,
        "scope": "Post-hoc partial-trace assessment, not normal episode completion or frozen V4. No model/judge calls. Original artifacts unchanged.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logger.remove()
    if args.output.exists():
        raise FileExistsError("Audit reports are immutable")
    report = audit(args.results, args.bundle)
    write_checkpoint(args.output, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("artifacts_verified", "nl_scored_rows_verified", "matrices")
            },
            indent=2,
        )
    )
