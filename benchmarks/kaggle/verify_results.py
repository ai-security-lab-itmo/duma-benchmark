"""Audit one frozen hosted matrix offline; never call models or alter results.

Use the bundle's installed minimal wheel and set DUMA_DATA_DIR to its data folder.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import duma
from loguru import logger

from duma.data_model.simulation import SimulationRun
from duma.data_model.tasks import RewardType
from duma.evaluator.evaluator_env import EnvironmentEvaluator
from duma.integrations.kaggle.artifacts import read_checkpoint, write_checkpoint
from duma.integrations.kaggle.metrics import aggregate, evaluation_rows
from duma.integrations.kaggle.protocol import fingerprint_files, load_protocol
from duma.registry import registry
from duma.runtime import get_tasks
from duma.utils.utils import DATA_DIR


def wheel_identity(bundle):
    with ZipFile(bundle) as archive:
        wheels = [name for name in archive.namelist() if name.endswith(".whl")]
        if len(wheels) != 1:
            raise ValueError("Expected one frozen runtime wheel")
        import io

        with ZipFile(io.BytesIO(archive.read(wheels[0]))) as wheel:
            for name in wheel.namelist():
                if name.startswith("duma/") and name.endswith(".py"):
                    path = Path(duma.__file__).parent / name.removeprefix("duma/")
                    if not path.exists() or path.read_bytes() != wheel.read(name):
                        raise ValueError(
                            f"Imported source differs from frozen wheel: {name}"
                        )
    return fingerprint_files(Path(duma.__file__).parent, ("**/*.py",))


def verify_episode(item, protocol, source, data, sdk="0.6.2"):
    meta = item["metadata"]
    if (
        meta["protocol"] != protocol.model_dump()
        or meta["protocol_hash"] != protocol.fingerprint
        or meta["source_hash"] != source
        or meta["data_hash"] != data
        or meta["sdk_version"] != sdk
        or meta["service_models"]
        != {"user": protocol.user.model, "judge": protocol.judge.model}
    ):
        raise ValueError("Protocol/source/data/SDK/service identity mismatch")
    if item["status"] != "completed":
        if item["score"] is not None:
            raise ValueError("Unfinished episode has a score")
        return None
    sim = SimulationRun.model_validate(item["simulation"])
    if sim.termination_reason.value in {"max_steps", "too_many_errors"}:
        raise ValueError("Unfinished trajectory marked completed")
    task = get_tasks(meta["domain"], [meta["task_id"]])[0]
    criteria = task.evaluation_criteria
    reward = item["simulation"]["reward_info"]
    if reward["reward_basis"] != [basis.value for basis in criteria.reward_basis]:
        raise ValueError("Reward basis differs from task")
    replayed = EnvironmentEvaluator.calculate_reward(
        registry.get_env_constructor(meta["domain"]),
        task,
        sim.messages,
        solo_mode=meta["regime"] == "solo",
    )
    env = [check.model_dump(mode="json") for check in replayed.env_assertions or []]
    if env != (reward["env_assertions"] or []):
        raise ValueError("ENV replay mismatch")
    nl = reward["nl_assertions"] or []
    needs_nl = RewardType.NL_ASSERTION in criteria.reward_basis
    expected = criteria.nl_assertions if needs_nl else []
    if Counter(check["nl_assertion"] for check in nl) != Counter(expected):
        raise ValueError("Communication assertion identity mismatch")
    if any(type(check["met"]) is not bool for check in nl):
        raise ValueError("Non-boolean communication decision")
    calls = item["usage_by_role"].get("judge", [])
    if len(calls) != int(needs_nl):
        raise ValueError("Unexpected communication judge calls")
    score = float(
        all(check["met"] for check in env) and all(check["met"] for check in nl)
    )
    if score != item["score"] or score != reward["reward"]:
        raise ValueError("Stored score differs from scored assertions")
    if item["outcome"] != ("success" if score == 1 else "policy_failure"):
        raise ValueError("Stored outcome differs from score")
    return {
        "failed_env": [
            check["env_assertion"]["func_name"] for check in env if not check["met"]
        ],
        "failed_nl": [check for check in nl if not check["met"]],
        "nl_scored": needs_nl,
        "message_counts": dict(Counter(message.role for message in sim.messages)),
    }


def audit(results, bundle, protocol, targets, sdk="0.6.2"):
    protocol.validate_data(DATA_DIR)
    source = wheel_identity(bundle)
    data = fingerprint_files(
        DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
    )
    grouped = defaultdict(list)
    failures, attempts, identities = [], [], set()
    nl_rows = 0
    expected = {
        (target, row["domain"], row["task_id"], regime, row["trial"], row["seed"])
        for target in targets
        for regime in protocol.regimes
        for row in evaluation_rows(protocol, regime)
    }
    for path in sorted(results.glob("**/duma-kaggle-results/*.json")):
        raw = json.loads(path.read_text())
        item = read_checkpoint(path, raw["metadata"])
        meta = item["metadata"]
        key = tuple(
            meta[k]
            for k in ("target_model", "domain", "task_id", "regime", "trial", "seed")
        )
        if key not in expected or key in identities:
            raise ValueError(f"Unexpected or duplicate row: {key}")
        identities.add(key)
        check = verify_episode(item, protocol, source, data, sdk)
        grouped[(meta["target_model"], meta["regime"])].append(item)
        entry = {
            "path": str(path.relative_to(results)),
            "artifact_hash": item["artifact_hash"],
            "status": item["status"],
        }
        attempts.append(entry)
        if check:
            nl_rows += check["nl_scored"]
            if item["score"] == 0:
                failures.append(
                    {
                        **entry,
                        "row": dict(
                            zip(
                                (
                                    "target",
                                    "domain",
                                    "task_id",
                                    "regime",
                                    "trial",
                                    "seed",
                                ),
                                key,
                            )
                        ),
                        **check,
                    }
                )
    summaries = {}
    for path in sorted(results.glob("**/duma-kaggle-results/summaries/*.json")):
        raw = json.loads(path.read_text())
        stored = (
            read_checkpoint(path, raw.get("metadata")) if "metadata" in raw else raw
        )
        # Summary checksums use the same canonical artifact hash, without metadata.
        from duma.integrations.kaggle.artifacts import checksum

        if stored.get("artifact_hash") != checksum(stored):
            raise ValueError("Summary checksum mismatch")
        for identity, items in grouped.items():
            target, regime = identity
            name = (
                hashlib.sha256(
                    f"{target}/{protocol.fingerprint}/{regime}".encode()
                ).hexdigest()
                + ".json"
            )
            if path.name != name:
                continue
            if identity in summaries:
                raise ValueError("Duplicate matrix summary")
            actual = aggregate(items, protocol, target, regime)
            if any(
                stored.get(field) != json.loads(json.dumps(value))
                for field, value in actual.items()
            ):
                raise ValueError("Hosted summary differs from independent aggregation")
            summaries[identity] = actual
    complete = (
        identities == expected
        and len(summaries) == len(targets) * len(protocol.regimes)
        and all(s["complete"] for s in summaries.values())
    )
    return {
        "complete": complete,
        "expected": len(expected),
        "observed": len(identities),
        "protocol_hash": protocol.fingerprint,
        "source_hash": source,
        "data_hash": data,
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "sdk_version": sdk,
        "nl_scored_rows": nl_rows,
        "matrices": [
            {"target": target, "regime": regime, **summary}
            for (target, regime), summary in sorted(summaries.items())
        ],
        "failures": failures,
        "checkpoints": attempts,
        "scope": "Raw compliance scores; not uniformly confirmed attack success. No model/judge reruns during audit.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logger.remove()
    if args.output.exists():
        raise FileExistsError("Audit reports are immutable; use a new output path")
    report = audit(args.results, args.bundle, load_protocol(args.protocol), args.target)
    write_checkpoint(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("complete", "expected", "observed", "nl_scored_rows")
            }
        )
    )
    if not report["complete"]:
        raise SystemExit("Incomplete evaluation; no final leaderboard score")
