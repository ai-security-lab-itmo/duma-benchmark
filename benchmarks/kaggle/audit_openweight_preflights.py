"""Offline diagnostics audit, preserving incomplete rows and valid failures."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy

from loguru import logger

from duma.data_model.simulation import SimulationRun, TerminationReason
from duma.data_model.tasks import RewardType
from duma.evaluator.evaluator_env import EnvironmentEvaluator
from duma.integrations.kaggle.artifacts import read_checkpoint, write_checkpoint
from duma.integrations.kaggle.protocol import (
    KaggleProtocol,
    fingerprint_files,
    load_protocol,
)
from duma.registry import registry
from duma.runtime import get_tasks
from duma.utils.utils import DATA_DIR


EXPECTED = {
    ("collab", "collab_poisoning_logs", "solo"),
    ("collab", "collab_poisoning_logs", "dual"),
    ("output_handling", "output_xss_block", "dual"),
}


def audit(results, model, mode, protocol_path, bundle, dual_only=False):
    if mode == "gptoss-compat":
        path = Path(__file__).with_name("gptoss_compat_preflight.py")
        config = json.loads(protocol_path.read_text())
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != config["wire_adapter_sha256"]
        ):
            raise ValueError("Compatibility adapter identity changed")
        nodes = [
            n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)
        ]
        ns = {"KaggleProtocol": KaggleProtocol}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
        base = ns["GPTOSSProtocol"].model_validate(config)
    else:
        base = load_protocol(protocol_path)
    base.validate_data(DATA_DIR)
    helper = runpy.run_path(str(Path(__file__).with_name("verify_results.py")))
    source = helper["wheel_identity"](bundle)
    data = fingerprint_files(
        DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
    )
    folder = {
        "gemma": "gemma-duma-diagnostics",
        "gemma-native": "gemma-native-solo-diagnostics",
        "standard": "duma-kaggle-results",
        "gptoss-compat": "duma-kaggle-results",
    }[mode]
    rows, seen = [], set()
    expected = {key for key in EXPECTED if not dual_only or key[2] == "dual"}
    if mode == "gemma-native":
        expected = {("collab", "collab_poisoning_logs", "solo")}
    if mode == "gptoss-compat":
        expected = {("output_handling", "output_xss_block", "dual")}
    for path in sorted(results.glob(f"**/{folder}/*.json")):
        raw = json.loads(path.read_text())
        item = read_checkpoint(path, raw["metadata"])
        meta = item["metadata"]
        key = meta["domain"], meta["task_id"], meta["regime"]
        if (
            key not in expected
            or key in seen
            or meta["target_model"] != model
            or meta["seed"] != 300
        ):
            raise ValueError("Unexpected/duplicate row or target")
        seen.add(key)
        if mode in {"gemma", "gemma-native"}:
            if (
                meta["base_protocol_hash"] != base.fingerprint
                or meta["base_source_hash"] != source
                or meta["data_hash"] != data
            ):
                raise ValueError("Changed Gemma base identity")
            if (
                meta["user"] != base.user.model_dump()
                or meta["judge"] != base.judge.model_dump()
                or (meta["max_steps"], meta["max_errors"]) != (100, 10)
            ):
                raise ValueError("Changed services/limits")
            if (
                meta["target_generation"] != {"temperature": 0, "max_tokens": 4096}
                or meta["target_reasoning"] != "parameter omitted; provider default"
                or meta["solo_transport_prefix"] != ("." if key[2] == "solo" else None)
            ):
                raise ValueError("Changed Gemma adapter settings")
            version = (
                "duma-gemma26-native-solo-probe-v1"
                if mode == "gemma-native"
                else "duma-gemma-omit-reasoning-probe-v1"
            )
            if meta["diagnostic_version"] != version:
                raise ValueError("Unexpected Gemma diagnostic version")
            if mode == "gemma-native" and meta.get("target_api") != "genai":
                raise ValueError("Unexpected native API identity")
        else:
            helper["verify_episode"](item, base, source, data)
        row = {
            "domain": key[0],
            "task_id": key[1],
            "regime": key[2],
            "status": item["status"],
            "score": item["score"],
            "artifact_hash": item["artifact_hash"],
        }
        if item["status"] != "completed":
            if (
                item["status"] not in {"technical_error", "incomplete"}
                or item["score"] is not None
            ):
                raise ValueError("Unfinished row has invalid status/score")
            row.update(
                error_type=item.get("error_type"),
                error_message=item.get("error_message"),
                termination_reason=(item.get("simulation") or {}).get(
                    "termination_reason"
                ),
            )
        else:
            sim = SimulationRun.model_validate(item["simulation"])
            if sim.termination_reason in {
                TerminationReason.MAX_STEPS,
                TerminationReason.TOO_MANY_ERRORS,
            }:
                raise ValueError("Incomplete row marked completed")
            task = get_tasks(key[0], [key[1]])[0]
            reward = sim.reward_info.model_dump(mode="json")
            replay = EnvironmentEvaluator.calculate_reward(
                registry.get_env_constructor(key[0]),
                task,
                sim.messages,
                solo_mode=key[2] == "solo",
            )
            env = [c.model_dump(mode="json") for c in replay.env_assertions or []]
            if env != (reward["env_assertions"] or []) or reward["reward_basis"] != [
                v.value for v in task.evaluation_criteria.reward_basis
            ]:
                raise ValueError("Assertion replay/basis mismatch")
            needs_nl = RewardType.NL_ASSERTION in task.evaluation_criteria.reward_basis
            nl = reward["nl_assertions"] or []
            if Counter(c["nl_assertion"] for c in nl) != Counter(
                task.evaluation_criteria.nl_assertions if needs_nl else []
            ):
                raise ValueError("NL identity mismatch")
            if any(type(c["met"]) is not bool for c in nl) or len(
                item["usage_by_role"].get("judge", [])
            ) != int(needs_nl):
                raise ValueError("NL decision/judge call mismatch")
            score = float(all(c["met"] for c in env + nl))
            if score != item["score"] or score != reward["reward"]:
                raise ValueError("Incorrect stored score")
            row.update(
                failed_env=[
                    c["env_assertion"]["func_name"] for c in env if not c["met"]
                ],
                failed_nl=[c for c in nl if not c["met"]],
                nl_scored=needs_nl,
                termination_reason=sim.termination_reason.value,
            )
        rows.append(row)
    if seen != expected:
        raise ValueError("Missing diagnostic rows")
    return {
        "target": model,
        "scope": "Compatibility episodes only, not full benchmark score",
        "all_completed": all(r["status"] == "completed" for r in rows),
        "base_or_variant_protocol_hash": base.fingerprint,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "protocol", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode",
        choices=("gemma", "gemma-native", "standard", "gptoss-compat"),
        required=True,
    )
    parser.add_argument("--dual-only", action="store_true")
    args = parser.parse_args()
    logger.remove()
    if args.output.exists():
        raise FileExistsError("Immutable audit output")
    report = audit(
        args.results, args.model, args.mode, args.protocol, args.bundle, args.dual_only
    )
    write_checkpoint(args.output, report)
    print(json.dumps(report, indent=2))
