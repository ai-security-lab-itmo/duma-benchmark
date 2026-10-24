"""Offline integrity and ENV replay audit of the frozen full-v1 attempts.

No generation calls, checkpoint edits, judge reruns or score corrections.
Run from the repository root with the same source/data as the hosted run.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

from loguru import logger

import duma
from duma.data_model.simulation import SimulationRun
from duma.evaluator.evaluator_env import EnvironmentEvaluator
from duma.integrations.kaggle.artifacts import read_checkpoint
from duma.integrations.kaggle.protocol import fingerprint_files, load_protocol
from duma.registry import registry
from duma.runtime import get_tasks
from duma.utils.utils import DATA_DIR


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / ".cache/kaggle-full-v1-results"
FIRST_REPORT_SHA = "0fd91f8fd6ec180fb89448eadd0a715c45757758b64bd442ed8dd7921ae31418"


def row_key(artifact):
    meta = artifact["metadata"]
    return tuple(
        meta[key]
        for key in ("target_model", "domain", "task_id", "regime", "trial", "seed")
    )


def load_selected(results=RESULTS, root=ROOT):
    first = results / "verified-summary.json"
    if hashlib.sha256(first.read_bytes()).hexdigest() != FIRST_REPORT_SHA:
        raise ValueError("First-attempt report changed")
    report = json.loads(
        (results / "verified-summary-after-technical-retry.json").read_text()
    )
    selected = {}
    inventory = []
    for path in sorted(results.glob("**/duma-kaggle-results/*.json")):
        raw = json.loads(path.read_text())
        artifact = read_checkpoint(path, raw["metadata"])
        key = row_key(artifact)
        if key in selected:
            raise ValueError(f"Duplicate original row: {key}")
        selected[key] = (path, artifact)
        inventory.append(
            {"path": str(path.relative_to(root)), "hash": artifact["artifact_hash"]}
        )
    if len(selected) != 420:
        raise ValueError("Expected exactly 420 original rows")
    path = root / report["retry_path"]
    raw = json.loads(path.read_text())
    retry = read_checkpoint(path, raw["metadata"])
    key = row_key(retry)
    original = selected[key][1]
    if (
        original["artifact_hash"] != report["replaced_artifact_hash"]
        or retry["artifact_hash"] != report["retry_artifact_hash"]
        or original["status"] != "technical_error"
        or retry["status"] != "completed"
        or original["metadata"] != retry["metadata"]
    ):
        raise ValueError("Invalid technical retry replacement")
    selected[key] = (path, retry)
    inventory.append(
        {"path": str(path.relative_to(root)), "hash": retry["artifact_hash"]}
    )
    return selected, inventory


def replay(artifact):
    meta = artifact["metadata"]
    simulation = SimulationRun.model_validate(artifact["simulation"])
    task = get_tasks(meta["domain"], [meta["task_id"]])[0]
    reward = EnvironmentEvaluator.calculate_reward(
        registry.get_env_constructor(meta["domain"]),
        task,
        simulation.messages,
        solo_mode=meta["regime"] == "solo",
    )
    actual = [check.model_dump(mode="json") for check in reward.env_assertions or []]
    expected = artifact["simulation"]["reward_info"]["env_assertions"] or []
    if actual != expected:
        raise ValueError(f"ENV replay mismatch: {row_key(artifact)}")
    return [check["env_assertion"]["func_name"] for check in actual if not check["met"]]


def stage_selected(selected, destination):
    """Prepare local resume inputs only; never upload or generate responses."""
    for path, artifact in selected.values():
        checked = read_checkpoint(path, artifact["metadata"])
        if checked != artifact or checked["status"] != "completed":
            raise ValueError("Only unchanged completed checkpoints can be staged")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = []
    for path, artifact in selected.values():
        meta = artifact["metadata"]
        folder = destination / meta["target_model"].replace("/", "__") / meta["regime"]
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / path.name
        shutil.copyfile(path, target)
        manifest.append(
            {
                "path": str(target.relative_to(destination)),
                "artifact_hash": artifact["artifact_hash"],
            }
        )
    (destination / "resume-manifest.json").write_text(
        json.dumps(
            {
                "diagnostic_only": True,
                "public_release_blocked": True,
                "reason": "See docs/kaggle-full-v1-audit.md before leaderboard promotion",
                "checkpoints": manifest,
            },
            indent=2,
        )
        + "\n"
    )


def audit():
    selected, inventory = load_selected()
    protocol = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v1.json")
    expected = {
        (target, domain, task, regime, trial, seed)
        for target in ("google/gemini-3.8-flash", "openai/gpt-5.4-mini-2026-03-17")
        for domain, tasks in protocol.tasks.items()
        for task in tasks
        for regime in protocol.regimes
        for trial, seed in enumerate(protocol.trial_seeds)
    }
    if set(selected) != expected:
        raise ValueError("Unexpected task/model/seed matrix")
    wheel = next(RESULTS.glob("**/duma_kaggle_runtime-*.whl"))
    digest = hashlib.sha256()
    # Hosted identity covers the wheel subset, not the full research package.
    with ZipFile(wheel) as archive:
        for name in sorted(
            name
            for name in archive.namelist()
            if name.startswith("duma/") and name.endswith(".py")
        ):
            content = archive.read(name)
            relative = name.removeprefix("duma/")
            if content != (Path(duma.__file__).parent / relative).read_bytes():
                raise ValueError(
                    f"Imported source differs from frozen v1 wheel: {name}; use an environment with that wheel installed"
                )
            digest.update(name.removeprefix("duma/").encode() + b"\0")
            digest.update(content + b"\0")
    source = digest.hexdigest()
    data = fingerprint_files(
        DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
    )
    failures = []
    counts = Counter()
    for path, artifact in selected.values():
        meta = artifact["metadata"]
        if meta["source_hash"] != source or meta["data_hash"] != data:
            raise ValueError("Replay source/data differs from hosted identity")
        if (
            meta["protocol"] != protocol.model_dump()
            or meta["protocol_hash"] != protocol.fingerprint
            or meta["sdk_version"] != "0.6.2"
            or meta["service_models"]
            != {"user": protocol.user.model, "judge": protocol.judge.model}
        ):
            raise ValueError("Protocol/SDK/service model identity mismatch")
        if artifact["status"] != "completed":
            raise ValueError("Selected matrix is incomplete")
        failed_env = replay(artifact)
        if artifact["score"] == 0:
            counts[meta["domain"]] += 1
            failures.append(
                {
                    "metadata": meta,
                    "artifact_path": str(path.relative_to(ROOT)),
                    "artifact_hash": artifact["artifact_hash"],
                    "failed_env_assertions": failed_env,
                    "nl_assertions": artifact["simulation"]["reward_info"][
                        "nl_assertions"
                    ],
                    "tool_calls": [
                        call
                        for message in artifact["simulation"]["messages"]
                        for call in message.get("tool_calls") or []
                    ],
                }
            )
    return {
        "version": "full-v1-offline-env-audit-1",
        "source_hash": source,
        "data_hash": data,
        "replayed_rows": len(selected),
        "raw_attempts": len(inventory),
        "env_replay_mismatches": 0,
        "judge_rerun": False,
        "canonical_scores_changed": False,
        "failed_rows": len(failures),
        "failed_rows_by_domain": dict(sorted(counts.items())),
        "artifacts": inventory,
        "failures": failures,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage-dir",
        type=Path,
        help="New local directory for verified resume inputs; no upload",
    )
    args = parser.parse_args()
    logger.remove()
    result = audit()
    # Exclusive creation keeps an earlier audit immutable.
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if args.stage_dir:
        stage_selected(load_selected()[0], args.stage_dir)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"artifacts", "failures"}
            },
            indent=2,
        )
    )
