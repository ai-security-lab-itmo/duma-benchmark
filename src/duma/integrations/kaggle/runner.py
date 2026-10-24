"""Execute one manifest episode using DUMA's original orchestration/scoring."""

import json
import uuid
from pathlib import Path
from typing import Literal

from duma.data_model.simulation import TerminationReason
from duma.integrations.kaggle.backend import KaggleBackend
from duma.integrations.kaggle.artifacts import (
    episode_lock,
    read_checkpoint,
    write_checkpoint,
)
from duma.integrations.kaggle.protocol import KaggleProtocol, fingerprint_files
from duma.runtime import get_tasks, run_task
from duma.utils.generation import use_generation_backends
from duma.utils.utils import DATA_DIR
from duma.integrations.kaggle.metrics import aggregate, evaluation_rows


def run_episode(
    llm,
    protocol: KaggleProtocol,
    *,
    domain: str,
    task_id: str,
    regime: Literal["solo", "dual"],
    trial: int,
    output_dir: Path,
    service_models: dict | None = None,
    resume: bool = True,
) -> bool:
    import kaggle_benchmarks as kbench

    protocol.require_service_models()
    protocol.validate_data(DATA_DIR)
    if regime not in protocol.regimes or not 0 <= trial < len(protocol.trial_seeds):
        raise ValueError("Regime or trial is not present in the protocol")
    if task_id not in protocol.tasks.get(domain, []):
        raise ValueError("Task is not present in the protocol")
    models = kbench.llms if service_models is None else service_models
    missing = {protocol.user.model, protocol.judge.model} - models.keys()
    if missing:
        raise ValueError(f"Pinned service models are unavailable: {sorted(missing)}")
    target_id = str(getattr(llm, "model", llm.name))
    backends = {
        "duma-kaggle-target": KaggleBackend(llm, "target"),
        "duma-kaggle-user": KaggleBackend(models[protocol.user.model], "user"),
        "duma-kaggle-judge": KaggleBackend(models[protocol.judge.model], "judge"),
    }
    metadata = {
        "protocol": protocol.model_dump(),
        "protocol_hash": protocol.fingerprint,
        "data_hash": fingerprint_files(
            DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
        ),
        "source_hash": fingerprint_files(Path(__file__).parents[2], ("**/*.py",)),
        "sdk_version": kbench.__version__,
        "target_model": target_id,
        "service_models": {
            role: str(getattr(models[slug], "model", models[slug].name))
            for role, slug in {
                "user": protocol.user.model,
                "judge": protocol.judge.model,
            }.items()
        },
        "domain": domain,
        "task_id": task_id,
        "regime": regime,
        "trial": trial,
        "seed": protocol.trial_seeds[trial],
    }
    # Model names can contain slashes; never use them directly as path components.
    from hashlib import sha256

    episode_id = sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{episode_id}.json"
    with episode_lock(path):
        if path.exists():
            previous = read_checkpoint(path, metadata)
            if not resume:
                raise FileExistsError(f"Episode already exists: {path.name}")
            if previous["status"] == "completed":
                return previous["score"] == 1.0
            attempts = path.parent / "attempts"
            attempts.mkdir(exist_ok=True)
            write_checkpoint(
                attempts / f"{path.stem}-{uuid.uuid4().hex}.json", previous
            )
        return _execute(
            path, metadata, backends, protocol, domain, task_id, regime, trial
        )


def _execute(path, metadata, backends, protocol, domain, task_id, regime, trial):
    artifact = {"metadata": metadata, "score": None}
    try:
        task = get_tasks(domain, task_ids=[task_id])[0]
        with use_generation_backends(backends):
            simulation = run_task(
                domain=domain,
                task=task,
                agent="llm_agent_solo" if regime == "solo" else "llm_agent",
                user="dummy_user" if regime == "solo" else "user_simulator",
                llm_agent="duma-kaggle-target",
                llm_user="duma-kaggle-user",
                llm_args_agent=protocol.agent.generation_args(),
                llm_args_user=protocol.user.generation_args(),
                judge_model="duma-kaggle-judge",
                judge_args=protocol.judge.generation_args(),
                max_steps=protocol.max_steps,
                max_errors=protocol.max_errors,
                seed=protocol.trial_seeds[trial],
                on_simulation_complete=lambda result: artifact.update(
                    simulation=result.model_dump(mode="json")
                ),
            )
        artifact["simulation"] = simulation.model_dump(mode="json")
        if simulation.termination_reason in {
            TerminationReason.MAX_STEPS,
            TerminationReason.TOO_MANY_ERRORS,
        }:
            artifact["status"] = (
                "incomplete"
                if simulation.termination_reason == TerminationReason.MAX_STEPS
                else "technical_error"
            )
            raise RuntimeError("Episode did not finish; inspect the saved trajectory")
        artifact["status"] = "completed"
        artifact["score"] = simulation.reward_info.reward
        artifact["outcome"] = (
            "success" if artifact["score"] == 1.0 else "policy_failure"
        )
        return simulation.reward_info.reward == 1.0
    except Exception as exc:
        artifact.setdefault("status", "technical_error")
        artifact["error_type"] = type(exc).__name__
        judge_response = backends["duma-kaggle-judge"].last_response
        if judge_response is not None:
            artifact["judge_response"] = judge_response.model_dump(mode="json")
        raise
    finally:
        artifact["usage_by_role"] = {
            role.removeprefix("duma-kaggle-"): backend.calls
            for role, backend in backends.items()
        }
        write_checkpoint(path, artifact)


def run_matrix(llm, protocol, *, regime, output_dir, service_models=None):
    """Return a leaderboard score only for the complete current evaluation grid."""
    import kaggle_benchmarks as kbench

    protocol.require_service_models()
    protocol.validate_data(DATA_DIR)
    models = kbench.llms if service_models is None else service_models
    if {protocol.user.model, protocol.judge.model} - models.keys():
        raise ValueError("Pinned service models are unavailable")
    errors = []
    for row in evaluation_rows(protocol, regime):
        try:
            run_episode(
                llm,
                protocol,
                output_dir=output_dir,
                service_models=service_models,
                **{key: value for key, value in row.items() if key != "seed"},
            )
        except Exception as exc:
            errors.append({**row, "error_type": type(exc).__name__})
    target = str(getattr(llm, "model", llm.name))
    source = fingerprint_files(Path(__file__).parents[2], ("**/*.py",))
    data = fingerprint_files(
        DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
    )
    artifacts = []
    for path in output_dir.glob("*.json"):
        item = json.loads(path.read_text())
        meta = item.get("metadata", {})
        if (
            meta.get("target_model") == target
            and meta.get("protocol_hash") == protocol.fingerprint
            and meta.get("regime") == regime
            and meta.get("source_hash") == source
            and meta.get("data_hash") == data
            and meta.get("sdk_version") == kbench.__version__
        ):
            artifacts.append(item)
    summary = aggregate(artifacts, protocol, target, regime)
    summary["errors"] = errors
    # Do not put summaries among episode JSON files.
    reports = output_dir / "summaries"
    reports.mkdir(parents=True, exist_ok=True)
    from hashlib import sha256

    report_id = sha256(f"{target}/{protocol.fingerprint}/{regime}".encode()).hexdigest()
    write_checkpoint(reports / f"{report_id}.json", summary)
    if errors or not summary["complete"]:
        raise RuntimeError("Incomplete evaluation matrix; no leaderboard score")
    return summary
