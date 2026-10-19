"""Execute one manifest episode using DUMA's original orchestration/scoring."""

import json
from pathlib import Path
from typing import Literal

from duma.data_model.simulation import TerminationReason
from duma.integrations.kaggle.backend import KaggleBackend
from duma.integrations.kaggle.protocol import KaggleProtocol, fingerprint_files
from duma.run import get_tasks, run_task
from duma.utils.generation import use_generation_backends
from duma.utils.utils import DATA_DIR


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
    if path.exists():
        raise FileExistsError(
            f"Episode already exists; refusing to overwrite {path.name}"
        )

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
            )
        artifact["simulation"] = simulation.model_dump(mode="json")
        if simulation.termination_reason in {
            TerminationReason.MAX_STEPS,
            TerminationReason.TOO_MANY_ERRORS,
        }:
            raise RuntimeError("Episode did not finish; inspect the saved trajectory")
        artifact["status"] = "completed"
        artifact["score"] = simulation.reward_info.reward
        return simulation.reward_info.reward == 1.0
    except Exception as exc:
        artifact["status"] = "technical_error"
        artifact["error_type"] = type(exc).__name__
        raise
    finally:
        artifact["usage_by_role"] = {
            role.removeprefix("duma-kaggle-"): backend.calls
            for role, backend in backends.items()
        }
        path.write_text(json.dumps(artifact, indent=2))
