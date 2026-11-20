"""Append to the verified 0.1.3 hosted bootstrap; run as a private Gemma task.

The builder injects GEMMA_ADAPTER_SHA256 and GEMMA_REGIME before this module.
This is a separate protocol profile, never a frozen V4 leaderboard row.
"""

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import kaggle_benchmarks as kbench

from duma.data_model.message import UserMessage
from duma.data_model.simulation import TerminationReason
from duma.integrations.kaggle.artifacts import read_checkpoint, write_checkpoint
from duma.integrations.kaggle.backend import KaggleBackend
from duma.integrations.kaggle.metrics import aggregate, evaluation_rows
from duma.integrations.kaggle.protocol import fingerprint_files, load_protocol
from duma.runtime import get_tasks, run_task
from duma.utils.generation import use_generation_backends
from duma.utils.utils import DATA_DIR


def adapt_target_messages(messages, regime):
    adapted = list(messages)
    if regime == "solo" and not any(m.role == "user" for m in adapted):
        index = next(
            (i for i, m in enumerate(adapted) if m.role != "system"), len(adapted)
        )
        adapted.insert(index, UserMessage(role="user", content="."))
    return adapted


def gemma_full(llm, regime) -> float:
    model = str(getattr(llm, "model", llm.name))
    if model != "google/gemma-4-31b":
        raise ValueError("This profile requires Gemma 4 31B, not a readiness model")
    base = load_protocol(os.environ["DUMA_KAGGLE_PROTOCOL"])
    base.require_service_models()
    base.validate_data(DATA_DIR)
    if {base.user.model, base.judge.model} - kbench.llms.keys():
        raise ValueError("Pinned user/judge unavailable; no automatic replacements")
    profile = {
        "version": "duma-kaggle-full-gemma-default-v1",
        "base_protocol": base.model_dump(),
        "base_protocol_hash": base.fingerprint,
        "adapter_sha256": GEMMA_ADAPTER_SHA256,
        "target_model": model,
        "target_reasoning": "parameter omitted; provider default",
        "target_generation": {"temperature": 0, "max_tokens": 4096},
        "solo_transport_prefix": ".",
    }
    profile_hash = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    identity = {
        "protocol": profile,
        "protocol_hash": profile_hash,
        "source_hash": fingerprint_files(Path(__import__("duma").__file__).parent, ("**/*.py",)),
        "data_hash": fingerprint_files(DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")),
        "sdk_version": kbench.__version__,
        "target_model": model,
        "service_models": {"user": base.user.model, "judge": base.judge.model},
    }
    # The original aggregator also works with an explicitly identified profile.
    matrix = SimpleNamespace(
        tasks=base.tasks, regimes=base.regimes, trial_seeds=base.trial_seeds,
        fingerprint=profile_hash,
    )
    output = Path(os.environ.get("DUMA_GEMMA_OUTPUT", "/kaggle/working/duma-gemma-full-results"))
    output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for row in evaluation_rows(base, regime):
        meta = {**identity, **row}
        episode_id = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        path = output / (episode_id + ".json")
        if path.exists():
            item = read_checkpoint(path, meta)
            if item["status"] != "completed":
                raise RuntimeError("Existing unfinished row requires an explicit retry protocol")
            artifacts.append(item)
            continue
        target = KaggleBackend(llm, "target")
        user = KaggleBackend(kbench.llms[base.user.model], "user")
        judge = KaggleBackend(kbench.llms[base.judge.model], "judge")

        def target_adapter(*, messages, **kwargs):
            if "reasoning" in kwargs:
                raise ValueError("Gemma profile requires omission, not reasoning=none")
            return target(messages=adapt_target_messages(messages, regime), **kwargs)

        item = {"metadata": meta, "status": "technical_error", "score": None}
        try:
            task = get_tasks(row["domain"], [row["task_id"]])[0]
            with use_generation_backends({
                "gemma-profile-target": target_adapter,
                "gemma-profile-user": user,
                "gemma-profile-judge": judge,
            }):
                simulation = run_task(
                    domain=row["domain"], task=task,
                    agent="llm_agent_solo" if regime == "solo" else "llm_agent",
                    user="dummy_user" if regime == "solo" else "user_simulator",
                    llm_agent="gemma-profile-target", llm_user="gemma-profile-user",
                    llm_args_agent=profile["target_generation"].copy(),
                    llm_args_user=base.user.generation_args(),
                    judge_model="gemma-profile-judge", judge_args=base.judge.generation_args(),
                    max_steps=base.max_steps, max_errors=base.max_errors, seed=row["seed"],
                    on_simulation_complete=lambda sim: item.update(simulation=sim.model_dump(mode="json")),
                )
            item["simulation"] = simulation.model_dump(mode="json")
            if simulation.termination_reason == TerminationReason.MAX_STEPS:
                item["status"] = "incomplete"
            elif simulation.termination_reason != TerminationReason.TOO_MANY_ERRORS:
                item["status"] = "completed"
                item["score"] = simulation.reward_info.reward
                item["outcome"] = "success" if item["score"] == 1 else "policy_failure"
        except Exception as exc:
            item["error_type"] = type(exc).__name__
            item["error_message"] = str(exc)[:1000]
            if judge.last_response is not None:
                item["judge_response"] = judge.last_response.model_dump(mode="json")
        item["usage_by_role"] = {"target": target.calls, "user": user.calls, "judge": judge.calls}
        write_checkpoint(path, item)
        artifacts.append(read_checkpoint(path, meta))
        print(json.dumps({**row, "status": item["status"], "score": item["score"]}), flush=True)
    summary = aggregate(artifacts, matrix, model, regime)
    (output / "summaries").mkdir(exist_ok=True)
    write_checkpoint(output / "summaries" / (regime + ".json"), {
        **summary, "profile": profile, "profile_hash": profile_hash,
        "scope": "Gemma provider-default reasoning; not frozen V4",
    })
    if not summary["complete"]:
        raise RuntimeError("Incomplete Gemma evaluation matrix; no leaderboard score")
    return summary["score"]


kbench.config.interactive_mode = False


@kbench.task(name="duma-gemma-default-solo-v1")
def gemma_solo(llm) -> dict:
    if str(getattr(llm, "model", llm.name)) != "google/gemma-4-31b":
        return {"skipped": "Gemma-only task; readiness run has no security score"}
    return {"score": gemma_full(llm, "solo"), "profile": "duma-kaggle-full-gemma-default-v1"}


@kbench.task(name="duma-gemma-default-dual-v1")
def gemma_dual(llm) -> dict:
    if str(getattr(llm, "model", llm.name)) != "google/gemma-4-31b":
        return {"skipped": "Gemma-only task; readiness run has no security score"}
    return {"score": gemma_full(llm, "dual"), "profile": "duma-kaggle-full-gemma-default-v1"}


if __name__ == "__main__":
    {"solo": gemma_solo, "dual": gemma_dual}[GEMMA_REGIME].run(kbench.llm)
