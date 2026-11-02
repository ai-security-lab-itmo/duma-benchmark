"""Separate native GenAI solo diagnostic; no task or policy modifications."""

import os
import json
from pathlib import Path
import kaggle_benchmarks as kbench
from duma.integrations.kaggle.protocol import load_protocol

kbench.config.interactive_mode = False


@kbench.task(name="duma-gemma26-native-solo-probe-v1")
def gemma_native_solo_probe(llm) -> dict:
    from kaggle_benchmarks.kaggle.model_proxy import ModelProxy
    from duma.data_model.message import UserMessage
    from duma.data_model.simulation import TerminationReason
    from duma.integrations.kaggle.artifacts import write_checkpoint
    from duma.integrations.kaggle.backend import KaggleBackend
    from duma.integrations.kaggle.protocol import fingerprint_files
    from duma.runtime import get_tasks, run_task
    from duma.utils.generation import use_generation_backends
    from duma.utils.utils import DATA_DIR

    model = str(getattr(llm, "model", llm.name))
    if model != "google/gemma-4-26b-a4b":
        return {
            "scope": "Experimental Gemma diagnostics only",
            "skipped": "Select Gemma",
        }
    protocol = load_protocol(os.environ["DUMA_KAGGLE_PROTOCOL"])
    protocol.validate_data(DATA_DIR)
    output = Path("/kaggle/working/gemma-native-solo-diagnostics")
    output.mkdir(exist_ok=True)
    rows = [
        ("collab", "collab_poisoning_logs", "solo"),
    ]
    outcomes = {}
    for domain, task_id, regime in rows:
        target = KaggleBackend(ModelProxy(model=model, api="genai"), "target")
        user_backend = KaggleBackend(kbench.llms[protocol.user.model], "user")
        judge = KaggleBackend(kbench.llms[protocol.judge.model], "judge")

        def target_adapter(*, messages, **kwargs):
            if "reasoning" in kwargs:
                raise ValueError("Diagnostic requires reasoning parameter omission")
            adapted = list(messages)
            if regime == "solo" and not any(m.role == "user" for m in adapted):
                index = next(
                    (i for i, m in enumerate(adapted) if m.role != "system"),
                    len(adapted),
                )
                adapted.insert(index, UserMessage(role="user", content="."))
            return target(messages=adapted, **kwargs)

        item = {
            "metadata": {
                "diagnostic_version": "duma-gemma26-native-solo-probe-v1",
                "scope": "One native solo diagnostic row; not frozen V4 leaderboard scores",
                "base_protocol_hash": protocol.fingerprint,
                "domain": domain,
                "task_id": task_id,
                "regime": regime,
                "seed": 300,
                "target_model": model,
                "target_api": "genai",
                "target_reasoning": "parameter omitted; provider default",
                "solo_transport_prefix": "." if regime == "solo" else None,
                "max_steps": 100,
                "max_errors": 10,
                "target_generation": {"temperature": 0, "max_tokens": 4096},
                "base_source_hash": fingerprint_files(
                    Path(__import__("duma").__file__).parent, ("**/*.py",)
                ),
                "data_hash": fingerprint_files(
                    DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
                ),
                "user": protocol.user.model_dump(),
                "judge": protocol.judge.model_dump(),
            },
            "status": "technical_error",
            "score": None,
        }
        try:
            task = get_tasks(domain, [task_id])[0]
            with use_generation_backends(
                {
                    "gemma-diagnostic-target": target_adapter,
                    "gemma-diagnostic-user": user_backend,
                    "gemma-diagnostic-judge": judge,
                }
            ):
                simulation = run_task(
                    domain=domain,
                    task=task,
                    agent="llm_agent_solo" if regime == "solo" else "llm_agent",
                    user="dummy_user" if regime == "solo" else "user_simulator",
                    llm_agent="gemma-diagnostic-target",
                    llm_user="gemma-diagnostic-user",
                    llm_args_agent={"temperature": 0, "max_tokens": 4096},
                    llm_args_user=protocol.user.generation_args(),
                    judge_model="gemma-diagnostic-judge",
                    judge_args=protocol.judge.generation_args(),
                    max_steps=100,
                    max_errors=10,
                    seed=300,
                    on_simulation_complete=lambda sim: item.update(
                        simulation=sim.model_dump(mode="json")
                    ),
                )
            item["simulation"] = simulation.model_dump(mode="json")
            if simulation.termination_reason == TerminationReason.MAX_STEPS:
                item["status"] = "incomplete"
            elif simulation.termination_reason == TerminationReason.TOO_MANY_ERRORS:
                item["status"] = "technical_error"
            else:
                item["status"] = "completed"
                item["score"] = simulation.reward_info.reward
        except Exception as exc:
            item["error_type"] = type(exc).__name__
            item["error_message"] = str(exc)[:1000]
            if judge.last_response is not None:
                item["judge_response"] = judge.last_response.model_dump(mode="json")
        item["usage_by_role"] = {
            "target": target.calls,
            "user": user_backend.calls,
            "judge": judge.calls,
        }
        write_checkpoint(output / (task_id + "-" + regime + ".json"), item)
        outcomes[task_id + "-" + regime] = item["status"]
        print(
            json.dumps(
                {
                    "row": task_id + "-" + regime,
                    "status": item["status"],
                    "score": item["score"],
                }
            )
        )
    return {
        "scope": "Native GenAI Gemma solo compatibility; not V4 ranking",
        "outcomes": outcomes,
    }


if __name__ == "__main__":
    gemma_native_solo_probe.run(kbench.llm)
