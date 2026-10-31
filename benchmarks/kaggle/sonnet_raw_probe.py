"""Append to frozen bootstrap/adapter: inspect responses without running tools."""

import json
from pathlib import Path

import kaggle_benchmarks as kbench

from duma.data_model.simulation import SimulationRun
from duma.data_model.message import SystemMessage
from duma.integrations.kaggle.artifacts import checksum, write_checkpoint
from duma.registry import registry


@kbench.task(name="duma-sonnet55-raw-response-probe")
def raw_response_probe(llm) -> dict:
    if str(getattr(llm, "model", llm.name)) != "anthropic/claude-sonnet-5-5@default":
        return {"skipped": "Diagnostic only, no inference or safety score"}
    kbench.config.interactive_mode = False
    output = Path("/kaggle/working/sonnet-raw-probe.json")
    report = {
        "scope": "Four diagnostic calls, no tools executed, no safety score",
        "cases": [],
    }
    create = llm.client.chat.completions.create
    current = {}

    def observed_create(*args, **kwargs):
        current["request"] = {k: v for k, v in kwargs.items() if k != "messages"}
        response = create(*args, **kwargs)
        current["raw_response"] = response.model_dump(mode="json")
        return response

    llm.client.chat.completions.create = observed_create
    try:
        for snapshot in PROBE_SNAPSHOTS:
            if checksum(snapshot) != snapshot["artifact_hash"]:
                raise ValueError("Probe snapshot checksum mismatch")
            sim = SimulationRun.model_validate(snapshot["simulation"])
            messages = [
                SystemMessage(role="system", content=sim.assistant_system_prompt)
            ] + list(sim.messages)
            if any(
                m.role == "tool" and getattr(m, "requestor", "assistant") != "assistant"
                for m in sim.messages
            ):
                raise ValueError("User tools need perspective projection")
            tools = registry.get_env_constructor(
                snapshot["metadata"]["domain"]
            )().get_tools()
            params = snapshot["metadata"]["protocol"]["agent"].copy()
            params.pop("model", None)
            params.pop("reasoning", None)
            params["seed"] = snapshot["metadata"]["seed"]
            for repeat in range(2):
                current = {
                    "task_id": snapshot["metadata"]["task_id"],
                    "repeat": repeat,
                    "source_artifact_hash": snapshot["artifact_hash"],
                }
                try:
                    response = SonnetBackend(llm, "target")(
                        messages=messages, tools=tools, **params
                    )
                    current["parsed_response"] = response.model_dump(mode="json")
                    current["status"] = "valid"
                except Exception as exc:
                    current.update(
                        status="error",
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:1200],
                    )
                report["cases"].append(current)
                write_checkpoint(output, report)
                raw = current.get("raw_response", {})
                print(
                    json.dumps(
                        {
                            "task": current["task_id"],
                            "repeat": repeat,
                            "status": current["status"],
                            "choices": [
                                {
                                    "finish_reason": c.get("finish_reason"),
                                    "message_is_none": c.get("message") is None,
                                }
                                for c in raw.get("choices") or []
                            ],
                            "usage": raw.get("usage"),
                        }
                    ),
                    flush=True,
                )
    finally:
        llm.client.chat.completions.create = create
    return {
        "scope": report["scope"],
        "statuses": [c["status"] for c in report["cases"]],
    }


if __name__ == "__main__":
    raw_response_probe.run(kbench.llm)
