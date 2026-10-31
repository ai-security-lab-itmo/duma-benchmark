"""Bounded transport experiments, never execute returned tool calls."""

from copy import deepcopy
import json
from pathlib import Path

import kaggle_benchmarks as kbench

from duma.agent.llm_agent import LLMSoloAgent
from duma.data_model.message import SystemMessage, UserMessage
from duma.data_model.simulation import SimulationRun
from duma.integrations.kaggle.artifacts import checksum, write_checkpoint
from duma.integrations.kaggle.backend import KaggleBackend
from duma.registry import registry
from duma.runtime import get_tasks


def transform_request(request, variant):
    result = deepcopy(request)
    if variant in {"empty_parameters", "named_empty_parameters"}:
        for tool in result.get("tools", []):
            function = tool.get("function")
            if not isinstance(function, dict):
                raise ValueError("Malformed function tool")
            function.setdefault(
                "parameters", {"type": "object", "properties": {}, "required": []}
            )
        if variant == "named_empty_parameters":
            for tool in result.get("tools", []):
                if not tool["function"].get("description"):
                    tool["function"]["description"] = tool["function"]["name"]
    elif variant == "filled_descriptions":
        for tool in result.get("tools", []):
            if not tool["function"].get("description"):
                tool["function"]["description"] = tool["function"]["name"]
    elif variant == "serial_tools":
        result["parallel_tool_calls"] = False
    elif variant == "auto":
        result["tool_choice"] = "auto"
    elif variant == "bootstrap_instruction":
        users = [m for m in result["messages"] if m["role"] == "user"]
        if len(users) != 1 or users[0]["content"] != ".":
            raise ValueError("Unexpected solo prefix")
        users[0]["content"] = (
            "Proceed with the ticket above using available tools only. If complete, call `done`."
        )
    elif variant != "original":
        raise ValueError("Unknown variant")
    return result


@kbench.task(name="duma-openweight-wire-probe-v1")
def wire_probe(llm) -> dict:
    model = str(getattr(llm, "model", llm.name))
    if model not in WIRE_SNAPSHOTS:
        return {"skipped": "Target-specific diagnostics; no inference or score"}
    kbench.config.interactive_mode = False
    snapshot = WIRE_SNAPSHOTS[model]
    if checksum(snapshot) != snapshot["artifact_hash"]:
        raise ValueError("Snapshot changed")
    sim = SimulationRun.model_validate(snapshot["simulation"])
    domain = snapshot["metadata"]["domain"]
    env = registry.get_env_constructor(domain)()
    history = list(sim.messages)
    if model.startswith("google/gemma-"):
        # Probe the first decision after all three mandatory reads succeeded.
        history = history[:6]
        task = get_tasks(domain, [snapshot["metadata"]["task_id"]])[0]
        agent = LLMSoloAgent(
            tools=env.get_tools(), domain_policy=env.get_policy(), task=task
        )
        tools = agent.tools
        messages = [
            SystemMessage(role="system", content=sim.assistant_system_prompt),
            UserMessage(role="user", content="."),
        ] + history
        params = {"tool_choice": "required"}
        variants = ("original", "auto", "bootstrap_instruction")
    else:
        # Fresh dual agents never see the greeting sent to the user simulator.
        messages = [
            SystemMessage(role="system", content=sim.assistant_system_prompt)
        ] + history[1:]
        tools = env.get_tools()
        params = {"reasoning": "low"}
        variants = globals().get(
            "OSS_VARIANTS", ("original", "empty_parameters", "serial_tools")
        )
    create = llm.client.chat.completions.create
    report = {
        "model": model,
        "source_artifact_hash": snapshot["artifact_hash"],
        "scope": "Three bounded request variants, no tool execution or benchmark scoring",
        "cases": [],
    }
    current, variant = {}, "original"

    def capture(*args, **kwargs):
        if args:
            raise ValueError("Expected keyword completion request")
        request = transform_request(kwargs, variant)
        current["original_request"] = kwargs
        current["effective_request"] = request
        response = create(**request)
        current["raw_response"] = response.model_dump(mode="json")
        return response

    llm.client.chat.completions.create = capture
    try:
        for variant in variants:
            current = {"variant": variant}
            try:
                response = KaggleBackend(llm, "target")(
                    messages=messages,
                    tools=tools,
                    temperature=0,
                    max_tokens=4096,
                    seed=300,
                    **params,
                )
                current.update(
                    status="valid", parsed_response=response.model_dump(mode="json")
                )
            except Exception as exc:
                current.update(
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:2000],
                    http_status=getattr(exc, "status_code", None),
                )
            report["cases"].append(current)
            write_checkpoint(Path("/kaggle/working/openweight-wire-probe.json"), report)
            print(
                json.dumps(
                    {k: current.get(k) for k in ("variant", "status", "error_message")}
                ),
                flush=True,
            )
            if current.get("http_status") in {401, 403, 404, 429}:
                break
    finally:
        llm.client.chat.completions.create = create
    return {
        "scope": report["scope"],
        "statuses": [c["status"] for c in report["cases"]],
    }


if __name__ == "__main__":
    wire_probe.run(kbench.llm)
