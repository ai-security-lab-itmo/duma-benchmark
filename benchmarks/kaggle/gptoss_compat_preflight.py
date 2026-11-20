"""Separate low-reasoning GPT-OSS transport profile; original assertions unchanged."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import kaggle_benchmarks as kbench

from duma.integrations.kaggle.artifacts import write_checkpoint
from duma.integrations.kaggle.protocol import KaggleProtocol, load_protocol
from duma.integrations.kaggle.runner import run_episode


def normalize_tools(request):
    result = deepcopy(request)
    for tool in result.get("tools", []):
        if tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise ValueError("Expected OpenAI function tools")
        function = tool["function"]
        if not function.get("description"):
            function["description"] = function["name"]
    return result


class GPTOSSProtocol(KaggleProtocol):
    base_protocol: dict
    wire_adapter_sha256: str

    def validate_data(self, data_dir):
        base = KaggleProtocol.model_validate(self.base_protocol)
        base.validate_data(data_dir)
        for field in (
            "user",
            "judge",
            "regimes",
            "trial_seeds",
            "max_steps",
            "max_errors",
            "tasks",
            "task_data_sha256",
        ):
            if getattr(self, field) != getattr(base, field):
                raise ValueError("Changed base task, service or limit")
        if (
            self.version != "duma-gptoss-low-named-descriptions-v1"
            or self.agent.reasoning != "low"
            or self.target_reasoning_overrides is not None
        ):
            raise ValueError("Unexpected compatibility profile")
        if (self.agent.temperature, self.agent.max_tokens) != (
            base.agent.temperature,
            base.agent.max_tokens,
        ):
            raise ValueError("Changed generation limits")


def make_protocol(base, adapter_sha):
    data = base.model_dump()
    data.pop("target_reasoning_overrides", None)
    data.update(
        version="duma-gptoss-low-named-descriptions-v1",
        base_protocol=base.model_dump(),
        wire_adapter_sha256=adapter_sha,
        agent={**base.agent.model_dump(), "reasoning": "low"},
    )
    return GPTOSSProtocol.model_validate(data)


@kbench.task(name="duma-gptoss-compatible-xss-preflight-v1")
def compatible_xss(llm) -> dict:
    if str(getattr(llm, "model", llm.name)) != "openai/gpt-oss-120b":
        return {"skipped": "Target-only compatibility row, no inference"}
    kbench.config.interactive_mode = False
    base = load_protocol(os.environ["DUMA_KAGGLE_PROTOCOL"])
    protocol = make_protocol(base, GPTOSS_ADAPTER_SHA256)
    create = llm.client.chat.completions.create
    wire = {"profile_hash": protocol.fingerprint, "target_calls": []}
    output = Path("/kaggle/working/duma-kaggle-results")
    wire_path = Path("/kaggle/working/gptoss-compat-wire.json")

    def capture(*args, **kwargs):
        if args:
            raise ValueError("Expected keyword request")
        if kwargs.get("model") != llm.model:
            return create(**kwargs)
        changed = normalize_tools(kwargs)
        call = {"original_request": kwargs, "effective_request": changed}
        wire["target_calls"].append(call)
        try:
            response = create(**changed)
            call["response"] = response.model_dump(mode="json")
            return response
        except Exception as exc:
            call.update(error_type=type(exc).__name__, error_message=str(exc)[:1200])
            raise
        finally:
            write_checkpoint(wire_path, wire)

    llm.client.chat.completions.create = capture
    try:
        try:
            run_episode(
                llm,
                protocol,
                domain="output_handling",
                task_id="output_xss_block",
                regime="dual",
                trial=0,
                output_dir=output,
            )
        except Exception as exc:
            wire.update(error_type=type(exc).__name__, error_message=str(exc)[:1200])
    finally:
        llm.client.chat.completions.create = create
        write_checkpoint(wire_path, wire)
    item = json.loads(next(output.glob("*.json")).read_text())
    return {
        "scope": "One separate compatibility episode, not a full score",
        "status": item["status"],
        "score": item["score"],
        "profile_hash": protocol.fingerprint,
    }


if __name__ == "__main__":
    compatible_xss.run(kbench.llm)
