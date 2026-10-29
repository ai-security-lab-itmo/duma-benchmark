"""Separate hosted Sonnet profile; frozen V4 runtime and scoring are reused."""

import os
from pathlib import Path

import kaggle_benchmarks as kbench

from duma.data_model.message import AssistantMessage, UserMessage
from duma.integrations.kaggle.backend import KaggleBackend
from duma.integrations.kaggle import runner
from duma.integrations.kaggle.protocol import (
    KaggleProtocol,
    ModelSettings,
    load_protocol,
)
from duma.integrations.kaggle.runner import run_episode, run_matrix


def adapt_messages(messages):
    adapted = list(messages)
    if adapted and isinstance(adapted[-1], AssistantMessage):
        if adapted[-1].tool_calls:
            raise ValueError(
                "Cannot append a user message before pending tools execute"
            )
        adapted.append(UserMessage(role="user", content="."))
    return adapted


class SonnetBackend(KaggleBackend):
    def __call__(self, *, messages, **kwargs):
        if self.role == "target":
            messages = adapt_messages(messages)
        return super().__call__(messages=messages, **kwargs)


class SonnetAgentSettings(ModelSettings):
    extra_body: dict | None = None


class SonnetProtocol(KaggleProtocol):
    agent: SonnetAgentSettings
    base_protocol: dict
    hosted_adapter_sha256: str
    assistant_prefill_transport_suffix: str = "."

    def validate_data(self, data_dir):
        base = KaggleProtocol.model_validate(self.base_protocol)
        base.validate_data(data_dir)
        fields = (
            "user",
            "judge",
            "tasks",
            "regimes",
            "trial_seeds",
            "max_steps",
            "max_errors",
            "task_data_sha256",
        )
        if any(getattr(self, field) != getattr(base, field) for field in fields):
            raise ValueError(
                "Sonnet profile must preserve base tasks, services and limits"
            )
        if (
            self.version != "duma-kaggle-sonnet55-transport-v2"
            or base.version != "duma-kaggle-private-full-v4"
        ):
            raise ValueError("Unexpected Sonnet or base protocol version")
        if (
            self.target_reasoning_overrides is not None
            or self.agent.reasoning is not None
        ):
            raise ValueError("Sonnet profile must not send reasoning=none")
        if self.assistant_prefill_transport_suffix != ".":
            raise ValueError("Sonnet transport suffix must remain explicit and fixed")
        if (self.agent.temperature, self.agent.max_tokens) != (
            base.agent.temperature,
            base.agent.max_tokens,
        ):
            raise ValueError("Sonnet generation limits must match V4")


def make_protocol(base, adapter_sha256, transport_args):
    data = base.model_dump()
    data.pop("target_reasoning_overrides", None)
    data.update(
        version="duma-kaggle-sonnet55-transport-v2",
        base_protocol=base.model_dump(),
        hosted_adapter_sha256=adapter_sha256,
        agent={**base.agent.model_dump(), "reasoning": None, **transport_args},
    )
    return SonnetProtocol.model_validate(data)


kbench.config.interactive_mode = False


@kbench.task(name="__SONNET_TASK_NAME__")
def sonnet_task(llm) -> dict:
    target = str(getattr(llm, "model", llm.name))
    if target != "anthropic/claude-sonnet-5-5@default":
        return {"skipped": "Sonnet-only readiness; no security score"}
    base = load_protocol(os.environ["DUMA_KAGGLE_PROTOCOL"])
    protocol = make_protocol(base, SONNET_ADAPTER_SHA256, SONNET_TRANSPORT_ARGS)
    output = Path("/kaggle/working/duma-sonnet55-results")
    original_backend = runner.KaggleBackend
    runner.KaggleBackend = SonnetBackend
    try:
        return evaluate_phase(llm, protocol, output)
    finally:
        runner.KaggleBackend = original_backend


def evaluate_phase(llm, protocol, output):
    if SONNET_PHASE in {"preflight", "preflight-dual"}:
        rows = [
            ("collab", "collab_poisoning_logs", "solo"),
            ("collab", "collab_poisoning_logs", "dual"),
            ("output_handling", "output_xss_block", "dual"),
        ]
        if SONNET_PHASE == "preflight-dual":
            rows = [row for row in rows if row[2] == "dual"]
        outcomes = {}
        for domain, task_id, regime in rows:
            outcomes[task_id + "-" + regime] = run_episode(
                llm,
                protocol,
                domain=domain,
                task_id=task_id,
                regime=regime,
                trial=0,
                output_dir=output,
            )
        return {
            "scope": "Compatibility rows, not full evaluation",
            "outcomes": outcomes,
        }
    if SONNET_PHASE not in {"solo", "dual"}:
        raise ValueError("Unknown Sonnet evaluation phase")
    result = run_matrix(llm, protocol, regime=SONNET_PHASE, output_dir=output)
    return {"profile": protocol.version, "score": result["score"]}


if __name__ == "__main__":
    sonnet_task.run(kbench.llm)
