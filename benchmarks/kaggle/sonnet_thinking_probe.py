"""Append to frozen bootstrap: bounded, side-effect-free Sonnet transport probe."""

import json
from pathlib import Path

import kaggle_benchmarks as kbench

from duma.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from duma.environment.tool import as_tool
from duma.integrations.kaggle.backend import KaggleBackend

kbench.config.interactive_mode = False


@kbench.task(name="duma-sonnet55-thinking-probe")
def sonnet_thinking_probe(llm) -> dict:
    model = str(getattr(llm, "model", llm.name))
    if model != "anthropic/claude-sonnet-5-5@default":
        return {"skipped": "Sonnet 5.5 transport only; no safety score"}

    def ping() -> str:
        """Return PONG_OK without side effects."""
        return "PONG_OK"

    configs = [
        ("omit_reasoning", {}),
        (
            "explicit_nested_anthropic",
            {
                "extra_body": {
                    "extra_body": {"anthropic": {"thinking": {"type": "between_tools"}}}
                }
            },
        ),
        ("explicit_top_level", {"extra_body": {"thinking": {"type": "between_tools"}}}),
    ]
    report = {
        "scope": "Transport compatibility, not security score",
        "model": model,
        "cases": [],
    }
    path = Path("/kaggle/working/sonnet55-thinking-probe.json")
    for label, params in configs:
        for regime in ("solo", "dual"):
            backend = KaggleBackend(llm, "target")
            history = [
                SystemMessage(
                    role="system",
                    content=(
                        "Diagnostic: call ping once, then return PONG_OK after its result."
                    ),
                )
            ]
            history.append(
                AssistantMessage(role="assistant", content="The diagnostic is ready.")
                if regime == "solo"
                else UserMessage(role="user", content="Run the diagnostic.")
            )
            row = {"configuration": label, "regime": regime, "parameters": params}
            try:
                response = backend(
                    messages=history, tools=[as_tool(ping)], max_tokens=512, **params
                )
                row["first_response"] = response.model_dump(mode="json")
                if not response.tool_calls or any(
                    c.name != "ping" or c.arguments for c in response.tool_calls
                ):
                    row["status"] = "unexpected_response"
                else:
                    second = backend(
                        messages=history
                        + [response]
                        + [
                            ToolMessage(role="tool", id=c.id, content="PONG_OK")
                            for c in response.tool_calls
                        ],
                        tools=[as_tool(ping)],
                        max_tokens=512,
                        **params,
                    )
                    row["second_response"] = second.model_dump(mode="json")
                    row["status"] = (
                        "roundtrip_ok"
                        if not second.tool_calls and "PONG_OK" in (second.content or "")
                        else "unexpected_replay"
                    )
            except Exception as exc:
                row.update(
                    status="error",
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:1500],
                )
            row["usage"] = backend.calls
            report["cases"].append(row)
            path.write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {"configuration": label, "regime": regime, "status": row["status"]}
                ),
                flush=True,
            )
    return {
        "scope": report["scope"],
        "outcomes": [r["status"] for r in report["cases"]],
    }


if __name__ == "__main__":
    sonnet_thinking_probe.run(kbench.llm)
