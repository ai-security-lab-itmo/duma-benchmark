"""One SDK response per DUMA turn; DUMA alone executes tool calls."""

import base64
import json
from dataclasses import asdict

from duma.data_model.message import AssistantMessage, Message, ToolCall, ToolMessage
from duma.environment.tool import Tool


class KaggleBackend:
    def __init__(self, llm, role: str):
        self.llm = llm
        self.role = role
        self.calls: list[dict] = []

    def __call__(
        self,
        *,
        messages: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: str | None = None,
        temperature: float = 0,
        **kwargs,
    ) -> AssistantMessage:
        import kaggle_benchmarks as kbench
        from kaggle_benchmarks.actors import Actor
        from kaggle_benchmarks.actors.llms import GoogleGenAI
        from kaggle_benchmarks.messages import Message as SDKMessage
        from kaggle_benchmarks.tools import ToolInvocation, ToolInvocationResult

        typed_calls = "thought_signature" in ToolInvocation.__dataclass_fields__
        genai = isinstance(self.llm, GoogleGenAI)

        if kbench.config.interactive_mode:
            raise ValueError(
                "Disable SDK interactive mode for complete tool-call records"
            )
        if temperature and not self.llm.support_temperature:
            raise ValueError(
                f"{self.llm.name} does not support the requested temperature"
            )
        if self.llm.support_temperature:
            kwargs["temperature"] = temperature
        if genai:
            if "max_tokens" in kwargs:
                kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
            if tool_choice is not None:
                modes = {"auto": "AUTO", "required": "ANY", "none": "NONE"}
                if tool_choice not in modes:
                    raise ValueError(f"Unsupported GenAI tool choice: {tool_choice}")
                kwargs["tool_config"] = {
                    "function_calling_config": {"mode": modes[tool_choice]}
                }
        elif tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        # Each turn gets exactly the history supplied by its DUMA participant.
        with kbench.chats.new(f"duma-{self.role}") as chat:
            tool_names = {}
            for message in messages:
                sender = Actor(name=message.role, role=message.role)
                if isinstance(message, ToolMessage):
                    content = ToolInvocationResult(
                        name=tool_names.get(message.id, "tool"),
                        arguments={},
                        call_id=message.id,
                        output=message.content,
                    )
                    sdk_message = SDKMessage(content=content, sender=sender)
                else:
                    sdk_message = SDKMessage(
                        content=message.content or "", sender=sender
                    )
                    if isinstance(message, AssistantMessage) and message.tool_calls:
                        invocations = []
                        carriers = (message.raw_data or {}).get("tool_metadata", {})
                        for call in message.tool_calls:
                            tool_names[call.id] = call.name
                            carrier = carriers.get(call.id, {})
                            signature = carrier.get("thought_signature")
                            if typed_calls:
                                invocation = ToolInvocation(
                                    name=call.name,
                                    arguments=call.arguments,
                                    call_id=call.id,
                                    thought_signature=base64.b64decode(signature)
                                    if signature
                                    else None,
                                    thought=carrier.get("thought"),
                                )
                            else:
                                # PyPI 0.6.1 uses wire dictionaries; newer hosted
                                # builds use typed calls under the same version.
                                invocation = {
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments
                                        if genai
                                        else json.dumps(call.arguments),
                                    },
                                }
                                if genai and signature:
                                    invocation["_thought_signature"] = base64.b64decode(
                                        signature
                                    )
                                if genai and carrier.get("thought") is not None:
                                    invocation["_thought"] = carrier["thought"]
                            invocations.append(invocation)
                        sdk_message._meta["tool_calls"] = invocations
                chat.append(sdk_message)
            response = self.llm.respond(tools=tools or None, **kwargs)

        calls = []
        tool_metadata = {}
        for call in response.tool_calls or []:
            signature = getattr(call, "thought_signature", None)
            thought = getattr(call, "thought", None)
            if isinstance(call, dict):
                signature = call.get("_thought_signature")
                thought = call.get("_thought")
                call = ToolInvocation.from_api_dict(call)
            if not isinstance(call.arguments, dict):
                raise ValueError("Kaggle returned non-object tool arguments")
            calls.append(
                ToolCall(
                    id=call.call_id or "", name=call.name, arguments=call.arguments
                )
            )
            tool_metadata[call.call_id or ""] = {
                "thought_signature": base64.b64encode(signature).decode()
                if signature
                else None,
                "thought": thought,
            }
        usage = asdict(response.usage) if response.usage is not None else {}
        self.calls.append(usage)
        costs = [
            usage.get(key)
            for key in (
                "input_tokens_cost_nanodollars",
                "output_tokens_cost_nanodollars",
            )
        ]
        cost = sum(costs) / 1e9 if all(c is not None for c in costs) else None
        token_usage = None
        if (
            usage.get("input_tokens") is not None
            and usage.get("output_tokens") is not None
        ):
            token_usage = {
                "prompt_tokens": usage["input_tokens"],
                "completion_tokens": usage["output_tokens"],
            }
        result = AssistantMessage(
            role="assistant",
            content=response.content or None,
            tool_calls=calls or None,
            cost=cost,
            usage=token_usage,
            raw_data={
                "backend": "kaggle",
                "model": self.llm.name,
                "tool_metadata": tool_metadata,
            },
        )
        result.validate()
        return result
