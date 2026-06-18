import json
import os
import random
import re
import time
from typing import Any, Callable, Optional

import litellm
from litellm import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    completion,
    completion_cost,
)
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from duma.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    RETRY_BASE_DELAY_SECONDS,
    RETRY_JITTER_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
    USE_LANGFUSE,
)
from duma.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from duma.environment.tool import Tool
from duma.utils.model_ref import (
    infer_provider,
    is_reasoning_model,
    normalize_model_ref,
    to_litellm_model,
)

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_duma_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of message dictionaries to duma message objects.
    """
    duma_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            duma_messages.append(UserMessage(**message))
        elif role == "assistant":
            duma_messages.append(AssistantMessage(**message))
        elif role == "tool":
            duma_messages.append(ToolMessage(**message))
        elif role == "system":
            duma_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return duma_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of duma messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def _short_repr(value: Any, max_len: int = 160) -> str:
    text = repr(value)
    if len(text) > max_len:
        return f"{text[:max_len]}..."
    return text


def _coerce_tool_call_arguments(
    raw_arguments: Any,
    *,
    tool_name: str,
) -> dict[str, Any]:
    """
    Coerce tool-call arguments into a dict.
    LLM providers occasionally return malformed payloads (empty string, function name, etc.).
    We fall back to {} instead of crashing the whole run.
    """
    if isinstance(raw_arguments, dict):
        return raw_arguments

    if raw_arguments is None:
        logger.warning(
            f"Tool call '{tool_name}' returned empty arguments. Falling back to {{}}."
        )
        return {}

    if isinstance(raw_arguments, str):
        raw_str = raw_arguments.strip()
        if raw_str == "" or raw_str.lower() in {"none", "null"}:
            logger.warning(
                f"Tool call '{tool_name}' returned blank arguments string. Falling back to {{}}."
            )
            return {}
        if raw_str == tool_name:
            logger.warning(
                f"Tool call '{tool_name}' returned function name instead of JSON arguments. Falling back to {{}}."
            )
            return {}
        try:
            parsed = json.loads(raw_str)
        except json.JSONDecodeError:
            logger.warning(
                f"Tool call '{tool_name}' has non-JSON arguments={_short_repr(raw_arguments)}. Falling back to {{}}."
            )
            return {}
        if isinstance(parsed, dict):
            return parsed
        logger.warning(
            f"Tool call '{tool_name}' has JSON arguments of type {type(parsed).__name__}, expected object. Falling back to {{}}."
        )
        return {}

    if hasattr(raw_arguments, "model_dump"):
        try:
            dumped = raw_arguments.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass

    logger.warning(
        f"Tool call '{tool_name}' has unsupported arguments type {type(raw_arguments).__name__} ({_short_repr(raw_arguments)}). Falling back to {{}}."
    )
    return {}


def _is_qwen_thinking_model(model: str) -> bool:
    """Check if model is a Qwen thinking-capable model (Qwen3+)."""
    model_l = model.lower()
    return "qwen" in model_l and any(
        v in model_l for v in ("qwen3", "qwen-3", "qwen3.5", "qwen-3.5")
    )


# Key aliases used by providers that serialize tool calls as JSON in the content
# field (notably DeepSeek-v3.2 via proxies, which emit valid JSON objects rather than
# the line-delimited text forms below).
NAME_ALIASES = ("tool", "tool_name", "name", "function", "tool_call_name")
ARG_ALIASES = ("args", "arguments", "parameters", "tool_call_arguments")


def _strip_code_fence(text: str) -> str:
    """Strip a leading ```json / ``` fence and trailing ``` from a string."""
    stripped = text.strip()
    if stripped.startswith("```"):
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[-1]
        else:
            # Single-line fence: drop the ``` and any inline language tag (```json{...}).
            stripped = re.sub(r"^[a-zA-Z0-9_+-]+", "", stripped[3:]).lstrip()
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _extract_name_and_args(obj: dict) -> tuple[str, dict] | None:
    """Pull (name, args) out of a single tool-call dict, honouring key aliases.

    Also handles the nested OpenAI form ``{"function": {"name": ..., "arguments": ...}}``
    by recursing into a dict-valued alias.
    """
    # Nested form: an alias holds the actual {name, arguments} object.
    for k in NAME_ALIASES:
        value = obj.get(k)
        if isinstance(value, dict):
            inner = _extract_name_and_args(value)
            if inner is not None:
                return inner
    name = next(
        (
            obj[k]
            for k in NAME_ALIASES
            if isinstance(obj.get(k), str) and obj[k].strip()
        ),
        None,
    )
    if name is None:
        return None
    raw_args = next((obj[k] for k in ARG_ALIASES if k in obj), {})
    args = _coerce_tool_call_arguments(raw_args, tool_name=name)
    return name.strip(), args


def _json_object_is_tool_shaped(data: Any) -> bool:
    """True if a parsed JSON object is an *executable* tool call.

    Kept a strict subset of what ``_parse_json_object_tool_calls`` actually executes,
    so an empty ``{"tool_calls": []}`` or a malformed ``{"name": 5}`` is NOT treated as
    an (un-executed) tool call — otherwise the inert-run guard would wrongly flag and
    exclude legitimate no-op replies.
    """
    if not isinstance(data, dict):
        return False
    tool_calls = data.get("tool_calls")
    if isinstance(tool_calls, list):
        return any(
            isinstance(item, dict) and _extract_name_and_args(item) is not None
            for item in tool_calls
        )
    return _extract_name_and_args(data) is not None


def _parse_json_object_tool_calls(
    content: str,
    available_tool_names: set[str] | None = None,
) -> list[ToolCall] | None:
    """Parse tool calls emitted as a JSON object in the content field.

    Handles the forms DeepSeek-v3.2 produces, e.g.::

        {"tool_calls": [{"tool": "send_email", "args": {"to": "x"}}]}
        {"tool_calls": [{"tool_name": "t", "arguments": {...}}]}
        {"tool_call_name": "t", "tool_call_arguments": {...}}

    Returns ``None`` for non-tool JSON such as ``{"message": "..."}`` reply wrappers.
    """
    stripped = _strip_code_fence(content)
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not _json_object_is_tool_shaped(data):
        return None

    if isinstance(data.get("tool_calls"), list):
        raw_list = data["tool_calls"]
    else:
        # Single-object form: the whole dict describes one tool call.
        raw_list = [data]

    tool_calls: list[ToolCall] = []
    for idx, item in enumerate(raw_list):
        if not isinstance(item, dict):
            logger.debug(f"Skipping non-dict tool_calls[{idx}]: {_short_repr(item)}")
            continue
        parsed = _extract_name_and_args(item)
        if parsed is None:
            logger.debug(f"Skipping unparseable tool_calls[{idx}]: {_short_repr(item)}")
            continue
        name, args = parsed
        if available_tool_names and name not in available_tool_names:
            logger.warning(
                f"JSON-parsed tool call '{name}' not in available tools "
                f"{available_tool_names}. Keeping anyway."
            )
        tool_calls.append(
            ToolCall(id=f"json_parsed_{name}_{idx}", name=name, arguments=args)
        )
    return tool_calls or None


def _looks_like_tool_call_text(content: Optional[str]) -> bool:
    """Cheap predicate: does this assistant content *look like* an (un-executed) tool call?

    Used by the orchestrator to flag inert runs where the agent emitted tool-call
    text that was never executed. Returns False for ``{"message": ...}`` reply
    wrappers and ordinary prose so legitimate no-action runs are not flagged.
    """
    if not content:
        return False
    stripped = _strip_code_fence(content)
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return False
        return _json_object_is_tool_shaped(data)
    return bool(re.search(r"(?:^|\n)tool_call(?:_name)?\n", stripped))


def _parse_text_tool_calls(
    content: str,
    available_tool_names: set[str] | None = None,
) -> tuple[str | None, list[ToolCall] | None]:
    """Parse tool calls embedded as plain text in the content field.

    Some providers (e.g. DeepSeek V3 via certain proxies) return tool calls as
    text in the ``content`` field instead of the standard ``tool_calls`` array.

    Supported formats::

        tool_call
        function_name
        {"arg": "value"}

        tool_call_name
        function_name
        tool_call_arguments
        {"arg": "value"}

    Returns ``(remaining_content, parsed_tool_calls)`` where
    ``parsed_tool_calls`` is ``None`` when no text tool calls are detected.
    """
    if not content:
        return content, None

    # First try the JSON-object forms (DeepSeek-v3.2 and similar). On success the
    # whole content was the tool call, so there is no remaining text.
    json_tool_calls = _parse_json_object_tool_calls(content, available_tool_names)
    if json_tool_calls:
        logger.info(
            f"Parsed {len(json_tool_calls)} tool call(s) from JSON content: "
            f"{[tc.name for tc in json_tool_calls]}"
        )
        return None, json_tool_calls

    # Pattern 1: "tool_call\nname\n{json}" (possibly at start or after text)
    pattern1 = re.compile(
        r"(?:^|\n)tool_call\n([^\n]+)\n(\{.*?)(?=\ntool_call\n|\Z)",
        re.DOTALL,
    )
    # Pattern 2: "tool_call_name\nname\ntool_call_arguments\n{json}"
    pattern2 = re.compile(
        r"(?:^|\n)tool_call_name\n([^\n]+)\ntool_call_arguments\n(\{.*?)(?=\ntool_call_name\n|\Z)",
        re.DOTALL,
    )

    matches: list[tuple[str, str]] = []  # (name, json_str)
    remaining = content

    for pattern in (pattern2, pattern1):
        found = list(pattern.finditer(remaining))
        if found:
            for m in found:
                name = m.group(1).strip()
                json_str = m.group(2).strip()
                # Strip markdown code fences if present
                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                if json_str.startswith("```"):
                    json_str = json_str[3:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                json_str = json_str.strip()
                matches.append((name, json_str))
            # Remove matched spans from content
            remaining = pattern.sub("", remaining).strip()
            break

    if not matches:
        return content, None

    # Optionally validate names against known tools
    tool_calls: list[ToolCall] = []
    for idx, (name, json_str) in enumerate(matches):
        if available_tool_names and name not in available_tool_names:
            logger.warning(
                f"Text-parsed tool call '{name}' not in available tools "
                f"{available_tool_names}. Keeping anyway."
            )
        try:
            arguments = json.loads(json_str)
            if not isinstance(arguments, dict):
                arguments = {}
        except json.JSONDecodeError:
            logger.warning(
                f"Text-parsed tool call '{name}' has invalid JSON arguments: "
                f"{_short_repr(json_str)}. Falling back to {{}}."
            )
            arguments = {}
        tool_calls.append(
            ToolCall(
                id=f"text_parsed_{name}_{idx}",
                name=name,
                arguments=arguments,
            )
        )

    logger.info(
        f"Parsed {len(tool_calls)} tool call(s) from text content: "
        f"{[tc.name for tc in tool_calls]}"
    )
    final_content = remaining if remaining else None
    return final_content, tool_calls


def _should_retry_with_auto_tool_choice(
    error: Exception,
    tool_choice: Any,
    has_tools: bool,
) -> bool:
    """
    Detect provider/model-specific incompatibility with tool_choice required/object.
    """
    if not has_tools:
        return False
    if not (tool_choice == "required" or isinstance(tool_choice, dict)):
        return False
    err_text = str(error).lower()
    has_tool_choice_marker = "tool_choice" in err_text
    has_unsupported_marker = (
        "not supported" in err_text
        or "does not support" in err_text
        or "unsupported tool use" in err_text
    )
    has_required_or_object = "required" in err_text or "object" in err_text
    return has_tool_choice_marker and has_unsupported_marker and has_required_or_object


_TRANSIENT_EXCEPTIONS = (
    RateLimitError,
    Timeout,
    APIConnectionError,
    ServiceUnavailableError,
    InternalServerError,
)
_TRANSIENT_STATUS_CODES = {408, 409, 429}


def _is_transient_llm_error(exc: Exception) -> bool:
    """Classify whether an LLM error is worth retrying.

    Transient: rate limits, timeouts, connection drops, 5xx, and 408/409/429.
    Permanent (never retried): 400/401/403/404, context-window overflow, etc.

    Classification is done on the ``status_code`` attribute regardless of exception
    class. litellm's typed 5xx exceptions (InternalServerError, BadGatewayError, ...)
    descend from ``openai.APIError``, NOT ``litellm.exceptions.APIError``, so an
    isinstance check against the latter would silently miss real provider 5xx errors.
    """
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return True
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            status = int(status)
        except (TypeError, ValueError):
            return False
        return status in _TRANSIENT_STATUS_CODES or 500 <= status < 600
    return False


def _completion_with_retry(
    call: Callable[[], Any],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Any:
    """Run ``call`` with bounded exponential backoff + jitter on transient errors.

    A single transient API blip must not kill an entire simulation run, so we retry
    here (the only place the typed litellm exceptions exist) before the error
    propagates to the orchestrator and becomes fatal. Permanent errors re-raise
    immediately.
    """
    max_retries = max(0, max_retries)
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 — re-raised below if not retryable
            if attempt >= max_retries or not _is_transient_llm_error(exc):
                raise
            delay = min(
                RETRY_BASE_DELAY_SECONDS * (2**attempt),
                RETRY_MAX_DELAY_SECONDS,
            ) + random.uniform(0, RETRY_JITTER_SECONDS)
            logger.warning(
                f"Transient LLM error (attempt {attempt + 1}/{max_retries + 1}): "
                f"{type(exc).__name__}: {exc}. Retrying in {delay:.1f}s."
            )
            time.sleep(delay)


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    # We retry transient failures ourselves in _completion_with_retry (with backoff
    # and a transient-vs-permanent classifier), so disable litellm's internal retries
    # to avoid compounding (e.g. 3x3 attempts) and to keep permanent errors fast-fail.
    kwargs.setdefault("num_retries", 0)

    normalized_model = normalize_model_ref(model)
    provider = infer_provider(
        model=model,
        api_base=kwargs.get("api_base"),
        explicit_provider=kwargs.get("custom_llm_provider"),
    )
    litellm_model = to_litellm_model(normalized_model, provider)
    kwargs.setdefault("custom_llm_provider", provider)

    if (
        litellm_model.startswith("claude")
        or litellm_model.startswith("anthropic/claude")
    ) and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}

    # Qwen3+ models may have thinking mode enabled by default on some providers,
    # which is incompatible with tool_choice="required".  Proactively disable
    # thinking so that tool calling works without a costly retry.
    if _is_qwen_thinking_model(normalized_model) and tools:
        extra_body = kwargs.get("extra_body") or {}
        extra_body.setdefault("enable_thinking", False)
        kwargs["extra_body"] = extra_body

    # OpenAI reasoning models (gpt-5 family, o-series) reject temperature != 1. The
    # benchmark injects temperature=0 for every agent; strip it for these models.
    # litellm.drop_params already handles this, but stripping explicitly makes the
    # behaviour independent of that global flag.
    if is_reasoning_model(normalized_model):
        kwargs.pop("temperature", None)

    litellm_messages = to_litellm_messages(messages)
    available_tool_names: set[str] | None = None
    if tools:
        available_tool_names = {t.name for t in tools if hasattr(t, "name")}
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    has_tools = tools is not None

    try:
        # Hugging Face router support:
        # - Allow model refs like vendor/model while setting provider independently
        # - Map HF_TOKEN -> HUGGINGFACE_API_KEY env var if not already set
        # - Provide a sensible default api_base if none was passed
        if provider == "huggingface":
            if (
                os.getenv("HUGGINGFACE_API_KEY") is None
                and os.getenv("HF_TOKEN") is not None
            ):
                os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HF_TOKEN")  # for litellm
            if "api_base" not in kwargs or not kwargs.get("api_base"):
                # Default to the HF router OpenAI-compatible endpoint
                kwargs["api_base"] = "https://router.huggingface.co/v1"

        # OpenRouter support (OpenAI-compatible endpoint + model catalog at https://openrouter.ai/models)
        #
        # This repo often runs with OPENROUTER_API_KEY, but some internal models
        # (e.g. output evaluator) might still be configured as "gpt-4o-mini".
        # If we detect an OpenRouter key (sk-or-...), route ALL such requests to
        # OpenRouter and normalize model names.
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        openrouter_mode = (
            (isinstance(openrouter_key, str) and openrouter_key.startswith("sk-or-"))
            or (isinstance(openai_key, str) and openai_key.startswith("sk-or-"))
            or (
                isinstance(kwargs.get("api_key"), str)
                and kwargs.get("api_key").startswith("sk-or-")
            )
        )

        if (
            provider == "openrouter"
            or openrouter_mode
            or (isinstance(model, str) and model.startswith("openrouter/"))
        ):
            # Convenience: allow reusing OPENAI_API_KEY as OPENROUTER_API_KEY.
            if (
                os.getenv("OPENROUTER_API_KEY") is None
                and os.getenv("OPENAI_API_KEY") is not None
            ):
                os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENAI_API_KEY")

            if "api_base" not in kwargs or not kwargs.get("api_base"):
                kwargs["api_base"] = "https://openrouter.ai/api/v1"

            extra_headers = dict(kwargs.get("extra_headers") or {})
            # Optional but recommended by OpenRouter
            extra_headers.setdefault(
                "X-Title", os.getenv("OPENROUTER_APP_NAME", "duma")
            )
            referer = os.getenv("OPENROUTER_HTTP_REFERER")
            if referer:
                extra_headers.setdefault("HTTP-Referer", referer)
            kwargs["extra_headers"] = extra_headers

        response = _completion_with_retry(
            lambda: completion(
                model=litellm_model,
                messages=litellm_messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
        )
    except Exception as e:
        if _should_retry_with_auto_tool_choice(
            error=e,
            tool_choice=tool_choice,
            has_tools=has_tools,
        ):
            logger.warning(
                f"Model/provider rejected tool_choice={tool_choice}. "
                f"Retrying with tool_choice='auto'. Error: {e}"
            )
            response = _completion_with_retry(
                lambda: completion(
                    model=litellm_model,
                    messages=litellm_messages,
                    tools=tools,
                    tool_choice="auto",
                    **kwargs,
                )
            )
        else:
            logger.error(e)
            raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    response_tool_calls = response.message.tool_calls or []
    tool_calls = []
    for tool_call in response_tool_calls:
        function = getattr(tool_call, "function", None)
        tool_name = getattr(function, "name", "") if function is not None else ""
        tool_name = tool_name if isinstance(tool_name, str) else str(tool_name)
        if tool_name == "":
            logger.warning(
                "Received tool call without function name. Using __invalid_tool_name__."
            )
            tool_name = "__invalid_tool_name__"

        tool_call_id = getattr(tool_call, "id", "")
        if not isinstance(tool_call_id, str):
            tool_call_id = str(tool_call_id)

        raw_arguments = (
            getattr(function, "arguments", None) if function is not None else None
        )
        arguments = _coerce_tool_call_arguments(
            raw_arguments,
            tool_name=tool_name,
        )
        tool_calls.append(
            ToolCall(
                id=tool_call_id,
                name=tool_name,
                arguments=arguments,
            )
        )
    tool_calls = tool_calls or None

    # Fallback: parse tool calls embedded as plain text in content (e.g. DeepSeek V3
    # via proxies that don't translate to OpenAI tool_calls format).
    if tool_calls is None and content and has_tools:
        parsed_content, parsed_tool_calls = _parse_text_tool_calls(
            content, available_tool_names
        )
        if parsed_tool_calls:
            content = parsed_content
            tool_calls = parsed_tool_calls

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
