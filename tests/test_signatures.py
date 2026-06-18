"""Signatures used on fatal paths (loop guard, give-up) must key on FULL content, so
two long payloads sharing a long prefix do not collapse and falsely trigger."""

from duma.data_model.message import AssistantMessage, ToolCall
from duma.utils.signatures import content_key, message_signature, tool_signature


def _assistant(content=None, tool=None):
    tool_calls = None
    if tool is not None:
        name, args = tool
        tool_calls = [ToolCall(id="tc", name=name, arguments=args)]
    return AssistantMessage(role="assistant", content=content, tool_calls=tool_calls)


def test_long_prefix_identical_content_has_distinct_signatures():
    prefix = "x" * 400
    a = _assistant(prefix + " ALPHA")
    b = _assistant(prefix + " BETA")
    assert message_signature(a) != message_signature(b)
    assert content_key(a.content) != content_key(b.content)


def test_long_prefix_identical_tool_args_have_distinct_signatures():
    base = {f"k{i}": i for i in range(40)}  # serializes well past 120 chars
    a = _assistant(tool=("t", {**base, "z": "ALPHA"}))
    b = _assistant(tool=("t", {**base, "z": "BETA"}))
    assert tool_signature(a) != tool_signature(b)
    assert message_signature(a) != message_signature(b)


def test_identical_payloads_collapse():
    a = _assistant("same message", tool=("t", {"x": 1}))
    b = _assistant("same message", tool=("t", {"x": 1}))
    assert message_signature(a) == message_signature(b)
    assert tool_signature(a) == tool_signature(b)
