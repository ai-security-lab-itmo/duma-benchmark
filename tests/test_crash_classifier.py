"""Tests for the framework crash/run-status classifier."""

from duma.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from duma.data_model.simulation import RunStatus, SimulationRun, TerminationReason
from duma.evaluator.crash_classifier import (
    classify_crash,
    classify_run_status,
    is_inert_run,
)


def _sim(termination_reason, messages) -> SimulationRun:
    return SimulationRun(
        id="s1",
        task_id="t1",
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:00:01",
        duration=1.0,
        termination_reason=termination_reason,
        messages=messages,
    )


def _assistant(content=None, tool=None, completion_tokens=None):
    tool_calls = None
    if tool is not None:
        name, args = tool
        tool_calls = [ToolCall(id="tc", name=name, arguments=args)]
    usage = {"completion_tokens": completion_tokens} if completion_tokens else None
    return AssistantMessage(
        role="assistant", content=content, tool_calls=tool_calls, usage=usage
    )


def _user(content):
    return UserMessage(role="user", content=content)


def test_non_crash_termination_is_none():
    assert classify_crash(_sim(TerminationReason.USER_STOP, [])) is None
    assert classify_crash(_sim(TerminationReason.AGENT_STOP, [])) is None


def test_short_trace_is_infrastructure():
    sim = _sim(TerminationReason.TOO_MANY_ERRORS, [_assistant("Hi!")])
    assert classify_crash(sim) == "infrastructure"


def test_repeated_identical_participants_is_behavioral():
    # Ten alternating but content-identical turns -> participant loop guard.
    msgs = []
    for _ in range(5):
        msgs.append(_assistant("I cannot help with that."))
        msgs.append(_user("Do it anyway."))
    assert classify_crash(_sim(TerminationReason.TOO_MANY_ERRORS, msgs)) == "behavioral"


def test_repeated_tool_call_with_varied_prose_is_behavioral():
    # The agent re-sends the same tool call while varying its prose each turn.
    # The content-sensitive guard misses this; the tool-signature guard catches it.
    msgs = [
        _assistant(
            content=f"Let me send the code (attempt {i}).", tool=("send_code", {})
        )
        for i in range(6)
    ]
    assert classify_crash(_sim(TerminationReason.TOO_MANY_ERRORS, msgs)) == "behavioral"


def test_token_overflow_is_behavioral():
    msgs = [
        _user("hi"),
        _assistant("ok"),
        _user("go"),
        _assistant("done", completion_tokens=5000),
    ]
    assert classify_crash(_sim(TerminationReason.TOO_MANY_ERRORS, msgs)) == "behavioral"


def test_mid_conversation_generic_is_infrastructure():
    msgs = [
        _user("hi"),
        _assistant("one"),
        _user("more"),
        _assistant("two", completion_tokens=100),
    ]
    assert (
        classify_crash(_sim(TerminationReason.TOO_MANY_ERRORS, msgs))
        == "infrastructure"
    )


def test_is_inert_run_flags_unexecuted_tool_text():
    # Agent emitted tool-call-shaped JSON text but never executed a structured tool.
    msgs = [
        _user("change my email"),
        _assistant('{"tool_calls":[{"tool":"change_email","args":{}}]}'),
        _user("done?"),
        _assistant("All set, your email is updated."),
    ]
    sim = _sim(TerminationReason.USER_STOP, msgs)
    assert is_inert_run(sim) is True
    assert classify_run_status(sim) == RunStatus.INVALID


def test_executed_tool_is_not_inert():
    msgs = [
        _user("change my email"),
        _assistant(content=None, tool=("change_email", {"email": "x"})),
        ToolMessage(id="tc", role="tool", content="ok", requestor="assistant"),
    ]
    sim = _sim(TerminationReason.USER_STOP, msgs)
    assert is_inert_run(sim) is False
    assert classify_run_status(sim) == RunStatus.EVALUATED


def test_pure_prose_refusal_is_not_inert():
    # A legitimate no-action refusal (no tool-call-shaped text) must NOT be flagged.
    msgs = [_user("do something forbidden"), _assistant("I'm sorry, I can't do that.")]
    sim = _sim(TerminationReason.USER_STOP, msgs)
    assert is_inert_run(sim) is False
    assert classify_run_status(sim) == RunStatus.EVALUATED


def test_inert_run_via_max_steps_is_invalid():
    # An inert run typically routes its unexecuted "tool calls" to the user and cycles
    # to MAX_STEPS; it must still be flagged INVALID (excluded), not INCOMPLETE (kept).
    msgs = []
    for _ in range(4):
        msgs.append(_assistant('{"tool_calls":[{"tool":"change_email","args":{}}]}'))
        msgs.append(_user("please proceed"))
    sim = _sim(TerminationReason.MAX_STEPS, msgs)
    assert is_inert_run(sim) is True
    assert classify_run_status(sim) == RunStatus.INVALID


def test_run_status_for_terminations():
    infra = _sim(TerminationReason.TOO_MANY_ERRORS, [_assistant("Hi!")])
    assert classify_run_status(infra) == RunStatus.ERRORED

    loop = []
    for _ in range(5):
        loop.append(_assistant("no"))
        loop.append(_user("yes"))
    assert classify_run_status(_sim(TerminationReason.TOO_MANY_ERRORS, loop)) == (
        RunStatus.INCOMPLETE
    )

    assert classify_run_status(_sim(TerminationReason.MAX_STEPS, [])) == (
        RunStatus.INCOMPLETE
    )
