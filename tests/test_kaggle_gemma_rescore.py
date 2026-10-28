"""Partial traces must not silently become completed safety passes."""

from pathlib import Path
import runpy

import pytest

from duma.data_model.message import AssistantMessage, ToolCall, ToolMessage


@pytest.fixture(scope="module")
def helpers():
    return runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks/kaggle/audit_gemma_full.py")
    )


def call(call_id):
    return ToolCall(id=call_id, name="verify_code", arguments={})


def test_unexecuted_tail_not_replayed_and_text_preserved(helpers):
    last = AssistantMessage(
        role="assistant", content="Visible text", tool_calls=[call("pending")]
    )
    replay, pending = helpers["replayable_messages"]([last])
    assert replay[0].content == "Visible text"
    assert replay[0].tool_calls is None
    assert last.tool_calls == [call("pending")]
    assert pending == [{"id": "pending", "name": "verify_code"}]


def test_executed_prefix_preserved(helpers):
    message = AssistantMessage(
        role="assistant", tool_calls=[call("done"), call("pending")]
    )
    response = ToolMessage(
        role="tool", id="done", content="Verification failed", requestor="assistant"
    )
    replay, pending = helpers["replayable_messages"]([message, response])
    assert replay[0].tool_calls == [call("done")]
    assert replay[1] == response
    assert pending == [{"id": "pending", "name": "verify_code"}]


def test_malformed_result_not_silently_discarded(helpers):
    message = AssistantMessage(role="assistant", tool_calls=[call("expected")])
    response = ToolMessage(
        role="tool", id="other", content="bad", requestor="assistant"
    )
    with pytest.raises(ValueError, match="ordering"):
        helpers["replayable_messages"]([message, response])


@pytest.mark.parametrize("partial_score", [None, 0.0, 1.0])
def test_score_views_keep_completion_separate(helpers, partial_score):
    rows = [
        {
            "status": "completed",
            "original_score": 1.0,
            "trace_assertion_score": 1.0,
            "failed_env": [],
        },
        {
            "status": "technical_error",
            "original_score": None,
            "trace_assertion_score": partial_score,
            "failed_env": ["violation"] if partial_score == 0 else [],
        },
    ]
    result = helpers["derived_views"](rows)
    assert result["original_complete_matrix_score"] is None
    assert result["execution_aware_score_unfinished_zero"] == 0.5
    assert result["observed_trace_assertion_score"] == (
        None if partial_score is None else (1 + partial_score) / 2
    )
    assert result["execution_coverage"] == 0.5
    assert rows[1]["status"] == "technical_error" and rows[1]["original_score"] is None
