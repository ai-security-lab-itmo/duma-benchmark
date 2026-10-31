"""Validate the Sonnet-only protocol without Kaggle inference or SDK imports."""

import ast
from pathlib import Path
import runpy

import pytest

from duma.integrations.kaggle.backend import KaggleBackend
from duma.integrations.kaggle.artifacts import write_checkpoint
from duma.data_model.message import AssistantMessage, UserMessage, ToolMessage, ToolCall
from duma.integrations.kaggle.protocol import (
    KaggleProtocol,
    ModelSettings,
    load_protocol,
)


def namespace():
    path = Path(__file__).parents[1] / "benchmarks/kaggle/sonnet_default.py"
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        or isinstance(node, ast.FunctionDef)
        and node.name in {"make_protocol", "adapt_messages"}
    ]
    ns = dict(globals())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_transport_suffix_keeps_original_history():
    adapt = namespace()["adapt_messages"]
    initial = [AssistantMessage(role="assistant", content="Original task ticket")]
    result = adapt(initial)
    assert len(initial) == 1 and result[0] is initial[0]
    assert result[-1] == UserMessage(role="user", content=".")
    assert adapt(result) == result
    tools = [ToolMessage(role="tool", id="a", content="Result", requestor="assistant")]
    assert adapt(tools) == tools
    pending = [
        AssistantMessage(
            role="assistant", tool_calls=[ToolCall(id="a", name="ping", arguments={})]
        )
    ]
    with pytest.raises(ValueError, match="pending"):
        adapt(pending)


def test_only_target_transport_is_adapted(monkeypatch):
    backend_class = namespace()["SonnetBackend"]
    captured = []
    monkeypatch.setattr(
        KaggleBackend, "__call__", lambda self, **kw: captured.append(kw["messages"])
    )
    messages = [AssistantMessage(role="assistant", content="Initial ticket")]
    for role in ("target", "user", "judge"):
        backend_class(object(), role)(messages=messages)
    assert captured[0][-1].role == "user"
    assert captured[1] == messages and captured[2] == messages
    assert len(messages) == 1


def test_required_tool_choice_is_not_relaxed(monkeypatch):
    backend_class = namespace()["SonnetBackend"]
    captured = []
    monkeypatch.setattr(
        KaggleBackend, "__call__", lambda self, **kw: captured.append(kw)
    )
    backend_class(object(), "target")(
        messages=[UserMessage(role="user", content="Existing user")],
        tool_choice="required",
    )
    assert captured[0]["tool_choice"] == "required"


def test_retry_selection_preserves_failed_attempt_and_rejects_completed(tmp_path):
    helpers = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks/kaggle/audit_sonnet_preflight.py")
    )
    meta = {
        "domain": "output_handling",
        "task_id": "output_xss_block",
        "regime": "dual",
        "trial": 0,
        "seed": 300,
    }
    old_path, retry_path = tmp_path / "old.json", tmp_path / "retry.json"
    old = {"metadata": meta, "status": "technical_error", "score": None}
    completed = {
        "metadata": meta,
        "status": "completed",
        "score": 0.0,
        "simulation": {"reward_info": {"reward": 0.0}},
    }
    write_checkpoint(old_path, old)
    write_checkpoint(retry_path, completed)
    selected, attempts = helpers["select_attempts"]([old_path], [retry_path])
    assert (
        selected == [retry_path]
        and attempts[0]["artifact_hash"] == old["artifact_hash"]
    )
    write_checkpoint(old_path, completed)
    with pytest.raises(ValueError, match="completed row"):
        helpers["select_attempts"]([old_path], [retry_path])


def test_distinct_profile_and_omitted_reasoning():
    base = load_protocol(
        Path(__file__).parents[1] / "benchmarks/kaggle/protocol.full.v4.json"
    )
    profile = namespace()["make_protocol"](base, "a" * 64, {})
    assert profile.fingerprint != base.fingerprint
    assert profile.agent.generation_args() == {"temperature": 0, "max_tokens": 4096}
    assert profile.base_protocol == base.model_dump()
    assert profile.user == base.user and profile.judge == base.judge
    restored = type(profile).model_validate_json(profile.model_dump_json())
    assert restored.fingerprint == profile.fingerprint
    other = namespace()["make_protocol"](base, "b" * 64, {})
    assert other.fingerprint != profile.fingerprint


def test_transport_configuration_part_of_identity():
    base = load_protocol(
        Path(__file__).parents[1] / "benchmarks/kaggle/protocol.full.v4.json"
    )
    make = namespace()["make_protocol"]
    plain = make(base, "a" * 64, {})
    explicit = make(
        base, "a" * 64, {"extra_body": {"thinking": {"type": "between_tools"}}}
    )
    assert (
        explicit.agent.generation_args()["extra_body"]["thinking"]["type"]
        == "between_tools"
    )
    assert plain.fingerprint != explicit.fingerprint
    assert explicit.model_dump()["agent"]["extra_body"] is not None


def test_base_data_validation_and_limit_preservation(monkeypatch):
    base = load_protocol(
        Path(__file__).parents[1] / "benchmarks/kaggle/protocol.full.v4.json"
    )
    profile = namespace()["make_protocol"](base, "a" * 64, {})
    validated = []
    monkeypatch.setattr(
        KaggleProtocol, "validate_data", lambda self, path: validated.append(path)
    )
    profile.validate_data(Path("data"))
    assert validated == [Path("data")]
    changed = profile.model_copy(update={"max_steps": 20})
    with pytest.raises(ValueError, match="preserve"):
        changed.validate_data(Path("data"))
