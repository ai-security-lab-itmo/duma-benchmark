"""Offline checks; loading the adapter must not trigger Kaggle inference."""

import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from duma.data_model.message import AssistantMessage, SystemMessage, UserMessage
from duma.data_model.simulation import TerminationReason
from duma.integrations.kaggle.artifacts import read_checkpoint, write_checkpoint
from duma.integrations.kaggle.metrics import aggregate, evaluation_rows
from duma.integrations.kaggle.protocol import load_protocol
from duma.utils.generation import use_generation_backends


def adapter_namespace():
    path = Path(__file__).parents[1] / "benchmarks/kaggle/gemma_full.py"
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for function in functions:
        function.decorator_list = []
    funcs = ast.Module(body=functions, type_ignores=[])
    namespace = dict(globals())
    exec(compile(funcs, str(path), "exec"), namespace)
    return namespace


def test_automatic_readiness_has_no_model_calls_or_score():
    ns = adapter_namespace()
    ns["gemma_full"] = Mock(side_effect=AssertionError("Unexpected inference"))
    other = SimpleNamespace(model="google/gemini-3.8-flash", name="gemini")
    for name in ("gemma_solo", "gemma_dual"):
        result = ns[name](other)
        assert "skipped" in result and "score" not in result
    ns["gemma_full"].assert_not_called()


def test_solo_prefix_stable_and_dual_unchanged():
    adapt = adapter_namespace()["adapt_target_messages"]
    system = SystemMessage(role="system", content="Unchanged policy")
    assistant = AssistantMessage(role="assistant", content="Previous turn")
    initial = [system]
    replay = [system, assistant]
    assert adapt(initial, "solo") == [system, UserMessage(role="user", content=".")]
    assert adapt(replay, "solo") == [system, UserMessage(role="user", content="."), assistant]
    assert initial == [system] and replay == [system, assistant]
    dual = [system, UserMessage(role="user", content="Actual user")]
    assert adapt(dual, "dual") == dual
    assert adapt(adapt(replay, "solo"), "solo") == adapt(replay, "solo")


def test_full_matrix_resume_and_incomplete(tmp_path):
    ns = adapter_namespace()
    protocol = load_protocol(Path(__file__).parents[1] / "benchmarks/kaggle/protocol.full.v4.json")
    protocol_view = Mock(wraps=protocol)
    for attr in ("user", "judge", "tasks", "regimes", "trial_seeds", "max_steps", "max_errors", "fingerprint"):
        setattr(protocol_view, attr, getattr(protocol, attr))
    protocol_view.validate_data = Mock()
    model = SimpleNamespace(model="google/gemma-4-31b", name="gemma")
    ns.update(
        kbench=SimpleNamespace(llms={protocol.user.model: model, protocol.judge.model: model}, __version__="0.6.2"),
        GEMMA_ADAPTER_SHA256="offline-adapter", DATA_DIR=tmp_path,
        load_protocol=lambda _: protocol_view, fingerprint_files=lambda *_: "offline-hash",
        KaggleBackend=lambda *args: SimpleNamespace(calls=[], last_response=None),
        get_tasks=lambda *_: [object()],
    )
    reward = {"reward_info": {"reward": 0.0}}
    simulation = SimpleNamespace(
        termination_reason=TerminationReason.AGENT_STOP,
        reward_info=SimpleNamespace(reward=0.0), model_dump=lambda **_: reward,
    )
    run = Mock(return_value=simulation)
    ns["run_task"] = run
    with patch.dict(os.environ, {"DUMA_KAGGLE_PROTOCOL": "test", "DUMA_GEMMA_OUTPUT": str(tmp_path)}):
        assert ns["gemma_full"](model, "solo") == 0.0
        assert run.call_count == 105
        assert "reasoning" not in run.call_args.kwargs["llm_args_agent"]
        assert ns["gemma_full"](model, "solo") == 0.0
        assert run.call_count == 105  # Valid policy failures are not resampled.
        simulation.termination_reason = TerminationReason.MAX_STEPS
        import pytest
        with pytest.raises(RuntimeError, match="Incomplete Gemma"):
            ns["gemma_full"](model, "dual")
        summary = json.loads((tmp_path / "summaries/dual.json").read_text())
        assert summary["score"] is None and summary["completed"] == 0
