"""The retry never invokes a full matrix or resamples completed failures."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from duma.integrations.kaggle.artifacts import read_checkpoint, write_checkpoint


def namespace():
    path = Path(__file__).parents[1] / "benchmarks/kaggle/gemini25_xss_retry.py"
    nodes = [
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
    ]
    for fn in nodes:
        fn.decorator_list = []
    ns = {
        "Path": Path,
        "read_checkpoint": read_checkpoint,
        "write_checkpoint": write_checkpoint,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_only_declared_row_runs_once(tmp_path):
    ns = namespace()
    meta = {
        "protocol": {"test": True},
        "protocol_hash": "profile",
        "sdk_version": "0.6.2",
        "target_model": "google/gemini-2.5-flash",
        "source_hash": "source",
        "data_hash": "data",
        "domain": "output_handling",
        "task_id": "output_xss_block",
        "regime": "dual",
        "trial": 1,
        "seed": 301,
    }
    profile = SimpleNamespace(
        model_dump=lambda: meta["protocol"], fingerprint="profile"
    )
    ns.update(
        GEMINI25_RETRY_METADATA=meta,
        DATA_DIR=tmp_path,
        kbench=SimpleNamespace(__version__="0.6.2"),
        fingerprint_files=lambda path, patterns: (
            "data" if path == tmp_path / "duma" else "source"
        ),
    )

    def run(*args, **kwargs):
        assert kwargs == {
            "domain": "output_handling",
            "task_id": "output_xss_block",
            "regime": "dual",
            "trial": 1,
            "output_dir": tmp_path,
            "resume": False,
        }
        write_checkpoint(
            tmp_path / "row.json",
            {
                "metadata": meta,
                "status": "completed",
                "score": 0.0,
                "simulation": {"reward_info": {"reward": 0.0}},
            },
        )

    ns["run_episode"] = Mock(side_effect=run)
    llm = SimpleNamespace(name="google/gemini-2.5-flash")
    assert ns["selected_matrix"](llm, profile, regime="dual", output_dir=tmp_path) == {
        "score": 0.0
    }
    ns["run_episode"].assert_called_once()
    meta["seed"] = 302
    with pytest.raises(ValueError, match="Only the declared"):
        ns["selected_matrix"](llm, profile, regime="dual", output_dir=tmp_path)
    ns["run_episode"].assert_called_once()


def test_readiness_does_not_run_models():
    ns = namespace()
    ns["gemini25_native"] = SimpleNamespace(
        func=Mock(side_effect=AssertionError("Inference"))
    )
    assert "skipped" in ns["gemini25_xss_retry"](SimpleNamespace(name="readiness"))
    ns["gemini25_native"].func.assert_not_called()


def test_restores_original_dispatch_after_failure(tmp_path):
    ns = namespace()
    original = Mock()

    def fake_path(path):
        return (
            tmp_path
            if path.endswith("duma-kaggle-results")
            else tmp_path / "selection.json"
        )

    ns.update(
        run_matrix=original,
        Path=fake_path,
        GEMINI25_RETRY_METADATA={},
        GEMINI25_RETRY_SOURCE_HASH="old",
        GEMINI25_RETRY_AUDIT_HASH="audit",
        gemini25_native=SimpleNamespace(
            func=Mock(side_effect=RuntimeError("API error"))
        ),
    )
    result = ns["gemini25_xss_retry"](SimpleNamespace(name="google/gemini-2.5-flash"))
    assert ns["run_matrix"] is original
    assert result["error_type"] == "RuntimeError"
    assert read_checkpoint(tmp_path / "selection.json", {})["scope"].startswith(
        "One unfinished"
    )
