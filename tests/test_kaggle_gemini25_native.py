"""Native transport must not change task, generation or scoring parameters."""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

from duma.integrations.kaggle.artifacts import checksum
from duma.integrations.kaggle.protocol import KaggleProtocol, load_protocol


ROOT = Path(__file__).parents[1]


def namespace():
    path = ROOT / "benchmarks/kaggle/gemini25_native.py"
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef))
    ]
    for node in nodes:
        node.decorator_list = []
    ns = {
        "KaggleProtocol": KaggleProtocol,
        "Literal": Literal,
        "kbench": SimpleNamespace(),
        "Path": Path,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_native_profile_preserves_base_and_identifies_transport():
    base = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v4.json")
    sha = hashlib.sha256(
        (ROOT / "benchmarks/kaggle/gemini25_native.py").read_bytes()
    ).hexdigest()
    profile = namespace()["native_protocol"](base, sha)
    dumped = profile.model_dump()
    for key, value in base.model_dump().items():
        assert dumped[key] == value
    assert profile.fingerprint != base.fingerprint
    assert profile.base_protocol_hash == base.fingerprint
    assert profile.native_adapter_sha256 == sha
    assert profile.target_api == "genai"
    assert profile.target_generation_args(profile.target_model)["reasoning"] == "none"


def test_profile_rejects_other_transport_and_target():
    import pytest
    from pydantic import ValidationError

    base = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v4.json")
    profile = namespace()["native_protocol"](base, "test").model_dump()
    for key, value in (("target_api", "openai"), ("target_model", "other")):
        with pytest.raises(ValidationError):
            namespace()["Gemini25NativeProtocol"](**{**profile, key: value})


def test_readiness_is_not_inference(monkeypatch):
    import sys

    module = SimpleNamespace(ModelProxy=Mock(side_effect=AssertionError("Inference")))
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks.kaggle.model_proxy", module)
    result = namespace()["gemini25_native"](SimpleNamespace(name="readiness"))
    assert "skipped" in result and "score" not in result
    module.ModelProxy.assert_not_called()


def test_wire_audit_rejects_generation_or_tool_changes():
    import copy
    import pytest

    path = ROOT / "benchmarks/kaggle/audit_gemini25_native.py"
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "verify_wire"
    )
    ns = {"checksum": checksum}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    item = {
        "profile_hash": "test",
        "model": "google/gemini-2.5-flash",
        "config": {
            "temperature": None,
            "max_output_tokens": 4096,
            "thinking_config": {"thinking_budget": 0},
            "tools": [{"function_declarations": [{"name": "read"}]}],
            "tool_config": {"function_calling_config": {"mode": "ANY"}},
        },
    }
    item["artifact_hash"] = checksum(item)
    ns["verify_wire"](item, "test")
    dual = copy.deepcopy(item)
    dual["config"]["tool_config"] = None
    dual["artifact_hash"] = checksum(dual)
    ns["verify_wire"](dual, "test")
    for field, value in (
        ("temperature", 1),
        ("thinking_config", {"thinking_budget": -1}),
        ("tools", []),
        ("tool_config", {"function_calling_config": {"mode": "NONE"}}),
    ):
        changed = copy.deepcopy(item)
        changed["config"][field] = value
        changed["artifact_hash"] = checksum(changed)
        with pytest.raises(ValueError):
            ns["verify_wire"](changed, "test")
    with pytest.raises(ValueError):
        ns["verify_wire"](item, "other-profile")
