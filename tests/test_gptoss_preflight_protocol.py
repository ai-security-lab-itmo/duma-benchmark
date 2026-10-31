"""Compatibility overrides never mutate the frozen V4 profile."""

import ast
from pathlib import Path

import pytest

from duma.integrations.kaggle.protocol import KaggleProtocol, load_protocol


def make_protocol():
    path = Path(__file__).parents[1] / "benchmarks/kaggle/gptoss_minimal_preflight.py"
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "diagnostic_protocol"
    )
    ns = {"KaggleProtocol": KaggleProtocol}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["diagnostic_protocol"]


def test_only_target_reasoning_override_changes():
    base = load_protocol(
        Path(__file__).parents[1] / "benchmarks/kaggle/protocol.full.v4.json"
    )
    before = base.model_dump()
    result = make_protocol()(base)
    assert base.model_dump() == before
    assert result.fingerprint != base.fingerprint
    assert (
        result.target_generation_args("openai/gpt-oss-120b")["reasoning"] == "minimal"
    )
    changed = result.model_dump()
    changed["target_reasoning_overrides"].pop("openai/gpt-oss-120b")
    assert changed == before


def test_changed_base_rejected():
    base = load_protocol(
        Path(__file__).parents[1] / "benchmarks/kaggle/protocol.full.v4.json"
    )
    with pytest.raises(ValueError, match="frozen V4"):
        make_protocol()(base.model_copy(update={"max_steps": 10}))


def test_low_has_explicit_separate_identity():
    base = load_protocol(
        Path(__file__).parents[1] / "benchmarks/kaggle/protocol.full.v4.json"
    )
    minimal = make_protocol()(base)
    low = make_protocol()(base, "low")
    assert low.target_generation_args("openai/gpt-oss-120b")["reasoning"] == "low"
    assert len({base.fingerprint, minimal.fingerprint, low.fingerprint}) == 3
    with pytest.raises(ValueError, match="Unsupported"):
        make_protocol()(base, "high")
