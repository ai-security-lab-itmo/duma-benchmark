"""Metadata repair preserves action schemas, policies and original results."""

import ast
from copy import deepcopy
from pathlib import Path

from duma.integrations.kaggle.protocol import KaggleProtocol, load_protocol


def definitions():
    path = Path(__file__).parents[1] / "benchmarks/kaggle/gptoss_compat_preflight.py"
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef)
        or isinstance(n, ast.FunctionDef)
        and n.name in {"normalize_tools", "make_protocol"}
    ]
    ns = {"deepcopy": deepcopy, "KaggleProtocol": KaggleProtocol}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_metadata_only_normalization():
    original = {
        "messages": [{"role": "system", "content": "policy"}],
        "tools": [
            {"type": "function", "function": {"name": "read", "description": ""}},
            {
                "type": "function",
                "function": {
                    "name": "dangerous",
                    "description": "Original warning",
                    "parameters": {"required": ["id"]},
                },
            },
        ],
        "tool_choice": "required",
    }
    before = deepcopy(original)
    fixed = definitions()["normalize_tools"](original)
    assert original == before
    assert fixed["messages"] == before["messages"]
    assert fixed["tools"][1] == before["tools"][1]
    assert fixed["tools"][0]["function"] == {"name": "read", "description": "read"}
    assert "parameters" not in fixed["tools"][0]["function"]
    assert fixed["tool_choice"] == "required"


def test_profile_is_explicit_and_base_immutable():
    base = load_protocol(
        Path(__file__).parents[1] / "benchmarks/kaggle/protocol.full.v4.json"
    )
    before = base.model_dump()
    profile = definitions()["make_protocol"](base, "a" * 64)
    assert profile.version == "duma-gptoss-low-named-descriptions-v1"
    assert profile.agent.reasoning == "low"
    assert profile.fingerprint != base.fingerprint
    assert profile.tasks == base.tasks
    assert (profile.max_steps, profile.max_errors) == (100, 10)
    assert profile.user == base.user and profile.judge == base.judge
    assert base.model_dump() == before
