"""Request variants preserve source messages and unchanged tool actions."""

import ast
from copy import deepcopy
from pathlib import Path

import pytest


def transform():
    path = Path(__file__).parents[1] / "benchmarks/kaggle/openweight_wire_probe.py"
    fn = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "transform_request"
    )
    ns = {"deepcopy": deepcopy}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns["transform_request"]


@pytest.fixture
def payload():
    return {
        "model": "model",
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "."},
        ],
        "tools": [
            {"type": "function", "function": {"name": "read"}},
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    },
                },
            },
        ],
        "tool_choice": "required",
        "seed": 300,
    }


@pytest.mark.parametrize(
    "variant",
    ["original", "empty_parameters", "serial_tools", "auto", "bootstrap_instruction"],
)
def test_variants_do_not_mutate_source(payload, variant):
    before = deepcopy(payload)
    result = transform()(payload, variant)
    assert payload == before
    assert result["seed"] == 300
    assert result["messages"][0] == payload["messages"][0]
    assert [t["function"]["name"] for t in result["tools"]] == ["read", "write"]
    assert result["tools"][1] == payload["tools"][1]
    if variant == "empty_parameters":
        assert result["tools"][0]["function"]["parameters"]["properties"] == {}
    if variant == "bootstrap_instruction":
        assert result["tool_choice"] == "required"
        assert "done" in result["messages"][1]["content"]


def test_invalid_variants_and_prefixes_rejected(payload):
    with pytest.raises(ValueError):
        transform()(payload, "unknown")
    payload["messages"][1]["content"] = "actual customer request"
    with pytest.raises(ValueError, match="prefix"):
        transform()(payload, "bootstrap_instruction")


@pytest.mark.parametrize("variant", ["filled_descriptions", "named_empty_parameters"])
def test_only_missing_descriptions_are_named(payload, variant):
    payload["tools"][1]["function"]["description"] = "Existing description"
    before = deepcopy(payload)
    result = transform()(payload, variant)
    assert result["tools"][0]["function"]["description"] == "read"
    assert result["tools"][1] == before["tools"][1]
    assert result["messages"] == before["messages"]
    assert payload == before
