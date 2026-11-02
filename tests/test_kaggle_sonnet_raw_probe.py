"""Diagnostics capture the wire response without repairing or scoring it."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize("empty", [False, True])
def test_raw_probe_is_bounded_restores_client_and_never_scores(tmp_path, empty):
    tree = ast.parse(
        (
            Path(__file__).parents[1] / "benchmarks/kaggle/sonnet_raw_probe.py"
        ).read_text()
    )
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "raw_response_probe"
    )
    fn.decorator_list = []
    calls, reports = [], []

    def create(**kwargs):
        calls.append(kwargs)
        return NS(
            model_dump=lambda **_: {
                "choices": [{"message": None if empty else {"content": "ok"}}]
            }
        )

    completions = NS(create=create)
    llm = NS(
        model="anthropic/claude-sonnet-5-5@default",
        name="sonnet",
        client=NS(chat=NS(completions=completions)),
    )

    class Backend:
        def __init__(self, llm, role):
            self.llm = llm

        def __call__(self, **kwargs):
            self.llm.client.chat.completions.create(**kwargs)
            if empty:
                raise ValueError("Empty response")
            return NS(model_dump=lambda **_: {"content": "ok"})

    snapshot = {
        "artifact_hash": "hash",
        "simulation": {},
        "metadata": {
            "domain": "test",
            "task_id": "task",
            "seed": 301,
            "protocol": {
                "agent": {"model": None, "reasoning": None, "max_tokens": 4096}
            },
        },
    }
    ns = dict(
        json=json,
        Path=lambda _: tmp_path / "probe.json",
        kbench=NS(config=NS()),
        PROBE_SNAPSHOTS=[snapshot, snapshot],
        checksum=lambda _: "hash",
        SimulationRun=NS(
            model_validate=lambda _: NS(assistant_system_prompt="policy", messages=[])
        ),
        SystemMessage=lambda **kw: kw,
        SonnetBackend=Backend,
        registry=NS(
            get_env_constructor=lambda _: lambda: NS(get_tools=lambda: ["tool schema"])
        ),
        write_checkpoint=lambda path, report: reports.append(
            json.loads(json.dumps(report))
        ),
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "probe", "exec"), ns)
    result = ns["raw_response_probe"](llm)
    assert len(calls) == 4
    assert completions.create is create
    assert "score" not in result
    assert all("raw_response" in c for c in reports[-1]["cases"])
    assert result["statuses"] == ["error" if empty else "valid"] * 4
    assert all(c["max_tokens"] == 4096 and c["seed"] == 301 for c in calls)
