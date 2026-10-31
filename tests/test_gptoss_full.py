"""Full dispatch keeps completed failures and stops on quota denial."""

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from duma.integrations.kaggle.artifacts import read_checkpoint, write_checkpoint


@pytest.mark.parametrize("quota", [False, True])
def test_complete_matrix_resume_and_quota_stop(tmp_path, quota):
    source = Path(__file__).parents[1] / "benchmarks/kaggle/gptoss_full.py"
    fn = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "execute_matrix"
    )
    rows = [
        dict(
            domain="d",
            task_id=f"t{i // 3}",
            regime="dual",
            trial=i % 3,
            seed=300 + i % 3,
        )
        for i in range(105)
    ]
    calls = []

    class QuotaError(Exception):
        status_code = 403

    def create(**kwargs):
        calls.append(kwargs)
        if quota:
            raise QuotaError("quota")
        return NS(choices=[NS(finish_reason="stop", message=NS())])

    llm = NS(
        model="openai/gpt-oss-120b", client=NS(chat=NS(completions=NS(create=create)))
    )

    class Backend:
        def __init__(self, llm, role):
            self.llm, self.role = llm, role

        def __call__(self, **kwargs):
            return self.llm.client.chat.completions.create(model=self.llm.model)

    runner = NS(KaggleBackend=Backend)

    def episode(llm, protocol, domain, task_id, regime, trial, output_dir):
        path = output_dir / f"{task_id}-{trial}.json"
        if path.exists():
            return
        item = {
            "metadata": {"domain": domain, "task_id": task_id, "trial": trial},
            "status": "technical_error",
            "score": None,
        }
        try:
            runner.KaggleBackend(llm, "target")()
            item.update(status="completed", score=0.0 if task_id == "t0" else 1.0)
            item["simulation"] = {"reward_info": {"reward": item["score"]}}
        finally:
            write_checkpoint(path, item)

    runner.run_episode = episode

    def aggregate(items, *args):
        complete = len(items) == 105 and all(x["status"] == "completed" for x in items)
        return {
            "complete": complete,
            "observed": len(items),
            "score": sum(x["score"] for x in items) / 105 if complete else None,
        }

    protocol = NS(
        validate_data=lambda _: None,
        trial_seeds=[300, 301, 302],
        fingerprint="profile",
        model_dump=lambda: {},
    )
    ns = dict(
        runner=runner,
        KaggleBackend=Backend,
        evaluation_rows=lambda *_: rows,
        DATA_DIR=tmp_path,
        hashlib=hashlib,
        json=json,
        read_checkpoint=read_checkpoint,
        write_checkpoint=write_checkpoint,
        normalize_tools=lambda x: x,
        aggregate=aggregate,
    )
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), ns)
    report = ns["execute_matrix"](llm, protocol, tmp_path / "results")
    assert runner.KaggleBackend is Backend
    assert llm.client.chat.completions.create is create
    if quota:
        assert len(calls) == 1
        assert report["score"] is None
        assert report["early_stop"]["http_status"] == 403
        with pytest.raises(RuntimeError, match="explicit retry"):
            ns["execute_matrix"](llm, protocol, tmp_path / "results")
        assert len(calls) == 1
    else:
        assert len(calls) == 105 and report["complete"]
        wire = tmp_path / "results/summaries/d-t0-300-wire.json"
        before = wire.read_bytes()
        ns["execute_matrix"](llm, protocol, tmp_path / "results")
        assert len(calls) == 105
        assert wire.read_bytes() == before
        saved = json.loads((tmp_path / "results/t0-0.json").read_text())
        assert saved["score"] == 0.0
