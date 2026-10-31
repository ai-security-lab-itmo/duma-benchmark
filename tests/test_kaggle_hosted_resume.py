"""Exercise retry-only execution offline, including immediate quota stop."""

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from duma.integrations.kaggle.artifacts import (
    checksum,
    read_checkpoint,
    write_checkpoint,
)
from duma.integrations.kaggle.metrics import evaluation_rows
from duma.integrations.kaggle.protocol import load_protocol


def make_fixture():
    protocol = load_protocol(
        Path(__file__).parents[1] / "benchmarks/kaggle/protocol.full.v4.json"
    )
    static = {
        "protocol": protocol.model_dump(),
        "protocol_hash": protocol.fingerprint,
        "source_hash": "hash",
        "data_hash": "hash",
        "sdk_version": "0.6.2",
        "target_model": "target",
        "service_models": {"user": protocol.user.model, "judge": protocol.judge.model},
    }
    records = []
    for i, row in enumerate(evaluation_rows(protocol, "dual")):
        meta = {**static, **row}
        records.append(
            {
                **row,
                "status": "completed" if i < 90 else "technical_error",
                "score": 0.0 if i < 90 else None,
                "metadata_hash": hashlib.sha256(
                    json.dumps(meta, sort_keys=True).encode()
                ).hexdigest(),
                "source_artifact_hash": str(i),
            }
        )
    manifest = {
        "target_model": "target",
        "static_metadata": static,
        "preserved_rows": records[:90],
        "retry_rows": records[90:],
    }
    manifest["artifact_hash"] = checksum(manifest)
    return protocol, manifest


def namespace(tmp_path):
    source = Path(__file__).parents[1] / "benchmarks/kaggle/hosted_resume.py"
    tree = ast.parse(source.read_text())
    funcs = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name != "resume_task"
    ]
    ns = {
        **globals(),
        "ROW_FIELDS": ("domain", "task_id", "regime", "trial", "seed"),
        "DATA_DIR": tmp_path,
        "fingerprint_files": lambda *_: "hash",
    }
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(source), "exec"), ns)
    return ns


def test_selection_rejects_completed_failure_in_retry(tmp_path):
    _, manifest = make_fixture()
    row = manifest["preserved_rows"].pop()
    manifest["retry_rows"].append(row)
    manifest["artifact_hash"] = checksum(manifest)
    with pytest.raises(ValueError, match="completed row"):
        namespace(tmp_path)["validate_selection"](manifest, manifest["artifact_hash"])


@pytest.mark.parametrize("quota", [False, True])
def test_only_declared_retries_and_early_stop(tmp_path, monkeypatch, quota):
    from duma.integrations.kaggle import runner
    from duma.integrations.kaggle.protocol import KaggleProtocol

    protocol, manifest = make_fixture()
    monkeypatch.setattr(KaggleProtocol, "validate_data", lambda *_: None)
    models = {
        slug: SimpleNamespace(model=slug, name=slug)
        for slug in manifest["static_metadata"]["service_models"].values()
    }
    monkeypatch.setitem(
        sys.modules,
        "kaggle_benchmarks",
        SimpleNamespace(__version__="0.6.2", llms=models),
    )
    calls = []

    class Denied(Exception):
        status_code = 403

    class Backend:
        def __call__(self, **kwargs):
            if quota:
                raise Denied("quota reservation denied")

    def episode(llm, protocol, *, output_dir, **row):
        calls.append(row)
        metadata = {
            **manifest["static_metadata"],
            **row,
            "seed": protocol.trial_seeds[row["trial"]],
        }
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        try:
            runner.KaggleBackend()()
            item = {
                "metadata": metadata,
                "status": "completed",
                "score": 0.0,
                "simulation": {"reward_info": {"reward": 0.0}},
            }
        except Denied:
            item = {"metadata": metadata, "status": "technical_error", "score": None}
        write_checkpoint(output_dir / (key + ".json"), item)

    monkeypatch.setattr(runner, "run_episode", episode)
    original_backend = runner.KaggleBackend
    result = namespace(tmp_path)["execute_selection"](
        SimpleNamespace(model="target", name="target"),
        manifest,
        manifest["artifact_hash"],
        protocol,
        tmp_path,
        Backend,
    )
    assert runner.KaggleBackend is original_backend
    assert len(calls) == (1 if quota else 15)
    assert result["completed_retries"] == (0 if quota else 15)
    assert len(result["not_attempted"]) == (14 if quota else 0)
    assert result["preserved_original_rows"] == 90
    assert "score" not in result
    if not quota:
        namespace(tmp_path)["execute_selection"](
            SimpleNamespace(model="target", name="target"),
            manifest,
            manifest["artifact_hash"],
            protocol,
            tmp_path,
            Backend,
        )
        assert len(calls) == 15


def test_selection_hash_must_match(tmp_path):
    _, manifest = make_fixture()
    with pytest.raises(ValueError, match="checksum"):
        namespace(tmp_path)["validate_selection"](manifest, "wrong")
