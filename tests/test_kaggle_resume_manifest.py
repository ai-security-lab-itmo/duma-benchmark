"""Retry selection must retain original audited successes and failures alike."""

import json
from pathlib import Path
import runpy

import pytest

from duma.integrations.kaggle.artifacts import checksum, write_checkpoint


@pytest.fixture
def source(tmp_path):
    paths, hashes = [], []
    protocol = {
        "tasks": {"domain": [f"task{i}" for i in range(35)]},
        "trial_seeds": [300, 301, 302],
    }
    for task in range(35):
        for trial in range(3):
            completed = task < 30
            metadata = {
                "protocol": protocol,
                "target_model": "target",
                "regime": "dual",
                "domain": "domain",
                "task_id": f"task{task}",
                "trial": trial,
                "seed": 300 + trial,
            }
            item = {
                "metadata": metadata,
                "status": "completed" if completed else "technical_error",
                "score": 0.0 if completed else None,
            }
            if completed:
                item["simulation"] = {"reward_info": {"reward": 0.0}}
            path = tmp_path / f"{task}-{trial}.json"
            write_checkpoint(path, item)
            paths.append(path)
            hashes.append({"artifact_hash": item["artifact_hash"]})
    audit = {"checkpoints": hashes}
    audit["artifact_hash"] = checksum(audit)
    build = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks/kaggle/build_resume_manifest.py")
    )["build_manifest"]
    return paths, audit, build


def test_completed_failures_are_never_selected(source):
    paths, audit, build = source
    manifest = build(paths, audit, audit["artifact_hash"], "target", 15)
    assert len(manifest["retry_rows"]) == 15
    assert len(manifest["preserved_rows"]) == 90
    assert all(row["score"] is None for row in manifest["retry_rows"])
    assert all(row["score"] == 0.0 for row in manifest["preserved_rows"])


def test_missing_or_duplicate_row_rejected(source):
    paths, audit, build = source
    with pytest.raises(ValueError, match="105"):
        build(paths[:-1], audit, audit["artifact_hash"], "target", 15)
    with pytest.raises(ValueError, match="Duplicate"):
        build(paths + paths[:1], audit, audit["artifact_hash"], "target", 15)


def test_changed_artifact_rejected_even_with_new_local_checksum(source):
    paths, audit, build = source
    item = json.loads(paths[0].read_text())
    item["metadata"]["seed"] = 999
    write_checkpoint(paths[0], item)
    with pytest.raises(ValueError, match="audited"):
        build(paths, audit, audit["artifact_hash"], "target", 15)


def test_declared_count_and_audit_hash_are_enforced(source):
    paths, audit, build = source
    with pytest.raises(ValueError, match="count"):
        build(paths, audit, audit["artifact_hash"], "target", 14)
    with pytest.raises(ValueError, match="checksum"):
        build(paths, audit, "wrong", "target", 15)
