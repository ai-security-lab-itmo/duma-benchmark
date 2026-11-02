"""Merging retries cannot resample completed policy failures."""

from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from duma.integrations.kaggle.artifacts import checksum


HELPERS = runpy.run_path(
    str(Path(__file__).parents[1] / "benchmarks/kaggle/audit_resumed_matrix.py")
)


def example():
    originals = [
        {"metadata": {"task_id": "failure"}, "status": "completed", "score": 0.0},
        {"metadata": {"task_id": "retry"}, "status": "technical_error", "score": None},
    ]
    for item in originals:
        item["artifact_hash"] = checksum(item)
    rows = [
        {
            "metadata_hash": HELPERS["metadata_hash"](item),
            "source_artifact_hash": item["artifact_hash"],
        }
        for item in originals
    ]
    manifest = {"preserved_rows": rows[:1], "retry_rows": rows[1:]}
    retry = deepcopy(originals[1])
    retry.update(status="completed", score=1.0)
    retry["artifact_hash"] = checksum(retry)
    return originals, retry, manifest


def test_retry_preserves_completed_failure_and_originals():
    originals, retry, manifest = example()
    before = deepcopy(originals)
    selected, provenance = HELPERS["select_matrix"](originals, [retry], manifest)
    assert originals == before
    assert selected[0] == originals[0]
    assert selected[0]["score"] == 0.0
    assert selected[1] == retry
    assert provenance[0]["original_artifact_hash"] == originals[1]["artifact_hash"]


def test_missing_retry_keeps_matrix_incomplete():
    originals, _, manifest = example()
    selected, provenance = HELPERS["select_matrix"](originals, [], manifest)
    assert selected == originals
    assert selected[1]["score"] is None
    assert not provenance


def test_completed_resample_duplicate_retry_and_changed_identity_rejected():
    originals, retry, manifest = example()
    for attempts in ([originals[0]], [retry, retry]):
        with pytest.raises(ValueError, match="Undeclared or duplicate"):
            HELPERS["select_matrix"](originals, attempts, manifest)
    retry["metadata"]["task_id"] = "different"
    retry["artifact_hash"] = checksum(retry)
    with pytest.raises(ValueError, match="Undeclared"):
        HELPERS["select_matrix"](originals, [retry], manifest)


def test_missing_changed_and_duplicate_originals_rejected():
    originals, _, manifest = example()
    for rows in (originals[:1], originals + originals[:1]):
        with pytest.raises(ValueError):
            HELPERS["select_matrix"](rows, [], manifest)
    originals[0]["score"] = 1.0
    originals[0]["artifact_hash"] = checksum(originals[0])
    with pytest.raises(ValueError, match="unaudited"):
        HELPERS["select_matrix"](originals, [], manifest)
