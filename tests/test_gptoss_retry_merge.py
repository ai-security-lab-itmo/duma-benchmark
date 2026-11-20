"""A technical retry never replaces completed successes or failures."""

from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from duma.integrations.kaggle.artifacts import checksum


SELECT = runpy.run_path(
    str(Path(__file__).parents[1] / "benchmarks/kaggle/audit_gptoss_full.py")
)["select_retry"]


def item(task, status, score):
    result = {"metadata": {"task": task}, "status": status, "score": score}
    result["artifact_hash"] = checksum(result)
    return result


def test_retry_keeps_completed_failure_and_original_bytes():
    old = [item("failure", "completed", 0), item("retry", "technical_error", None)]
    before = deepcopy(old)
    new = item("retry", "completed", 0)
    selected, index = SELECT(old, new, old[1]["artifact_hash"])
    assert old == before
    assert selected[0] == before[0]
    assert selected[1]["score"] == 0
    assert index == 1


def test_undeclared_completed_changed_and_corrupt_retries_rejected():
    old = [item("failure", "completed", 0), item("retry", "technical_error", None)]
    for new, source in [
        (item("failure", "completed", 1), old[0]["artifact_hash"]),
        (item("different", "completed", 1), old[1]["artifact_hash"]),
        (item("retry", "completed", 1), "wrong"),
    ]:
        with pytest.raises(ValueError):
            SELECT(old, new, source)
    new = item("retry", "completed", 1)
    new["score"] = 0
    with pytest.raises(ValueError, match="checksum"):
        SELECT(old, new, old[1]["artifact_hash"])
