import json
from pathlib import Path

import pytest

from duma.integrations.kaggle.artifacts import (
    checksum,
    episode_lock,
    read_checkpoint,
    write_checkpoint,
)
from duma.integrations.kaggle.metrics import aggregate, evaluation_rows
from duma.integrations.kaggle.protocol import load_protocol


def protocol():
    value = load_protocol(
        Path(__file__).parents[1] / "benchmarks/kaggle/protocol.pilot.v1.json"
    )
    return value.model_copy(
        update={
            "tasks": {"small": ["a"], "large": ["b", "c"]},
            "trial_seeds": [300, 301],
        }
    )


def artifacts(value):
    result = []
    for row in evaluation_rows(value, "dual"):
        score = float(
            row["task_id"] == "a" or (row["task_id"] == "b" and row["trial"] == 0)
        )
        body = {
            "metadata": {
                **row,
                "protocol_hash": value.fingerprint,
                "target_model": "target",
                "source_hash": "source",
                "data_hash": "data",
                "sdk_version": "0.6.1",
            },
            "score": score,
            "status": "completed",
            "simulation": {"reward_info": {"reward": score}},
            "usage_by_role": {
                "target": [
                    {
                        "input_tokens_cost_nanodollars": None,
                        "output_tokens_cost_nanodollars": None,
                    }
                ]
            },
        }
        body["artifact_hash"] = checksum(body)
        result.append(body)
    return result


def test_task_weighting_not_equal_domain_weighting_and_pass_power_k():
    value = protocol()
    summary = aggregate(artifacts(value), value, "target", "dual")
    assert summary["complete"]
    assert summary["score"] == 0.5  # Not (1 + .25) / 2.
    assert summary["by_domain"] == {"small": 1, "large": 0.25}
    assert summary["pass_power_k"][1] == 0.5
    assert summary["pass_power_k"][2] == pytest.approx(1 / 3)
    assert summary["cost_usd_by_role"]["target"] is None


def test_missing_and_technical_runs_cannot_produce_leaderboard_score():
    value = protocol()
    data = artifacts(value)
    summary = aggregate(data[:-1], value, "target", "dual")
    assert summary["score"] is None and not summary["complete"]
    data[-1].update(status="technical_error", score=None)
    data[-1]["artifact_hash"] = checksum(data[-1])
    summary = aggregate(data, value, "target", "dual")
    assert summary["score"] is None
    assert summary["coverage"] == pytest.approx(5 / 6)


def test_mixed_and_duplicate_results_are_rejected():
    value = protocol()
    data = artifacts(value)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(data + data[:1], value, "target", "dual")
    data[-1]["metadata"]["source_hash"] = "other"
    data[-1]["artifact_hash"] = checksum(data[-1])
    with pytest.raises(ValueError, match="Mixed code"):
        aggregate(data, value, "target", "dual")


def test_atomic_checkpoint_and_exclusive_lock(tmp_path):
    body = artifacts(protocol())[0]
    path = tmp_path / "episode.json"
    with episode_lock(path):
        with pytest.raises(FileExistsError):
            with episode_lock(path):
                pass
        write_checkpoint(path, body)
    assert read_checkpoint(path, body["metadata"]) == body
    assert sorted(p.name for p in tmp_path.iterdir()) == ["episode.json"]
