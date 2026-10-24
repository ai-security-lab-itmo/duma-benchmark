from copy import deepcopy
import hashlib
from pathlib import Path
import runpy

import pytest

from duma.integrations.kaggle.protocol import load_protocol
from duma.runtime import run_task


ROOT = Path(__file__).resolve().parents[1]
verify_episode = runpy.run_path(str(ROOT / "benchmarks/kaggle/verify_results.py"))[
    "verify_episode"
]


def test_complete_audit_recomputes_hosted_summary(examples, tmp_path, monkeypatch):
    from duma.integrations.kaggle.artifacts import write_checkpoint
    from duma.integrations.kaggle.metrics import aggregate

    original, rows = examples
    protocol = original.model_copy(
        update={
            "tasks": {"collab": ["collab_poisoning_logs"]},
            "trial_seeds": [300],
            "regimes": ["dual"],
        }
    )
    row = deepcopy(rows["dual-safe"])
    row["metadata"].update(
        protocol=protocol.model_dump(),
        protocol_hash=protocol.fingerprint,
        target_model="target",
        trial=0,
        seed=300,
    )
    folder = tmp_path / "run/duma-kaggle-results"
    folder.mkdir(parents=True)
    write_checkpoint(folder / "episode.json", row)
    summary = aggregate([row], protocol, "target", "dual")
    reports = folder / "summaries"
    reports.mkdir()
    name = (
        hashlib.sha256(f"target/{protocol.fingerprint}/dual".encode()).hexdigest()
        + ".json"
    )
    write_checkpoint(reports / name, summary)
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"test identity is injected")
    audit = runpy.run_path(str(ROOT / "benchmarks/kaggle/verify_results.py"))["audit"]
    monkeypatch.setitem(audit.__globals__, "wheel_identity", lambda _: "source")
    monkeypatch.setitem(audit.__globals__, "fingerprint_files", lambda *_: "data")
    result = audit(tmp_path, bundle, protocol, ["target"])
    assert result["complete"] and result["expected"] == 1
    assert result["matrices"][0]["score"] == 0.0
    (reports / name).unlink()
    assert not audit(tmp_path, bundle, protocol, ["target"])["complete"]


@pytest.fixture(scope="module")
def examples():
    protocol = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v3.json")
    results = runpy.run_path(str(ROOT / "tests/kaggle_replay.py"))["replays"](run_task)
    output = {}
    for case in ("dual-safe", "communication"):
        simulation = results[case]["simulation"]
        simulation.update(
            start_time="2026-10-24T00:00:00",
            end_time="2026-10-24T00:00:01",
            duration=1.0,
        )
        output[case] = {
            "metadata": {
                "protocol": protocol.model_dump(),
                "protocol_hash": protocol.fingerprint,
                "source_hash": "source",
                "data_hash": "data",
                "sdk_version": "0.6.2",
                "service_models": {
                    "user": protocol.user.model,
                    "judge": protocol.judge.model,
                },
                "domain": "output_handling" if case == "communication" else "collab",
                "task_id": simulation["task_id"],
                "regime": "dual",
            },
            "status": "completed",
            "score": simulation["reward_info"]["reward"],
            "outcome": "success"
            if simulation["reward_info"]["reward"] == 1
            else "policy_failure",
            "simulation": simulation,
            "usage_by_role": {"judge": [{}] if case == "communication" else []},
        }
    return protocol, output


def test_valid_environment_and_communication_rows_replay(examples):
    protocol, rows = examples
    assert verify_episode(rows["dual-safe"], protocol, "source", "data")[
        "failed_env"
    ] == ["assert_security_approval_requested"]
    assert verify_episode(rows["communication"], protocol, "source", "data")[
        "nl_scored"
    ]


def test_incomplete_episode_cannot_have_confirmed_score(examples):
    protocol, rows = examples
    row = deepcopy(rows["dual-safe"])
    row["status"] = "technical_error"
    with pytest.raises(ValueError, match="Unfinished episode"):
        verify_episode(row, protocol, "source", "data")


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda r: r["metadata"].update(source_hash="wrong"), "identity mismatch"),
        (
            lambda r: r["simulation"]["reward_info"]["env_assertions"].clear(),
            "ENV replay",
        ),
        (lambda r: r.update(score=1.0), "Stored score"),
        (lambda r: r["usage_by_role"].update(judge=[{}]), "judge calls"),
    ],
)
def test_verification_rejects_inconsistent_rows(examples, mutation, message):
    protocol, rows = examples
    row = deepcopy(rows["dual-safe"])
    mutation(row)
    with pytest.raises(ValueError, match=message):
        verify_episode(row, protocol, "source", "data")


def test_communication_decisions_cannot_be_changed_to_strings(examples):
    protocol, rows = examples
    row = deepcopy(rows["communication"])
    row["simulation"]["reward_info"]["nl_assertions"][0]["met"] = "true"
    with pytest.raises(ValueError, match="Non-boolean"):
        verify_episode(row, protocol, "source", "data")
