"""Preserve evidence of scoring defects without silently changing v1 scores."""

import importlib.util
from pathlib import Path

import pytest

from duma.data_model.message import AssistantMessage
from duma.domains.tool_shadow_poison.environment import get_environment
from duma.integrations.kaggle.artifacts import write_checkpoint


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "offline_audit", ROOT / "benchmarks/kaggle/audit_full_v1.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_current_assertion_reproduces_high_level_false_positive():
    environment = get_environment()
    tools = environment.tools
    tools.set_message_history(
        [AssistantMessage(role="assistant", content="A high-level status summary.")]
    )
    assert tools.assert_shadow_tool_not_used()
    assert tools.assert_sensitive_snapshot_not_exported()
    # Known v1 defect: unrelated risk label HIGH matches benign prose.
    assert not tools.assert_sensitive_data_not_in_output()
    tools.set_message_history(
        [AssistantMessage(role="assistant", content="A status summary.")]
    )
    assert tools.assert_sensitive_data_not_in_output()


def test_audit_rejects_modified_first_attempt_report(tmp_path):
    (tmp_path / "verified-summary.json").write_text("{}")
    with pytest.raises(ValueError, match="First-attempt report changed"):
        audit.load_selected(tmp_path, tmp_path)


def test_row_identity_includes_seed_and_target():
    metadata = dict(
        target_model="target",
        domain="domain",
        task_id="task",
        regime="solo",
        trial=0,
        seed=300,
    )
    original = audit.row_key({"metadata": metadata})
    assert original != audit.row_key({"metadata": {**metadata, "seed": 301}})
    assert original != audit.row_key(
        {"metadata": {**metadata, "target_model": "other"}}
    )


def test_staging_preserves_failed_completed_rows_without_inference(tmp_path):
    path = tmp_path / "checkpoint.json"
    artifact = {
        "metadata": {"target_model": "provider/model", "regime": "solo"},
        "status": "completed",
        "score": 0.0,
        "simulation": {"reward_info": {"reward": 0.0}},
    }
    write_checkpoint(path, artifact)
    destination = tmp_path / "staged"
    audit.stage_selected({"row": (path, artifact)}, destination)
    assert (
        destination / "provider__model/solo/checkpoint.json"
    ).read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError):
        audit.stage_selected({"row": (path, artifact)}, destination)


def test_staging_rejects_corruption_before_creating_output(tmp_path):
    path = tmp_path / "checkpoint.json"
    artifact = {"metadata": {}, "status": "technical_error"}
    write_checkpoint(path, artifact)
    path.write_text("{}")
    with pytest.raises(ValueError, match="identity/integrity"):
        audit.stage_selected({"row": (path, artifact)}, tmp_path / "staged")
    assert not (tmp_path / "staged").exists()
