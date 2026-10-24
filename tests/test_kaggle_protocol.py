import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from duma.data_model.message import AssistantMessage
from duma.integrations.kaggle.protocol import fingerprint_files, load_protocol
from duma.utils.generation import use_generation_backends
from duma.utils.generation import generate
from tests.kaggle_legacy import materialize

ROOT = Path(__file__).parents[1]


def test_draft_manifest_covers_all_tasks_but_cannot_run_unpinned(tmp_path):
    protocol = load_protocol(ROOT / "benchmarks/kaggle/protocol.v1.draft.json")
    protocol.validate_data(materialize(tmp_path))
    assert len(protocol.tasks) == 8
    assert sum(map(len, protocol.tasks.values())) == 35
    with pytest.raises(ValueError, match="Pin user.model"):
        protocol.require_service_models()


def test_protocol_rejects_duplicate_trials_and_credential_fields():
    protocol = load_protocol(ROOT / "benchmarks/kaggle/protocol.v1.draft.json")
    data = protocol.model_dump()
    data["trial_seeds"] = [1, 1]
    with pytest.raises(ValidationError, match="unique"):
        type(protocol).model_validate(data)
    data = protocol.model_dump()
    data["user"]["api_key"] = "should-never-be-a-protocol-field"
    with pytest.raises(ValidationError, match="Extra inputs"):
        type(protocol).model_validate(data)


def test_data_validation_and_content_fingerprints(tmp_path):
    protocol = load_protocol(ROOT / "benchmarks/kaggle/protocol.v1.draft.json")
    data = protocol.model_dump()
    data["tasks"] = {"collab": ["not-a-task"]}
    with pytest.raises(ValueError, match="Unknown tasks"):
        type(protocol).model_validate(data).validate_data(materialize(tmp_path))
    asset = tmp_path / "asset.json"
    asset.write_text(json.dumps({"state": 1}))
    first = fingerprint_files(tmp_path, ("*.json",))
    asset.write_text(json.dumps({"state": 2}))
    assert fingerprint_files(tmp_path, ("*.json",)) != first


def test_generation_bindings_are_scoped_and_unbound_aliases_fail(monkeypatch):
    response = AssistantMessage(role="assistant", content="offline")
    with pytest.raises(ValueError, match="No Kaggle"):
        generate("duma-kaggle-target", [])
    with use_generation_backends({"duma-kaggle-target": lambda **_: response}):
        assert generate("duma-kaggle-target", []) == response
        with pytest.raises(RuntimeError):
            with use_generation_backends({"duma-kaggle-target": lambda **_: response}):
                raise RuntimeError("exit inner scope")
        assert generate("duma-kaggle-target", []) == response
    with pytest.raises(ValueError, match="No Kaggle"):
        generate("duma-kaggle-target", [])
