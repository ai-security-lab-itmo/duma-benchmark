import json
from pathlib import Path
import runpy
from zipfile import ZipFile

import pytest
from pydantic import ValidationError

from duma.integrations.kaggle.protocol import KaggleProtocol, load_protocol
from duma.integrations.kaggle.task_revision import MARKER, revise_assets


ROOT = Path(__file__).resolve().parents[1]


def test_v4_only_changes_version_and_explicit_target_overrides():
    baseline = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v3.json")
    current = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v4.json")
    current.validate_data(ROOT / "data")
    current.require_service_models()
    expected = baseline.model_dump()
    expected["version"] = "duma-kaggle-private-full-v4"
    expected["target_reasoning_overrides"] = {
        "qwen/qwen3-235b-a22b-instruct-2507": "minimal", "zai/glm-5": "minimal"
    }
    assert current.model_dump() == expected
    assert baseline.fingerprint == "8315b8cdcf882fa83f20d2224e78d573d8ea2f924aeead5351bad76704878d03"
    assert "target_reasoning_overrides" not in baseline.model_dump()
    assert current.fingerprint != baseline.fingerprint
    assert current.target_generation_args("zai/glm-5")["reasoning"] == "minimal"
    assert current.target_generation_args("qwen/qwen3-235b-a22b-instruct-2507")["reasoning"] == "minimal"
    assert current.target_generation_args("anthropic/claude-haiku-4-5@20251001")["reasoning"] == "none"
    assert current.target_generation_args("openai/gpt-5.4-mini-2026-03-17")["reasoning"] == "none"
    assert current.user == baseline.user and current.judge == baseline.judge
    assert current.agent == baseline.agent


@pytest.mark.parametrize("change", [
    {"target_reasoning_overrides": None},
    {"target_reasoning_overrides": {"glm-5": "minimal"}},
    {"agent": {"reasoning": "minimal"}},
])
def test_v4_rejects_missing_or_ambiguous_target_profile(change):
    data = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v4.json").model_dump()
    data.update(change)
    with pytest.raises(ValidationError):
        KaggleProtocol.model_validate(data)


def test_historical_protocol_does_not_accept_new_overrides():
    data = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v3.json").model_dump()
    data["target_reasoning_overrides"] = {"zai/glm-5": "minimal"}
    with pytest.raises(ValidationError, match="require the V4"):
        KaggleProtocol.model_validate(data)


def test_v4_preserves_exact_hardened_data_and_revision_marker():
    contents = {
        path.relative_to(ROOT).as_posix(): path.read_bytes()
        for path in (ROOT / "data/duma").rglob("*")
        if path.is_file() and path.suffix in {".md", ".json"}
    }
    markers = []
    for version in ("v3", "v4"):
        revised = contents.copy()
        revise_assets(revised, "duma-kaggle-private-full-" + version)
        markers.append(revised.pop(MARKER))
        assert revised == contents
    assert len(set(markers)) == 1
    protocol = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v4.json")
    with pytest.raises(ValueError, match="requires pinned"):
        protocol.model_copy(update={"task_data_sha256": None}).validate_data(ROOT / "data")


def test_v4_bundle_has_distinct_task_dataset_and_runtime_version(tmp_path):
    pytest.importorskip("pdm.backend")
    build = runpy.run_path(str(ROOT / "benchmarks/kaggle/build_bundle.py"))["build_bundle"]
    output = tmp_path / "bundle"
    build(ROOT, output, ROOT / "benchmarks/kaggle/protocol.full.v4.json", "owner", "dual")
    metadata = json.loads((output / "dataset-metadata.json").read_text())
    assert metadata["isPrivate"] is True
    assert metadata["id"] == "owner/duma-bench-private-v4-runtime"
    assert "model-specific reasoning" in metadata["description"]
    assert 'name="duma-v4-dual"' in (output / "hosted_pilot.py").read_text()
    with ZipFile(output / "duma-pilot.zip") as archive:
        assert "duma_kaggle_runtime-0.1.3-py3-none-any.whl" in archive.namelist()
        assert json.loads(archive.read("protocol.json"))["target_reasoning_overrides"]["zai/glm-5"] == "minimal"
