import hashlib
import json
from pathlib import Path
import runpy
import shutil
import zipfile

import pytest

from duma.integrations.kaggle.protocol import load_protocol
from duma.integrations.kaggle.metrics import evaluation_rows
from duma.integrations.kaggle.task_revision import (
    MARKER,
    revise_assets,
    validate_revision,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v3_inventory_and_models_are_explicitly_versioned():
    old = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v2.json")
    full = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v3.json")
    pilot = load_protocol(ROOT / "benchmarks/kaggle/protocol.pilot.v6.json")
    full.validate_data(ROOT / "data")
    pilot.validate_data(ROOT / "data")
    assert full.user == old.user and full.judge == old.judge and full.agent == old.agent
    assert full.max_steps == 100 and full.max_errors == 10
    assert full.trial_seeds == [300, 301, 302]
    assert len(evaluation_rows(full, "solo")) == 105
    assert len(evaluation_rows(full, "dual")) == 105
    assert sum(map(len, pilot.tasks.values())) == 3
    assert pilot.task_data_sha256 == full.task_data_sha256
    assert (
        old.fingerprint
        == "a5284860d56b0514c4ec022eeb96bdf4bc730d1f2b08bdefe8dbfdc1be8e21eb"
    )
    assert full.fingerprint != old.fingerprint


@pytest.mark.parametrize("version", ["full.v1", "full.v2", "pilot.v5"])
def test_historical_protocol_cannot_score_hardened_assets(version):
    with pytest.raises(ValueError, match="v3 protocol"):
        load_protocol(
            ROOT / f"benchmarks/kaggle/protocol.{version}.json"
        ).validate_data(ROOT / "data")


def test_v3_rejects_modified_missing_or_unpinned_data(tmp_path):
    protocol = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v3.json")
    shutil.copytree(ROOT / "data", tmp_path / "data")
    path = tmp_path / "data/duma/domains/collab/policy.md"
    path.write_text(path.read_text() + "\nChanged policy")
    with pytest.raises(ValueError, match="hash mismatch"):
        protocol.validate_data(tmp_path / "data")
    path.unlink()
    with pytest.raises(ValueError, match="hash mismatch"):
        protocol.validate_data(tmp_path / "data")
    with pytest.raises(ValueError, match="requires pinned"):
        protocol.model_copy(update={"task_data_sha256": None}).validate_data(
            ROOT / "data"
        )


def test_v3_bundle_copies_hardened_tasks_without_another_overlay(tmp_path):
    pytest.importorskip("pdm.backend")
    builder = runpy.run_path(str(ROOT / "benchmarks/kaggle/build_bundle.py"))[
        "build_bundle"
    ]
    output = tmp_path / "bundle"
    builder(
        ROOT,
        output,
        ROOT / "benchmarks/kaggle/protocol.full.v3.json",
        "example",
        "dual",
    )
    assert (
        json.loads((output / "dataset-metadata.json").read_text())["id"]
        == "example/duma-bench-private-v3-runtime"
    )
    assert 'name="duma-v3-dual"' in (output / "hosted_pilot.py").read_text()
    with zipfile.ZipFile(output / "duma-pilot.zip") as archive:
        marker = json.loads(archive.read(MARKER))
        assert marker["revision"] == "duma-kaggle-tasks-v3-hardened"
        assert len(marker["revised_sha256"]) >= 24
        for name, expected in marker["revised_sha256"].items():
            assert archive.read(name) == (ROOT / name).read_bytes()
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        archive.extractall(tmp_path / "extracted")
    protocol = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v3.json")
    protocol.validate_data(tmp_path / "extracted/data")
    marker_path = tmp_path / "extracted" / MARKER
    marker_path.write_text("{}")
    with pytest.raises(ValueError, match="Unknown hardened"):
        validate_revision(protocol.version, tmp_path / "extracted/data")


def test_v3_rejects_historical_assets():
    from tests.kaggle_legacy import assets

    with pytest.raises(ValueError, match="original checkout"):
        revise_assets(assets(), "duma-kaggle-private-full-v3")
