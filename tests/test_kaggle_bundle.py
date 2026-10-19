# Bundle tests deliberately gate optional build tooling before imports.
# ruff: noqa: E402
import ast
import io
import hashlib
import json
from pathlib import Path
import runpy
import zipfile

import pytest

pytest.importorskip(
    "pdm.backend", reason="Bundle tests require the optional pdm-backend build tool"
)

from duma.integrations.kaggle.protocol import load_protocol
from duma.utils.utils import DATA_DIR


ROOT = Path(__file__).resolve().parents[1]
build_bundle = runpy.run_path(str(ROOT / "benchmarks/kaggle/build_bundle.py"))[
    "build_bundle"
]


def test_pilot_protocol_uses_available_pinned_models():
    protocol = load_protocol(ROOT / "benchmarks/kaggle/protocol.pilot.v1.json")
    protocol.require_service_models()
    protocol.validate_data(DATA_DIR)
    assert len(protocol.tasks) == 3
    assert protocol.max_steps == 40
    assert protocol.user.model.startswith("google/")
    assert protocol.judge.model.startswith("google/")


def test_bundle_is_reproducible_and_excludes_credentials(tmp_path):
    protocol = ROOT / "benchmarks/kaggle/protocol.pilot.v1.json"
    first, second = tmp_path / "first", tmp_path / "second"
    digest = build_bundle(ROOT, first, protocol, "example")
    assert build_bundle(ROOT, second, protocol, "example") == digest
    assert hashlib.sha256((first / "duma-pilot.zip").read_bytes()).hexdigest() == digest
    with zipfile.ZipFile(first / "duma-pilot.zip") as archive:
        names = archive.namelist()
        wheel_name = next(name for name in names if name.endswith(".whl"))
        with zipfile.ZipFile(io.BytesIO(archive.read(wheel_name))) as wheel:
            assert "duma/integrations/kaggle/runner.py" in wheel.namelist()
            assert "duma/runtime.py" in wheel.namelist()
            assert "duma/run.py" not in wheel.namelist()
            assert "duma/utils/llm_utils.py" not in wheel.namelist()
            assert "duma/environment/server.py" not in wheel.namelist()
            metadata_path = next(
                name for name in wheel.namelist() if name.endswith("/METADATA")
            )
            metadata = wheel.read(metadata_path).decode()
            assert "Requires-Dist: pydantic" in metadata
            for forbidden in (
                "litellm",
                "langfuse",
                "pandas",
                "openai",
                "fastapi",
                "kaggle-benchmarks",
            ):
                assert f"Requires-Dist: {forbidden}" not in metadata
        assert "uv.lock" not in names
        assert "pyproject.toml" not in names
        assert "data/duma/domains/collab/tasks.json" in names
        assert not any(
            ".env" in name or ".git/" in name or "__pycache__" in name for name in names
        )
        hashes = json.loads(archive.read("bundle-files.json"))
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    hosted = (first / "hosted_pilot.py").read_text()
    assert digest in hosted
    assert "__DUMA_BUNDLE_SHA256__" not in hosted
    assert "__DUMA_MANIFEST_SHA256__" not in hosted
    compile(hosted, "hosted_pilot.py", "exec")
    metadata = json.loads((first / "dataset-metadata.json").read_text())
    assert metadata["id"] == "example/duma-bench-private-pilot-runtime"
    assert "isPublic" not in metadata
    with pytest.raises(FileExistsError):
        build_bundle(ROOT, first, protocol, "example")


def hosted_helpers():
    # Load pure helpers without executing installation or model inference.
    tree = ast.parse((ROOT / "benchmarks/kaggle/hosted_pilot.py").read_text())
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.ImportFrom) and node.module.startswith("duma"))
        and not (
            isinstance(node, ast.Import) and node.names[0].name == "kaggle_benchmarks"
        )
        or isinstance(node, ast.FunctionDef)
        and node.name in {"copy_verified_assets", "check_host"}
    ]
    namespace = {}
    exec(compile(tree, "hosted-assets", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("extracted", [False, True])
def test_hosted_assets_verify_archived_and_kaggle_extracted_data(tmp_path, extracted):
    output = tmp_path / "dataset"
    digest = build_bundle(
        ROOT, output, ROOT / "benchmarks/kaggle/protocol.pilot.v1.json", "example"
    )
    archive_path = output / "duma-pilot.zip"
    with zipfile.ZipFile(archive_path) as archive:
        manifest_digest = hashlib.sha256(archive.read("bundle-files.json")).hexdigest()
        if extracted:
            archive.extractall(output / "duma-pilot")
    if extracted:
        archive_path.unlink()
    copy_assets = hosted_helpers()["copy_verified_assets"]
    runtime = tmp_path / "runtime"
    copy_assets(output, runtime, digest, manifest_digest)
    assert (runtime / "protocol.json").read_bytes() == (
        ROOT / "benchmarks/kaggle/protocol.pilot.v1.json"
    ).read_bytes()
    if extracted:
        (output / "duma-pilot/protocol.json").write_text("tampered")
        with pytest.raises(ValueError, match="asset hash mismatch"):
            copy_assets(output, tmp_path / "other-runtime", digest, manifest_digest)


def test_host_check_rejects_full_distribution_and_unsupported_sdk(monkeypatch):
    helpers = hosted_helpers()
    with pytest.raises(RuntimeError, match="full DUMA"):
        helpers["check_host"]()
    monkeypatch.setattr(helpers["importlib"].metadata, "version", lambda _: "0.7.0")
    with pytest.raises(RuntimeError, match="Unsupported Kaggle"):
        helpers["check_host"]()

    def installed(name):
        if name == "duma":
            raise helpers["importlib"].metadata.PackageNotFoundError(name)
        return "0.6.1"

    monkeypatch.setattr(helpers["importlib"].metadata, "version", installed)
    with pytest.raises(RuntimeError, match="Restart the notebook"):
        helpers["check_host"]()
