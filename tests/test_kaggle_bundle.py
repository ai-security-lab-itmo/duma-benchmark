import ast
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import zipfile

import pytest

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
        assert "src/duma/integrations/kaggle/runner.py" in names
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
        and node.name in {"copy_verified_assets", "runtime_constraints"}
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


def test_runtime_constraints_use_lock_and_preserve_host_sdk_dependencies(monkeypatch):
    helpers = hosted_helpers()
    monkeypatch.setattr(
        helpers["importlib"].metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": "Protobuf"}, version="5.29.6"),
            SimpleNamespace(metadata={"Name": "OpenAI"}, version="2.17.0"),
        ],
    )
    pins = helpers["runtime_constraints"](ROOT / "uv.lock")
    assert "protobuf==5.29.6\n" in pins
    assert "openai==2.17.0\n" in pins
    assert "litellm==1.81.15\n" in pins
    assert "langfuse==3.14.5\n" in pins
    assert "fastapi==0.133.1\n" in pins
