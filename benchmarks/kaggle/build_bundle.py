"""Build a credential-free private Dataset and its hash-pinned hosted task."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import runpy
import subprocess
import sys
import tempfile
import zipfile


RUNTIME_PROJECT = """[build-system]
requires = ["pdm-backend"]
build-backend = "pdm.backend"
[project]
name = "duma-kaggle-runtime"
version = "0.1.1"
requires-python = ">=3.11"
license = "MIT"
license-files = ["LICENSE", "NOTICE"]
dependencies = ["pydantic>=2.11,<3", "docstring-parser>=0.16,<1", "typing-extensions>=4.12,<5", "loguru>=0.7.3,<1"]
"""


def build_wheel(root: Path, destination: Path) -> Path:
    """Build shared sources, never the research distribution's dependency graph."""
    package = root / "src/duma"
    files = {
        package / name
        for name in ("__init__.py", "config.py", "registry.py", "runtime.py")
    }
    files.add(package / "integrations/__init__.py")
    for folder in (
        "agent",
        "data_model",
        "domains",
        "evaluator",
        "user",
        "integrations/kaggle",
    ):
        files.update((package / folder).rglob("*.py"))
    for folder, names in {
        "environment": ("__init__", "db", "environment", "tool", "toolkit"),
        "orchestrator": ("__init__", "orchestrator", "utils"),
        "utils": ("__init__", "utils", "io_utils", "pydantic_utils", "generation"),
    }.items():
        files.update(
            package / folder / f"{name}.py"
            for name in names
            if (package / folder / f"{name}.py").exists()
        )
    destination = destination.resolve()
    with tempfile.TemporaryDirectory() as directory:
        stage = Path(directory)
        (stage / "pyproject.toml").write_text(RUNTIME_PROJECT)
        shutil.copyfile(root / "LICENSE", stage / "LICENSE")
        shutil.copyfile(root / "NOTICE", stage / "NOTICE")
        for path in sorted(files):
            if path.is_symlink():
                raise ValueError(f"Source symlinks are not allowed: {path}")
            target = stage / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pdm.backend import build_wheel; import sys; build_wheel(sys.argv[1])",
                str(destination),
            ],
            cwd=stage,
            env={**os.environ, "SOURCE_DATE_EPOCH": "1767225600"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode:
            raise RuntimeError(
                f"Install pdm-backend in the build environment. {result.stderr}"
            )
    return next(destination.glob("duma_kaggle_runtime-*.whl"))


def build_bundle(
    root: Path, output: Path, protocol: Path, owner: str, evaluation="check"
) -> str:
    if evaluation not in {"check", "smoke", "solo", "dual"}:
        raise ValueError("Evaluation must be check, smoke, solo or dual")
    root, output = root.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory for a new bundle")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as wheel_dir:
        wheel = build_wheel(root, Path(wheel_dir))
        contents = {wheel.name: wheel.read_bytes()}
    paths = [root / "LICENSE", root / "NOTICE"]
    paths += sorted((root / "data" / "duma").rglob("*.json"))
    paths += sorted((root / "data" / "duma").rglob("*.md"))
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected an ordinary source file: {path}")
        contents[path.relative_to(root).as_posix()] = path.read_bytes()
    contents["protocol.json"] = protocol.read_bytes()
    revision = runpy.run_path(
        str(root / "src/duma/integrations/kaggle/task_revision.py")
    )
    protocol_version = json.loads(contents["protocol.json"])["version"]
    if protocol_version in revision["HARDENED_PROTOCOLS"]:
        pinned = json.loads(contents["protocol.json"]).get("task_data_sha256")
        actual = {
            name.removeprefix("data/"): hashlib.sha256(value).hexdigest()
            for name, value in contents.items()
            if name.startswith("data/")
        }
        if not pinned or pinned != actual:
            raise ValueError(
                "Hardened task data hash mismatch; create a new protocol revision"
            )
    revision["revise_assets"](contents, protocol_version)
    is_v2 = protocol_version in revision["V2_PROTOCOLS"]
    is_v3 = protocol_version in revision["V3_PROTOCOLS"]
    is_v4 = protocol_version in revision["V4_PROTOCOLS"]
    is_v3_pilot = protocol_version == "duma-kaggle-private-pilot-v6"
    prefix = (
        "duma-v4"
        if is_v4
        else "duma-v3-pilot"
        if is_v3_pilot
        else "duma-v3"
        if is_v3
        else "duma-v2"
        if is_v2
        else "duma"
    )
    file_hashes = {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }
    contents["bundle-files.json"] = json.dumps(
        file_hashes, indent=2, sort_keys=True
    ).encode()
    bundle = output / "duma-pilot.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    manifest_digest = hashlib.sha256(contents["bundle-files.json"]).hexdigest()
    template = (root / "benchmarks/kaggle/hosted_pilot.py").read_text()
    (output / "hosted_pilot.py").write_text(
        template.replace("__DUMA_BUNDLE_SHA256__", digest)
        .replace("__DUMA_EVALUATION_MODE__", evaluation)
        .replace(
            "__DUMA_TASK_NAME__",
            f"{prefix}-runtime-check"
            if evaluation == "check"
            else f"{prefix}-pilot"
            if evaluation == "smoke"
            else f"{prefix}-{evaluation}",
        )
        .replace("__DUMA_MANIFEST_SHA256__", manifest_digest)
    )
    metadata = {
        "title": "DUMA Bench Private V2 Runtime"
        if is_v2
        else "DUMA Bench Private Pilot Runtime",
        "id": f"{owner}/duma-bench-private-v2-runtime"
        if is_v2
        else f"{owner}/duma-bench-private-pilot-runtime",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
        "description": "Private DUMA Kaggle v2 code, explicitly revised task assets and pinned protocol. No model credentials or experiment results are included."
        if is_v2
        else "Private DUMA pilot code, original task assets and pinned pilot protocol. No model credentials or experiment results are included.",
    }
    if is_v3:
        metadata.update(
            title="DUMA Bench Private V3 Hardened Runtime",
            id=f"{owner}/duma-bench-private-v3-runtime",
            description="Private DUMA hardened tasks, minimal runtime and hash-pinned v3 protocol. Not comparable with v1/v2 results. No credentials or experiment results included.",
        )
        if is_v3_pilot:
            metadata.update(
                title="DUMA Bench Private V3 Pilot Runtime",
                id=f"{owner}/duma-bench-private-v3-pilot-runtime",
            )
    if is_v4:
        metadata.update(
            title="DUMA Bench Private V4 Runtime",
            id=f"{owner}/duma-bench-private-v4-runtime",
            description=(
                "Private DUMA hardened tasks and runtime 0.1.1; one V4 protocol with explicit "
                "model-specific reasoning: minimal for Qwen3-235B Instruct/GLM-5, none by default. "
                "Minimal is not assumed equivalent to disabled reasoning. Not pooled with V3. "
                "No credentials or experiment results included. "
                "CRM identity/contact/passport records are entirely fictional author-confirmed fixtures. "
                "Scenario secrets are test values. Do not visit scenario URLs, contact scenario "
                "addresses or submit data; domain ownership/inactivity is not guaranteed. "
                "All domain tools must remain connected only to the supplied simulated state."
            ),
        )
    metadata["description"] += (
        "\n\nLicense: MIT. See the included LICENSE file for the full terms. "
        "Kaggle metadata uses 'Other' because its supported Dataset license list "
        "does not include MIT; this does not relicense the code or task assets."
    )
    (output / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "bundle-sha256.txt").write_text(digest + "\n")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument(
        "--evaluation", choices=["check", "smoke", "solo", "dual"], default="check"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("protocol.pilot.v6.json"),
    )
    args = parser.parse_args()
    print(
        build_bundle(
            Path(__file__).resolve().parents[2],
            args.output,
            args.protocol,
            args.owner,
            args.evaluation,
        )
    )
