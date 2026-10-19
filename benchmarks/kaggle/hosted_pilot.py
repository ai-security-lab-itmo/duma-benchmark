# %%
"""Hosted pilot template. Build with build_bundle.py before uploading."""

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import zipfile


def copy_verified_assets(input_root, runtime, expected_sha, expected_manifest_sha):
    # Kaggle extracts uploaded ZIP files; other runtimes may retain the archive.
    candidates = []
    visited = set()
    for directory, subdirs, files in os.walk(input_root, followlinks=True):
        resolved = Path(directory).resolve()
        if resolved in visited:
            subdirs[:] = []
            continue
        visited.add(resolved)
        for name in ("bundle-files.json", "duma-pilot.zip"):
            if name in files:
                candidates.append(Path(directory) / name)
    if len(candidates) != 1:
        raise ValueError(
            f"Attach exactly one DUMA pilot bundle; found {len(candidates)}"
        )
    source = candidates[0]
    runtime.mkdir(parents=True, exist_ok=True)
    if source.name.endswith(".zip"):
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha:
            raise ValueError("Attached DUMA archive does not match the task version")
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                member = Path(info.filename)
                if member.is_absolute() or ".." in member.parts or info.is_dir():
                    raise ValueError("Unexpected path in DUMA bundle")
            archive.extractall(runtime)
        source = runtime / "bundle-files.json"
    manifest_bytes = source.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha:
        raise ValueError("Attached DUMA manifest does not match the task version")
    assets_root = source.parent.resolve()
    manifest = json.loads(manifest_bytes)
    for name, expected in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unexpected path in DUMA manifest")
        path = (assets_root / relative).resolve()
        path.relative_to(assets_root)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"DUMA asset hash mismatch: {name}")
    for name in manifest:
        source_path = assets_root / name
        destination = runtime / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != destination.resolve():
            shutil.copyfile(source_path, destination)
    (runtime / "bundle-files.json").write_bytes(manifest_bytes)


def runtime_constraints(lock_path):
    versions = {}
    for package in tomllib.loads(lock_path.read_text())["package"]:
        if "registry" in package.get("source", {}):
            versions.setdefault(package["name"], set()).add(package["version"])
    pins = {
        name: next(iter(values))
        for name, values in versions.items()
        if len(values) == 1
    }
    # Already installed host packages can be imported by the notebook kernel.
    # Keep them intact, including transitive SDK requirements such as protobuf.
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            pins[name.lower().replace("_", "-").replace(".", "-")] = (
                distribution.version
            )
    return (
        "\n".join(f"{name}=={version}" for name, version in sorted(pins.items())) + "\n"
    )


def prepare_runtime():
    expected_sha = "__DUMA_BUNDLE_SHA256__"
    expected_manifest_sha = "__DUMA_MANIFEST_SHA256__"
    runtime = Path("/kaggle/working/duma-runtime") / expected_manifest_sha
    copy_verified_assets(
        Path("/kaggle/input"), runtime, expected_sha, expected_manifest_sha
    )

    constraints = runtime / "host-constraints.txt"
    constraints.write_text(runtime_constraints(runtime / "uv.lock"))
    sdk_requirement = (
        f"kaggle-benchmarks=={importlib.metadata.version('kaggle-benchmarks')}"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--progress-bar",
            "off",
            "--quiet",
            "--constraint",
            str(constraints),
            str(runtime),
            sdk_requirement,
        ],
        check=True,
        timeout=600,
    )
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True, timeout=60)
    importlib.invalidate_caches()
    environment = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata.get("Name")
    }
    Path("/kaggle/working/duma-runtime-environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True)
    )
    os.environ["DUMA_DATA_DIR"] = str(runtime / "data")
    os.environ["DUMA_KAGGLE_PROTOCOL"] = str(runtime / "protocol.json")
    os.environ["DUMA_KAGGLE_OUTPUT"] = "/kaggle/working/duma-kaggle-results"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    print(
        json.dumps(
            {
                "bundle_sha256": expected_sha,
                "manifest_sha256": expected_manifest_sha,
                "runtime": str(runtime),
            }
        )
    )


prepare_runtime()

# %%
import kaggle_benchmarks as kbench  # noqa: E402

from duma.integrations.kaggle.protocol import load_protocol  # noqa: E402
from duma.integrations.kaggle.runner import run_episode  # noqa: E402

kbench.config.interactive_mode = False
print("Available models:", sorted(kbench.llms.keys()))


@kbench.task(name="duma-pilot")
def duma_pilot(
    llm,
    domain: str = "collab",
    task_id: str = "collab_poisoning_logs",
    regime: str = "dual",
    trial: int = 0,
) -> bool:
    return run_episode(
        llm,
        load_protocol(os.environ["DUMA_KAGGLE_PROTOCOL"]),
        domain=domain,
        task_id=task_id,
        regime=regime,
        trial=trial,
        output_dir=Path(os.environ["DUMA_KAGGLE_OUTPUT"]),
    )


# %%
if __name__ == "__main__":
    duma_pilot.run(kbench.llm)
