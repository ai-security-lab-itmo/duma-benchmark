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


def check_host():
    from packaging.specifiers import SpecifierSet

    sdk = importlib.metadata.version("kaggle-benchmarks")
    version = tuple(int(part) for part in sdk.split(".")[:3])
    if version[:2] != (0, 6) or version < (0, 6, 1):
        raise RuntimeError(f"Unsupported Kaggle SDK version: {sdk}; expected 0.6.x")
    try:
        importlib.metadata.version("duma")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("Use a clean environment, not the full DUMA distribution")
    if "duma" in sys.modules:
        raise RuntimeError(
            "Restart the notebook kernel before loading a runtime bundle"
        )
    for name, spec in {
        "pydantic": ">=2.11,<3",
        "typing-extensions": ">=4.12,<5",
    }.items():
        if importlib.metadata.version(name) not in SpecifierSet(spec):
            raise RuntimeError(f"Host {name} is incompatible; expected {spec}")
    return sdk


def prepare_runtime():
    check_host()
    expected_sha = "__DUMA_BUNDLE_SHA256__"
    expected_manifest_sha = "__DUMA_MANIFEST_SHA256__"
    runtime = Path("/kaggle/working/duma-runtime") / expected_manifest_sha
    copy_verified_assets(
        Path("/kaggle/input"), runtime, expected_sha, expected_manifest_sha
    )

    wheels = list(runtime.glob("duma_kaggle_runtime-*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one minimal runtime wheel")
    from packaging.specifiers import SpecifierSet

    install = [str(wheels[0])]
    for name, spec in {"docstring-parser": ">=0.16,<1", "loguru": ">=0.7.3,<1"}.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            install.append(f"{name}{spec}")
        else:
            if version not in SpecifierSet(spec):
                raise RuntimeError(f"Host {name} is incompatible; expected {spec}")
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
            "--no-deps",
            "--force-reinstall",
            *install,
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
from duma.integrations.kaggle.runner import run_episode, run_matrix  # noqa: E402

kbench.config.interactive_mode = False
print("Available models:", sorted(kbench.llms.keys()))


EVALUATION_MODE = "__DUMA_EVALUATION_MODE__"


@kbench.task(name="__DUMA_TASK_NAME__")
def duma_pilot(
    llm,
    domain: str = "collab",
    task_id: str = "collab_poisoning_logs",
    regime: str = "dual",
    trial: int = 0,
) -> float:
    protocol = load_protocol(os.environ["DUMA_KAGGLE_PROTOCOL"])
    output = Path(os.environ["DUMA_KAGGLE_OUTPUT"])
    if EVALUATION_MODE == "check":
        from duma.utils.utils import DATA_DIR

        protocol.validate_data(DATA_DIR)
        print(
            "Runtime readiness only, not a security score:",
            sum(len(ids) for ids in protocol.tasks.values()),
            "tasks",
        )
        return 1.0
    if EVALUATION_MODE != "smoke":
        return run_matrix(llm, protocol, regime=EVALUATION_MODE, output_dir=output)[
            "score"
        ]
    return float(
        run_episode(
            llm,
            protocol,
            domain=domain,
            task_id=task_id,
            regime=regime,
            trial=trial,
            output_dir=Path(os.environ["DUMA_KAGGLE_OUTPUT"]),
        )
    )


# %%
if __name__ == "__main__":
    duma_pilot.run(kbench.llm)
