"""Build an isolated native Gemini task; never update the frozen V4 tasks."""

import argparse
import hashlib
import json
from pathlib import Path


def build(phase, output):
    root = Path(__file__).parent
    bootstrap = (
        (root / "hosted_pilot.py")
        .read_text()
        .split("# %%\nimport kaggle_benchmarks", 1)[0]
    )
    bootstrap = bootstrap.replace(
        "__DUMA_BUNDLE_SHA256__",
        "9f5f1cc07f397f81c6c435af6600271aa2b7f475bb44572e18eeabbbd206c611",
    ).replace(
        "__DUMA_MANIFEST_SHA256__",
        "aa5b1d8870b8d215564023bcc2bdb8b3593e86fcac91e132ecf9bf1853e5df89",
    )
    if "__DUMA_" in bootstrap:
        raise ValueError("Unresolved bootstrap placeholder")
    adapter = (root / "gemini25_native.py").read_bytes()
    sha = hashlib.sha256(adapter).hexdigest()
    task = f"duma-gemini25-native-{phase}-v1"
    source = (
        bootstrap
        + f"\n# %%\nGEMINI25_ADAPTER_SHA256 = {sha!r}\nGEMINI25_PHASE = {phase!r}\n"
        + adapter.decode().replace("__GEMINI25_TASK_NAME__", task)
    )
    compile(source, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    print(
        json.dumps(
            {
                "task": task,
                "adapter_sha256": sha,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "solo", "dual"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.phase, args.output)
