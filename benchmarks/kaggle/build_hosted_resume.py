"""Embed audited row selections, not original trajectories, in private tasks."""

import argparse
import hashlib
import json
from pathlib import Path

from duma.integrations.kaggle.artifacts import checksum


def build(manifest_path, expected_hash, task_name, output):
    root = Path(__file__).parent
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("artifact_hash") != expected_hash
        or checksum(manifest) != expected_hash
    ):
        raise ValueError("Manifest differs from audited selection")
    prefix = (
        (root / "hosted_pilot.py")
        .read_text()
        .split("# %%\nimport kaggle_benchmarks", 1)[0]
    )
    prefix = prefix.replace(
        "__DUMA_BUNDLE_SHA256__",
        "9f5f1cc07f397f81c6c435af6600271aa2b7f475bb44572e18eeabbbd206c611",
    )
    prefix = prefix.replace(
        "__DUMA_MANIFEST_SHA256__",
        "aa5b1d8870b8d215564023bcc2bdb8b3593e86fcac91e132ecf9bf1853e5df89",
    )
    if "__DUMA_" in prefix:
        raise ValueError("Unresolved frozen bootstrap value")
    if manifest["target_model"].startswith("anthropic/"):
        adapter = (root / "sonnet_default.py").read_bytes()
        expected = manifest["static_metadata"]["protocol"]["hosted_adapter_sha256"]
        if hashlib.sha256(adapter).hexdigest() != expected:
            raise ValueError("Sonnet adapter differs from original profile")
        definitions = adapter.decode().rsplit('\nif __name__ == "__main__":', 1)[0]
        prefix += "\n# %%\n" + definitions.replace(
            "__SONNET_TASK_NAME__", "duma-unused-sonnet-resume-definitions"
        )
    source = (
        prefix
        + "\n# %%\n"
        + f"RESUME_MANIFEST = json.loads({json.dumps(manifest, sort_keys=True)!r})\n"
        + f"RESUME_MANIFEST_SHA256 = {expected_hash!r}\n"
        + (root / "hosted_resume.py")
        .read_text()
        .replace("__RESUME_TASK_NAME__", task_name)
    )
    compile(source, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    print(
        json.dumps(
            {
                "task": task_name,
                "retry_rows": len(manifest["retry_rows"]),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "bytes": len(source.encode()),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.manifest, args.manifest_hash, args.task_name, args.output)
