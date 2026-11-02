"""Generate a private one-row retry using the measured native adapter bytes."""

import argparse
import hashlib
import json
from pathlib import Path


def build(original, audit, output):
    from duma.integrations.kaggle.artifacts import read_checkpoint

    old = read_checkpoint(original, json.loads(original.read_text())["metadata"])
    prior = read_checkpoint(audit, None)
    if (
        old["status"] != "technical_error"
        or old["score"] is not None
        or old["artifact_hash"]
        != "1a518747dcc37011122132a6903f9a977eb7fedeae5f7c4d9853b919b2adfea7"
        or prior["artifact_hash"]
        != "162ce3f7743bb5047fedf115f2eaf66950c6e8b6377f17880ca5327f0ebeb52c"
    ):
        raise ValueError("Unexpected retry source/audit")
    if old["artifact_hash"] not in {c["artifact_hash"] for c in prior["checkpoints"]}:
        raise ValueError("Retry source not present in audited original matrix")
    root = Path(__file__).parent
    adapter = (root / "gemini25_native.py").read_bytes()
    sha = hashlib.sha256(adapter).hexdigest()
    if sha != old["metadata"]["protocol"]["native_adapter_sha256"]:
        raise ValueError("Measured adapter changed")
    bootstrap = (
        (root / "hosted_pilot.py")
        .read_text()
        .split("# %%\nimport kaggle_benchmarks", 1)[0]
        .replace(
            "__DUMA_BUNDLE_SHA256__",
            "9f5f1cc07f397f81c6c435af6600271aa2b7f475bb44572e18eeabbbd206c611",
        )
        .replace(
            "__DUMA_MANIFEST_SHA256__",
            "aa5b1d8870b8d215564023bcc2bdb8b3593e86fcac91e132ecf9bf1853e5df89",
        )
    )
    definitions = adapter.decode().rsplit('\nif __name__ == "__main__":', 1)[0]
    source = (
        bootstrap
        + f"\n# %%\nGEMINI25_ADAPTER_SHA256 = {sha!r}\nGEMINI25_PHASE = 'dual'\n"
        + definitions.replace(
            "__GEMINI25_TASK_NAME__", "duma-gemini25-native-definition-v1"
        )
        + f"\n# %%\nGEMINI25_RETRY_METADATA = {old['metadata']!r}\n"
        + f"GEMINI25_RETRY_SOURCE_HASH = {old['artifact_hash']!r}\n"
        + f"GEMINI25_RETRY_AUDIT_HASH = {prior['artifact_hash']!r}\n"
        + (root / "gemini25_xss_retry.py").read_text()
    )
    compile(source, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    print(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "adapter_sha256": sha,
                "task": "duma-gemini25-xss-retry-v1",
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.original, args.audit, args.output)
