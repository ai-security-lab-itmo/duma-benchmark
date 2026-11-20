"""Generate Sonnet-only private tasks with a separate, explicit profile."""

import argparse
import hashlib
import json
from pathlib import Path


TRANSPORTS = {
    "omit": {},
    "anthropic": {
        "extra_body": {
            "extra_body": {"anthropic": {"thinking": {"type": "between_tools"}}}
        }
    },
    "top": {"extra_body": {"thinking": {"type": "between_tools"}}},
}


def build(phase, transport, output):
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
    adapter = (root / "sonnet_default.py").read_bytes()
    sha = hashlib.sha256(adapter).hexdigest()
    task_name = f"duma-sonnet55-transport-{phase}-v2"
    source = (
        bootstrap
        + f"\n# %%\nSONNET_ADAPTER_SHA256 = {sha!r}\nSONNET_PHASE = {phase!r}\n"
        + f"SONNET_TRANSPORT_ARGS = {TRANSPORTS[transport]!r}\n"
        + adapter.decode().replace("__SONNET_TASK_NAME__", task_name)
    )
    compile(source, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    print(
        json.dumps(
            {
                "phase": phase,
                "transport": transport,
                "task": task_name,
                "adapter_sha256": sha,
                "hosted_sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("preflight", "preflight-dual", "solo", "dual"),
        required=True,
    )
    parser.add_argument("--transport", choices=tuple(TRANSPORTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.phase, args.transport, args.output)
