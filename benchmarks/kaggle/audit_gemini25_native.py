"""Verify the separate native transport profile without calling models."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Literal

from loguru import logger

from duma.integrations.kaggle.artifacts import (
    checksum,
    read_checkpoint,
    write_checkpoint,
)
from duma.integrations.kaggle.protocol import (
    KaggleProtocol,
    fingerprint_files,
    load_protocol,
)
from duma.utils.utils import DATA_DIR
from verify_results import audit, verify_episode, wheel_identity


def load_native_protocol(base_path):
    adapter = Path(__file__).with_name("gemini25_native.py")
    nodes = [
        node
        for node in ast.parse(adapter.read_text()).body
        if isinstance(node, ast.ClassDef)
        or isinstance(node, ast.FunctionDef)
        and node.name == "native_protocol"
    ]
    ns = {"KaggleProtocol": KaggleProtocol, "Literal": Literal}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(adapter), "exec"), ns)
    return ns["native_protocol"](
        load_protocol(base_path), hashlib.sha256(adapter.read_bytes()).hexdigest()
    )


def verify_wire(item, profile_hash):
    if (
        checksum(item) != item.get("artifact_hash")
        or item["profile_hash"] != profile_hash
    ):
        raise ValueError("Native wire identity/integrity mismatch")
    if item["model"] != "google/gemini-2.5-flash":
        raise ValueError("Native target changed")
    config = item["config"]
    if (
        config["thinking_config"]["thinking_budget"] != 0
        or config["temperature"] is not None
        or config["max_output_tokens"] != 4096
    ):
        raise ValueError("Native generation parameters changed")
    tools = config["tools"]
    if len(tools) != 1 or not tools[0]["function_declarations"]:
        raise ValueError("Native tools were not grouped into one declaration list")
    # Dual agents leave tool choice unset; solo agents explicitly require ANY.
    choice = config.get("tool_config")
    mode = choice["function_calling_config"]["mode"] if choice is not None else None
    if mode not in {None, "AUTO", "ANY"}:
        raise ValueError("Native tool selection changed")


def audit_native(results, bundle, protocol_path, phase):
    protocol = load_native_protocol(protocol_path)
    protocol.validate_data(DATA_DIR)
    if phase == "full":
        report = audit(results, bundle, protocol, [protocol.target_model])
    else:
        expected = {
            ("collab", "collab_poisoning_logs", "solo", 0, 300),
            ("collab", "collab_poisoning_logs", "dual", 0, 300),
            ("output_handling", "output_xss_block", "dual", 0, 300),
        }
        source = wheel_identity(bundle)
        data = fingerprint_files(
            DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
        )
        rows, seen = [], set()
        for path in sorted(results.glob("**/duma-kaggle-results/*.json")):
            raw = json.loads(path.read_text())
            item = read_checkpoint(path, raw["metadata"])
            meta = item["metadata"]
            key = tuple(
                meta[k] for k in ("domain", "task_id", "regime", "trial", "seed")
            )
            if (
                key not in expected
                or key in seen
                or meta["target_model"] != protocol.target_model
            ):
                raise ValueError("Unexpected native preflight row")
            seen.add(key)
            checked = verify_episode(item, protocol, source, data)
            rows.append(
                {
                    "row": key,
                    "status": item["status"],
                    "score": item["score"],
                    "artifact_hash": item["artifact_hash"],
                    "assertions": checked,
                }
            )
        report = {
            "complete": seen == expected
            and all(r["status"] == "completed" for r in rows),
            "expected": len(expected),
            "observed": len(rows),
            "rows": rows,
            "scope": "Compatibility preflight, not a leaderboard score",
        }
    wire = []
    for path in sorted(results.glob("**/gemini25-native-wire/*.json")):
        event = json.loads(path.read_text())
        verify_wire(event, protocol.fingerprint)
        wire.append(
            {
                "path": str(path.relative_to(results)),
                "artifact_hash": event["artifact_hash"],
                "error_type": event.get("error_type"),
            }
        )
    if not wire:
        raise ValueError("Missing native transport evidence")
    report.update(
        profile=protocol.model_dump(), profile_hash=protocol.fingerprint, wire=wire
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--phase", choices=("preflight", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logger.remove()
    if args.output.exists():
        raise FileExistsError("Audit reports are immutable")
    report = audit_native(args.results, args.bundle, args.protocol, args.phase)
    write_checkpoint(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("complete", "expected", "observed", "profile_hash")
            }
        )
    )
    if not report["complete"]:
        raise SystemExit("Incomplete native evaluation; no final score")
