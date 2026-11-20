"""Merge only the declared Gemini technical retry, preserving all other rows."""

import argparse
import json
from pathlib import Path

from loguru import logger

from audit_gemini25_native import audit_native, load_native_protocol, verify_wire
from audit_gptoss_full import select_retry
from duma.integrations.kaggle.artifacts import (
    checksum,
    read_checkpoint,
    write_checkpoint,
)
from duma.integrations.kaggle.metrics import aggregate
from duma.utils.utils import DATA_DIR
from verify_results import verify_episode


SOURCE_AUDIT = "162ce3f7743bb5047fedf115f2eaf66950c6e8b6377f17880ca5327f0ebeb52c"
SOURCE_ARTIFACT = "1a518747dcc37011122132a6903f9a977eb7fedeae5f7c4d9853b919b2adfea7"


def audit_retry(results, retry, bundle, protocol_path, original_audit):
    prior = read_checkpoint(original_audit, None)
    if prior["artifact_hash"] != SOURCE_AUDIT:
        raise ValueError("Unexpected original audit")
    original = audit_native(results, bundle, protocol_path, "full")
    if checksum(original) != prior["artifact_hash"]:
        raise ValueError("Original results changed since audit")
    profile = load_native_protocol(protocol_path)
    paths = list(retry.glob("**/duma-kaggle-results/*.json"))
    if len(paths) != 1:
        raise ValueError("Expected one retry row")
    raw = json.loads(paths[0].read_text())
    new = read_checkpoint(paths[0], raw["metadata"])
    items = []
    for path in sorted(results.glob("**/duma-kaggle-results/*.json")):
        raw = json.loads(path.read_text())
        items.append(read_checkpoint(path, raw["metadata"]))
    selected, _ = select_retry(items, new, SOURCE_ARTIFACT)
    checked = verify_episode(
        new, profile, original["source_hash"], original["data_hash"]
    )
    selections = list(retry.glob("**/gemini25-xss-retry-selection.json"))
    if len(selections) != 1:
        raise ValueError("Missing retry selection provenance")
    selection = read_checkpoint(selections[0], new["metadata"])
    if (
        selection["source_audit_hash"] != SOURCE_AUDIT
        or selection["source_artifact_hash"] != SOURCE_ARTIFACT
        or selection["retry_artifact_hash"] != new["artifact_hash"]
        or selection["retry_status"] != new["status"]
        or selection["retry_score"] != new["score"]
    ):
        raise ValueError("Retry provenance mismatch")
    wire = []
    for path in sorted(retry.glob("**/gemini25-native-wire/*.json")):
        event = json.loads(path.read_text())
        verify_wire(event, profile.fingerprint)
        wire.append(
            {
                "path": str(path.relative_to(retry)),
                "artifact_hash": event["artifact_hash"],
                "error_type": event.get("error_type"),
            }
        )
    if not wire:
        raise ValueError("Missing retry native-wire evidence")
    matrices = [
        {
            "target": profile.target_model,
            "regime": regime,
            **aggregate(
                [x for x in selected if x["metadata"]["regime"] == regime],
                profile,
                profile.target_model,
                regime,
            ),
        }
        for regime in profile.regimes
    ]
    return {
        "complete": all(m["complete"] for m in matrices),
        "profile_hash": profile.fingerprint,
        "source_audit_hash": SOURCE_AUDIT,
        "superseded_artifact_hash": SOURCE_ARTIFACT,
        "retry_artifact_hash": new["artifact_hash"],
        "preserved_original_rows": 209,
        "preserved_dual_rows": 104,
        "preserved_dual_failures": sum(
            x["metadata"]["regime"] == "dual" and x["score"] == 0 for x in items
        ),
        "retry_assertions": checked,
        "retry_wire": wire,
        "matrices": matrices,
        "selected_checkpoints": [
            {"metadata": x["metadata"], "artifact_hash": x["artifact_hash"]}
            for x in selected
        ],
        "scope": "Original strict scoring, one technical retry; no safe-block credit or completed-failure resampling",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "retry", "bundle", "protocol", "original-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    logger.remove()
    if args.output.exists():
        raise FileExistsError("Immutable audit output already exists")
    report = audit_retry(
        args.results, args.retry, args.bundle, args.protocol, args.original_audit
    )
    write_checkpoint(args.output, report)
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "artifact_hash": report["artifact_hash"],
                "matrices": report["matrices"],
            }
        )
    )
    if not report["complete"]:
        raise SystemExit("Retry did not complete; no final score")
