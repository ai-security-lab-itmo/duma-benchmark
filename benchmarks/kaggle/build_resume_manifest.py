"""Freeze an audited retry-only selection; never rerun completed policy failures."""

import argparse
import hashlib
import json
from pathlib import Path

from duma.integrations.kaggle.artifacts import (
    checksum,
    read_checkpoint,
    write_checkpoint,
)


ROW_FIELDS = ("domain", "task_id", "regime", "trial", "seed")


def build_manifest(paths, audit, audit_hash, target, expected_retries):
    if audit.get("artifact_hash") != audit_hash or checksum(audit) != audit_hash:
        raise ValueError("Source audit checksum mismatch")
    verified = {
        row["artifact_hash"] for row in audit.get("checkpoints", audit.get("rows", []))
    }
    seen, records, static = set(), [], None
    for path in sorted(paths):
        raw = json.loads(path.read_text())
        item = read_checkpoint(path, raw["metadata"])
        meta = item["metadata"]
        if (
            item["artifact_hash"] not in verified
            or meta["target_model"] != target
            or meta["regime"] != "dual"
        ):
            raise ValueError("Episode is not in the declared audited target/matrix")
        current = {key: value for key, value in meta.items() if key not in ROW_FIELDS}
        if static is not None and current != static:
            raise ValueError("Mixed protocol/source/data/SDK/service identities")
        static = current
        key = tuple(meta[field] for field in ROW_FIELDS)
        if key in seen:
            raise ValueError("Duplicate source row")
        seen.add(key)
        if item["status"] not in {"completed", "technical_error", "incomplete"}:
            raise ValueError("Unexpected source status")
        if item["status"] != "completed" and item["score"] is not None:
            raise ValueError("Unfinished episode must not have a safety score")
        records.append(
            {
                **{field: meta[field] for field in ROW_FIELDS},
                "status": item["status"],
                "score": item["score"],
                "source_artifact_hash": item["artifact_hash"],
                "metadata_hash": hashlib.sha256(
                    json.dumps(meta, sort_keys=True).encode()
                ).hexdigest(),
            }
        )
    if not static:
        raise ValueError("Empty source matrix")
    protocol = static["protocol"]
    expected = {
        (domain, task, "dual", trial, seed)
        for domain, tasks in protocol["tasks"].items()
        for task in tasks
        for trial, seed in enumerate(protocol["trial_seeds"])
    }
    if seen != expected or len(expected) != 105:
        raise ValueError("Expected all 105 original rows, including completed failures")
    retry = [row for row in records if row["status"] != "completed"]
    preserved = [row for row in records if row["status"] == "completed"]
    if len(retry) != expected_retries:
        raise ValueError("Retry count differs from the explicitly reviewed selection")
    return {
        "version": "duma-audited-interrupted-only-resume-v1",
        "target_model": target,
        "regime": "dual",
        "source_audit_hash": audit_hash,
        "static_metadata": static,
        "retry_rows": retry,
        "preserved_rows": preserved,
        "retry_policy": "One new attempt per declared interrupted row; no completed row, including score 0, may be resampled. Original attempts retained.",
        "scope": "Selection manifest only, not a resumed execution or final score. No trajectories, credentials or paid API keys included.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-hash", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--expected-retries", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Resume selections are immutable")
    report = build_manifest(
        list(args.episodes.glob("*.json")),
        json.loads(args.audit.read_text()),
        args.audit_hash,
        args.target,
        args.expected_retries,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_checkpoint(args.output, report)
    print(
        json.dumps(
            {
                "target": args.target,
                "retry": len(report["retry_rows"]),
                "preserved": len(report["preserved_rows"]),
                "manifest_hash": report["artifact_hash"],
            }
        )
    )
