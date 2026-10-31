"""Merge only declared technical retries with unchanged original results."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy

from loguru import logger

from duma.data_model.message import AssistantMessage, UserMessage
from duma.integrations.kaggle.artifacts import (
    checksum,
    read_checkpoint,
    write_checkpoint,
)
from duma.integrations.kaggle.backend import KaggleBackend
from duma.integrations.kaggle.metrics import aggregate
from duma.integrations.kaggle.protocol import (
    KaggleProtocol,
    ModelSettings,
    fingerprint_files,
)
from duma.utils.utils import DATA_DIR


def metadata_hash(item):
    return hashlib.sha256(
        json.dumps(item["metadata"], sort_keys=True).encode()
    ).hexdigest()


def select_matrix(originals, retries, manifest):
    declared = {
        row["metadata_hash"]: row
        for row in manifest["retry_rows"] + manifest["preserved_rows"]
    }
    allowed = {row["metadata_hash"] for row in manifest["retry_rows"]}
    selected = {}
    for item in originals:
        key = metadata_hash(item)
        if (
            key not in declared
            or key in selected
            or checksum(item) != item.get("artifact_hash")
            or item["artifact_hash"] != declared[key]["source_artifact_hash"]
        ):
            raise ValueError("Changed, duplicate or unaudited original row")
        selected[key] = item
    if set(selected) != set(declared):
        raise ValueError("Missing original rows")
    seen, provenance = set(), []
    for item in retries:
        key = metadata_hash(item)
        if (
            key not in allowed
            or key in seen
            or checksum(item) != item.get("artifact_hash")
        ):
            raise ValueError(
                "Undeclared or duplicate retry, possibly a completed-row resample"
            )
        seen.add(key)
        old = selected[key]
        if (
            old["status"] == "completed"
            or old["score"] is not None
            or old["metadata"] != item["metadata"]
        ):
            raise ValueError("Cannot replace completed results or change identities")
        provenance.append(
            {
                "metadata_hash": key,
                "original_artifact_hash": old["artifact_hash"],
                "retry_artifact_hash": item["artifact_hash"],
                "original_status": old["status"],
                "retry_status": item["status"],
            }
        )
        selected[key] = item
    return list(selected.values()), provenance


def read_episodes(paths):
    items = []
    for path in sorted(paths):
        raw = json.loads(path.read_text())
        items.append(read_checkpoint(path, raw["metadata"]))
    return items


def audit(original_dir, retry_results, manifest, expected_hash, bundle):
    if (
        manifest.get("artifact_hash") != expected_hash
        or checksum(manifest) != expected_hash
    ):
        raise ValueError("Resume selection hash mismatch")
    root = Path(__file__).parent
    tree = ast.parse((root / "hosted_resume.py").read_text())
    funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_selection"
    ]
    ns = {**globals(), "ROW_FIELDS": ("domain", "task_id", "regime", "trial", "seed")}
    exec(
        compile(ast.Module(body=funcs, type_ignores=[]), "validate_selection", "exec"),
        ns,
    )
    static = ns["validate_selection"](manifest, expected_hash)
    if manifest["target_model"].startswith("anthropic/"):
        adapter = root / "sonnet_default.py"
        if (
            hashlib.sha256(adapter.read_bytes()).hexdigest()
            != static["protocol"]["hosted_adapter_sha256"]
        ):
            raise ValueError("Sonnet adapter changed")
        definitions = [
            node
            for node in ast.parse(adapter.read_text()).body
            if isinstance(node, ast.ClassDef)
            or isinstance(node, ast.FunctionDef)
            and node.name == "adapt_messages"
        ]
        exec(
            compile(
                ast.Module(body=definitions, type_ignores=[]), str(adapter), "exec"
            ),
            ns,
        )
        profile = ns["SonnetProtocol"].model_validate(static["protocol"])
    else:
        profile = KaggleProtocol.model_validate(static["protocol"])
    profile.validate_data(DATA_DIR)
    helpers = runpy.run_path(str(root / "verify_results.py"))
    source = helpers["wheel_identity"](bundle)
    data = fingerprint_files(
        DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
    )
    original = read_episodes(original_dir.glob("*.json"))
    retries = read_episodes(retry_results.glob("**/duma-resume-results/*.json"))
    selected, provenance = select_matrix(original, retries, manifest)
    nl = 0
    for item in selected:
        check = helpers["verify_episode"](item, profile, source, data)
        if check:
            nl += check["nl_scored"]
    summary = aggregate(selected, profile, manifest["target_model"], "dual")
    return {
        "manifest_hash": expected_hash,
        "profile_hash": profile.fingerprint,
        "complete": summary["complete"],
        "matrix": summary,
        "verified_selected_rows": len(selected),
        "verified_nl_rows": nl,
        "retry_attempts": provenance,
        "preserved_original_rows": len(manifest["preserved_rows"]),
        "selected_artifact_hashes": [item["artifact_hash"] for item in selected],
        "scope": "Declared technical retries plus unchanged originals. No model/judge calls or source edits. Matrix costs cover selected attempts only; original superseded attempt costs are additional and may be unknown.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--retry-results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logger.remove()
    if args.output.exists():
        raise FileExistsError("Resume audit reports are immutable")
    report = audit(
        args.original_dir,
        args.retry_results,
        json.loads(args.manifest.read_text()),
        args.manifest_hash,
        args.bundle,
    )
    write_checkpoint(args.output, report)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"selected_artifact_hashes", "retry_attempts"}
            },
            indent=2,
        )
    )
