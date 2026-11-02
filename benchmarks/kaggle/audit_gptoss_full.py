"""Independently audit only the explicit GPT-OSS dual profile."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy

from loguru import logger

from duma.integrations.kaggle.artifacts import (
    checksum,
    read_checkpoint,
    write_checkpoint,
)
from duma.integrations.kaggle.metrics import aggregate, evaluation_rows
from duma.integrations.kaggle.protocol import KaggleProtocol, fingerprint_files
from duma.utils.utils import DATA_DIR


def select_retry(items, new, source_hash):
    if checksum(new) != new.get("artifact_hash"):
        raise ValueError("Retry checksum mismatch")
    matches = [i for i, old in enumerate(items) if old["metadata"] == new["metadata"]]
    if len(matches) != 1:
        raise ValueError("Retry identity differs from exactly one original row")
    index = matches[0]
    old = items[index]
    if (
        old["status"] == "completed"
        or old["score"] is not None
        or old["artifact_hash"] != source_hash
    ):
        raise ValueError("Cannot replace a completed or undeclared source row")
    selected = list(items)
    selected[index] = new
    return selected, index


def audit(results, bundle, retry_path=None):
    paths = sorted(results.glob("**/duma-kaggle-results/*.json"))
    if len(paths) != 105:
        raise ValueError("Expected all 105 artifacts")
    items = []
    for path in paths:
        raw = json.loads(path.read_text())
        items.append(read_checkpoint(path, raw["metadata"]))
    adapter = Path(__file__).with_name("gptoss_compat_preflight.py")
    profile_data = items[0]["metadata"]["protocol"]
    if (
        hashlib.sha256(adapter.read_bytes()).hexdigest()
        != profile_data["wire_adapter_sha256"]
    ):
        raise ValueError("Adapter identity mismatch")
    ns = {"KaggleProtocol": KaggleProtocol}
    classes = [
        n for n in ast.parse(adapter.read_text()).body if isinstance(n, ast.ClassDef)
    ]
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(adapter), "exec"), ns)
    profile = ns["GPTOSSProtocol"].model_validate(profile_data)
    if (
        profile.fingerprint
        != "7792ba790baee00c290b213c431203b4f969241e25e174a44764326c53dc0315"
    ):
        raise ValueError("Unexpected measured profile")
    profile.validate_data(DATA_DIR)
    helper = runpy.run_path(str(Path(__file__).with_name("verify_results.py")))
    source = helper["wheel_identity"](bundle)
    data = fingerprint_files(
        DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
    )
    expected = {
        (r["domain"], r["task_id"], r["trial"], r["seed"])
        for r in evaluation_rows(profile, "dual")
    }
    seen, rows, nl = set(), [], 0
    for item in items:
        meta = item["metadata"]
        key = tuple(meta[k] for k in ("domain", "task_id", "trial", "seed"))
        if (
            key not in expected
            or key in seen
            or meta["target_model"] != "openai/gpt-oss-120b"
            or meta["regime"] != "dual"
        ):
            raise ValueError("Duplicate or unexpected row")
        seen.add(key)
        checked = helper["verify_episode"](item, profile, source, data)
        if checked:
            nl += int(checked["nl_scored"])
        rows.append(
            {
                "row": {k: meta[k] for k in ("domain", "task_id", "trial", "seed")},
                "status": item["status"],
                "score": item["score"],
                "artifact_hash": item["artifact_hash"],
                **(checked or {}),
            }
        )
    summaries = list(results.glob("**/duma-kaggle-results/summaries/matrix.json"))
    if len(summaries) != 1 or seen != expected:
        raise ValueError("Missing matrix/summary")
    stored = read_checkpoint(summaries[0], None)
    actual = aggregate(items, profile, "openai/gpt-oss-120b", "dual")
    if any(
        not helper["same_summary"](json.loads(json.dumps(v)), stored.get(k))
        for k, v in actual.items()
    ):
        raise ValueError("Hosted summary differs from independent aggregation")
    for row in evaluation_rows(profile, "dual"):
        wire_paths = list(
            results.glob(
                f"**/summaries/{row['domain']}-{row['task_id']}-{row['seed']}-wire.json"
            )
        )
        if len(wire_paths) != 1:
            raise ValueError("Missing wire record")
        wire = read_checkpoint(wire_paths[0], None)
        if wire["row"] != row or wire["profile_hash"] != profile.fingerprint:
            raise ValueError("Wire record identity mismatch")
    original_matrix, retry_provenance = actual, None
    if retry_path:
        prior = read_checkpoint(results / "verified-incomplete.json", None)
        if prior[
            "artifact_hash"
        ] != "f06f7290cfb3074a93cb19f77d3b93d775958c9a6d5cb39199330b846705c94f" or {
            x["artifact_hash"] for x in prior["rows"]
        } != {x["artifact_hash"] for x in items}:
            raise ValueError("Original matrix differs from pinned audit")
        raw = json.loads(retry_path.read_text())
        new = read_checkpoint(retry_path, raw["metadata"])
        items, index = select_retry(
            items,
            new,
            "f31cab08ae2e84939b6731cc7c4f54df83551611e745824f46583a4224e8a73a",
        )
        checked = helper["verify_episode"](new, profile, source, data)
        if checked:
            nl += int(checked["nl_scored"])
        retry_provenance = {
            "source_audit_hash": prior["artifact_hash"],
            "original_artifact_hash": rows[index]["artifact_hash"],
            "new_artifact_hash": new["artifact_hash"],
            "preserved_rows": 104,
        }
        rows[index] = {
            "row": rows[index]["row"],
            "status": new["status"],
            "score": new["score"],
            "artifact_hash": new["artifact_hash"],
            **(checked or {}),
        }
        actual = aggregate(items, profile, "openai/gpt-oss-120b", "dual")
    return {
        "profile_hash": profile.fingerprint,
        "source_hash": source,
        "data_hash": data,
        "verified_rows": len(rows),
        "verified_nl_rows": nl,
        "verified_wire_rows": 105,
        "matrix": actual,
        "original_matrix": original_matrix,
        "retry_provenance": retry_provenance,
        "hosted_summary_hash": stored["artifact_hash"],
        "rows": rows,
        "scope": "Original strict scoring, no model/judge calls or safe-block adjudication; incomplete matrix has no final score",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("results", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--retry", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Immutable audit output already exists")
    logger.remove()
    report = audit(args.results, args.bundle, args.retry)
    write_checkpoint(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
