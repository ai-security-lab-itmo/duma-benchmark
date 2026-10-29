"""Verify the separate Sonnet dual profile on two real tasks without inference."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy

from loguru import logger

from duma.data_model.message import AssistantMessage, UserMessage
from duma.integrations.kaggle.artifacts import read_checkpoint, write_checkpoint
from duma.integrations.kaggle.backend import KaggleBackend
from duma.integrations.kaggle.protocol import (
    KaggleProtocol,
    ModelSettings,
    fingerprint_files,
    load_protocol,
)
from duma.utils.utils import DATA_DIR


def select_attempts(paths, retry_paths):
    selected = {}
    superseded = []
    for retry, candidates in ((False, paths), (True, retry_paths)):
        for path in candidates:
            raw = json.loads(path.read_text())
            item = read_checkpoint(path, raw["metadata"])
            meta = item["metadata"]
            key = tuple(
                meta[k] for k in ("domain", "task_id", "regime", "trial", "seed")
            )
            if retry:
                if (
                    key != ("output_handling", "output_xss_block", "dual", 0, 300)
                    or key not in selected
                ):
                    raise ValueError(
                        "Retry is not the declared interrupted communication row"
                    )
                old_path, old = selected[key]
                if (
                    old["status"] == "completed"
                    or old["score"] is not None
                    or old["metadata"] != meta
                ):
                    raise ValueError(
                        "Cannot replace a completed row or a different profile"
                    )
                superseded.append(
                    {
                        "path": str(old_path),
                        "artifact_hash": old["artifact_hash"],
                        "status": old["status"],
                    }
                )
            elif key in selected:
                raise ValueError("Duplicate original preflight row")
            selected[key] = (path, item)
    return [value[0] for value in selected.values()], superseded


def audit(results, bundle, retry_results=None):
    root = Path(__file__).parent
    path = root / "sonnet_default.py"
    tree = ast.parse(path.read_text())
    definitions = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        or isinstance(n, ast.FunctionDef)
        and n.name == "adapt_messages"
    ]
    namespace = dict(globals())
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    profile_type = namespace["SonnetProtocol"]
    base = load_protocol(root / "protocol.full.v4.json")
    helpers = runpy.run_path(str(root / "verify_results.py"))
    source = helpers["wheel_identity"](bundle)
    data = fingerprint_files(
        DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
    )
    adapter_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = {
        ("collab", "collab_poisoning_logs", "dual"),
        ("output_handling", "output_xss_block", "dual"),
    }
    seen, rows, profile_hashes = set(), [], set()
    paths, superseded = select_attempts(
        sorted(results.glob("**/duma-sonnet55-results/*.json")),
        sorted(retry_results.glob("**/duma-sonnet55-results/*.json"))
        if retry_results
        else [],
    )
    for path in paths:
        raw = json.loads(path.read_text())
        item = read_checkpoint(path, raw["metadata"])
        meta = item["metadata"]
        profile = profile_type.model_validate(meta["protocol"])
        profile.validate_data(DATA_DIR)
        if (
            profile.hosted_adapter_sha256 != adapter_sha
            or profile.base_protocol != base.model_dump()
            or meta["target_model"] != "anthropic/claude-sonnet-5-5@default"
            or profile.agent.generation_args()
            != {
                "temperature": 0,
                "max_tokens": 4096,
                "extra_body": {
                    "extra_body": {"anthropic": {"thinking": {"type": "between_tools"}}}
                },
            }
        ):
            raise ValueError("Unexpected Sonnet adapter/settings")
        key = tuple(meta[k] for k in ("domain", "task_id", "regime"))
        if (
            key not in expected
            or key in seen
            or (meta["trial"], meta["seed"]) != (0, 300)
        ):
            raise ValueError("Unexpected Sonnet preflight row")
        seen.add(key)
        check = helpers["verify_episode"](item, profile, source, data)
        if check is None:
            raise ValueError("Sonnet preflight did not complete")
        profile_hashes.add(profile.fingerprint)
        rows.append(
            {
                "domain": key[0],
                "task_id": key[1],
                "regime": key[2],
                "score": item["score"],
                "artifact_hash": item["artifact_hash"],
                **check,
            }
        )
    if seen != expected or len(profile_hashes) != 1:
        raise ValueError("Missing or mixed Sonnet preflights")
    return {
        "complete": True,
        "observed": len(rows),
        "expected": 2,
        "profile_hash": next(iter(profile_hashes)),
        "adapter_sha256": adapter_sha,
        "source_hash": source,
        "data_hash": data,
        "rows": rows,
        "superseded_technical_attempts": superseded,
        "scope": "Two dual compatibility episodes, not a full leaderboard score; no new model/judge calls",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-results", type=Path)
    args = parser.parse_args()
    logger.remove()
    if args.output.exists():
        raise FileExistsError("Audit reports are immutable")
    report = audit(args.results, args.bundle, args.retry_results)
    write_checkpoint(args.output, report)
    print(json.dumps(report, indent=2))
