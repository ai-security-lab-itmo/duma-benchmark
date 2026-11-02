"""Append to frozen bootstrap (and unchanged Sonnet definitions when needed)."""

import hashlib
import json
from pathlib import Path

from duma.integrations.kaggle.artifacts import (
    checksum,
    read_checkpoint,
    write_checkpoint,
)
from duma.integrations.kaggle.protocol import KaggleProtocol, fingerprint_files
from duma.utils.utils import DATA_DIR


ROW_FIELDS = ("domain", "task_id", "regime", "trial", "seed")


def validate_selection(manifest, expected_hash):
    if (
        manifest.get("artifact_hash") != expected_hash
        or checksum(manifest) != expected_hash
    ):
        raise ValueError("Resume selection checksum mismatch")
    static = manifest["static_metadata"]
    protocol = static["protocol"]
    seen = set()
    for retry, rows in (
        (False, manifest["preserved_rows"]),
        (True, manifest["retry_rows"]),
    ):
        for row in rows:
            key = tuple(row[k] for k in ROW_FIELDS)
            if key in seen:
                raise ValueError("Duplicate or overlapping resume selection")
            seen.add(key)
            if retry and (
                row["status"] not in {"technical_error", "incomplete"}
                or row["score"] is not None
            ):
                raise ValueError("Cannot resample a completed row")
            if not retry and (
                row["status"] != "completed" or row["score"] not in (0, 1)
            ):
                raise ValueError("Invalid preserved result")
            metadata = {**static, **{key: row[key] for key in ROW_FIELDS}}
            if (
                hashlib.sha256(
                    json.dumps(metadata, sort_keys=True).encode()
                ).hexdigest()
                != row["metadata_hash"]
            ):
                raise ValueError("Original row metadata mismatch")
    expected = {
        (domain, task, "dual", trial, seed)
        for domain, tasks in protocol["tasks"].items()
        for task in tasks
        for trial, seed in enumerate(protocol["trial_seeds"])
    }
    if seen != expected or len(seen) != 105:
        raise ValueError("Resume must cover exactly the original 105-row matrix")
    if (
        static["target_model"] != manifest["target_model"]
        or static["sdk_version"] != "0.6.2"
    ):
        raise ValueError("Unexpected resume target or original SDK")
    return static


def execute_selection(llm, manifest, expected_hash, protocol, output, backend_class):
    import kaggle_benchmarks as kbench
    import duma
    from duma.integrations.kaggle import runner

    static = validate_selection(manifest, expected_hash)
    if str(getattr(llm, "model", llm.name)) != manifest["target_model"]:
        raise ValueError("Selected model differs from audited target")
    protocol.validate_data(DATA_DIR)
    if (
        protocol.model_dump() != static["protocol"]
        or protocol.fingerprint != static["protocol_hash"]
        or kbench.__version__ != static["sdk_version"]
        or fingerprint_files(Path(duma.__file__).parent, ("**/*.py",))
        != static["source_hash"]
        or fingerprint_files(
            DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
        )
        != static["data_hash"]
    ):
        raise ValueError("Resume protocol/source/data/SDK identity mismatch")
    services = {
        role: str(getattr(kbench.llms[slug], "model", kbench.llms[slug].name))
        for role, slug in {
            "user": protocol.user.model,
            "judge": protocol.judge.model,
        }.items()
    }
    if services != static["service_models"]:
        raise ValueError("Pinned service models changed")
    stop = {}

    class ObservedBackend(backend_class):
        def __call__(self, **kwargs):
            try:
                return super().__call__(**kwargs)
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status in {401, 403, 404, 429}:
                    stop.update(http_status=status, error_type=type(exc).__name__)
                raise

    output.mkdir(parents=True, exist_ok=True)
    reports = output / "resume"
    reports.mkdir(exist_ok=True)
    write_checkpoint(reports / "selection.json", manifest)
    attempted = []
    original_backend = runner.KaggleBackend
    runner.KaggleBackend = ObservedBackend
    try:
        for row in manifest["retry_rows"]:
            metadata = {**static, **{key: row[key] for key in ROW_FIELDS}}
            path = output / (row["metadata_hash"] + ".json")
            if (
                path.exists()
                and read_checkpoint(path, metadata)["status"] != "completed"
            ):
                raise RuntimeError(
                    "A saved unfinished retry requires a new explicit selection"
                )
            if not path.exists():
                try:
                    runner.run_episode(
                        llm,
                        protocol,
                        **{key: row[key] for key in ROW_FIELDS if key != "seed"},
                        output_dir=output,
                    )
                except Exception:
                    if not path.exists():
                        raise
            item = read_checkpoint(path, metadata)
            attempted.append(
                {
                    "metadata_hash": row["metadata_hash"],
                    "source_artifact_hash": row["source_artifact_hash"],
                    "new_artifact_hash": item["artifact_hash"],
                    "status": item["status"],
                    "score": item["score"],
                }
            )
            print(
                json.dumps(
                    {
                        **{key: row[key] for key in ROW_FIELDS},
                        "status": item["status"],
                        "score": item["score"],
                    }
                ),
                flush=True,
            )
            if stop:
                break
    finally:
        runner.KaggleBackend = original_backend
    report = {
        "scope": "Retry subset only, not a full leaderboard score",
        "manifest_hash": expected_hash,
        "expected_retries": len(manifest["retry_rows"]),
        "completed_retries": sum(row["status"] == "completed" for row in attempted),
        "preserved_original_rows": len(manifest["preserved_rows"]),
        "attempts": attempted,
        "not_attempted": [
            row["metadata_hash"] for row in manifest["retry_rows"][len(attempted) :]
        ],
        "early_stop": stop,
    }
    write_checkpoint(reports / "summary.json", report)
    return {key: value for key, value in report.items() if key != "attempts"}


import kaggle_benchmarks as kbench  # noqa: E402

kbench.config.interactive_mode = False


@kbench.task(name="__RESUME_TASK_NAME__")
def resume_task(llm) -> dict:
    if str(getattr(llm, "model", llm.name)) != RESUME_MANIFEST["target_model"]:
        return {"skipped": "Target-specific readiness; no inference or security score"}
    from duma.integrations.kaggle.backend import KaggleBackend

    if RESUME_MANIFEST["target_model"].startswith("anthropic/"):
        protocol = SonnetProtocol.model_validate(
            RESUME_MANIFEST["static_metadata"]["protocol"]
        )
        backend = SonnetBackend
    else:
        protocol = KaggleProtocol.model_validate(
            RESUME_MANIFEST["static_metadata"]["protocol"]
        )
        backend = KaggleBackend
    return execute_selection(
        llm,
        RESUME_MANIFEST,
        RESUME_MANIFEST_SHA256,
        protocol,
        Path("/kaggle/working/duma-resume-results"),
        backend,
    )


if __name__ == "__main__":
    resume_task.run(kbench.llm)
