"""Append to the immutable native adapter; retry exactly one unfinished row."""

from duma.integrations.kaggle.artifacts import read_checkpoint
from duma.integrations.kaggle.protocol import fingerprint_files
from duma.utils.utils import DATA_DIR


def selected_matrix(llm, protocol, *, regime, output_dir):
    meta = GEMINI25_RETRY_METADATA
    if (
        protocol.model_dump() != meta["protocol"]
        or protocol.fingerprint != meta["protocol_hash"]
        or kbench.__version__ != meta["sdk_version"]
        or str(getattr(llm, "model", llm.name)) != meta["target_model"]
        or regime != "dual"
        or fingerprint_files(Path(__import__("duma").__file__).parent, ("**/*.py",))
        != meta["source_hash"]
        or fingerprint_files(
            DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
        )
        != meta["data_hash"]
    ):
        raise ValueError("Retry identity changed")
    if tuple(meta[k] for k in ("domain", "task_id", "regime", "trial", "seed")) != (
        "output_handling",
        "output_xss_block",
        "dual",
        1,
        301,
    ):
        raise ValueError("Only the declared unfinished XSS row can be retried")
    run_episode(
        llm,
        protocol,
        domain=meta["domain"],
        task_id=meta["task_id"],
        regime=regime,
        trial=meta["trial"],
        output_dir=output_dir,
        resume=False,
    )
    paths = list(output_dir.glob("*.json"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one retry artifact")
    item = read_checkpoint(paths[0], meta)
    return {"score": item["score"]}


@kbench.task(name="duma-gemini25-xss-retry-v1")
def gemini25_xss_retry(llm) -> dict:
    global run_matrix
    if str(getattr(llm, "model", llm.name)) != "google/gemini-2.5-flash":
        return {"skipped": "Target-only retry; no inference or score"}
    original = run_matrix
    report = {
        "scope": "One unfinished-row retry, not a matrix score",
        "source_artifact_hash": GEMINI25_RETRY_SOURCE_HASH,
        "source_audit_hash": GEMINI25_RETRY_AUDIT_HASH,
        "metadata": GEMINI25_RETRY_METADATA,
    }
    try:
        run_matrix = selected_matrix
        # Keep native client creation, capture and generation exactly as measured.
        gemini25_native.func(llm)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error_message=str(exc)[:2000])
    finally:
        run_matrix = original
        output = Path("/kaggle/working/duma-kaggle-results")
        paths = list(output.glob("*.json"))
        if len(paths) == 1:
            item = read_checkpoint(paths[0], GEMINI25_RETRY_METADATA)
            report.update(
                retry_status=item["status"],
                retry_score=item["score"],
                retry_artifact_hash=item["artifact_hash"],
            )
        write_checkpoint(
            Path("/kaggle/working/gemini25-xss-retry-selection.json"), report
        )
    return report


if __name__ == "__main__":
    gemini25_xss_retry.run(kbench.llm)
