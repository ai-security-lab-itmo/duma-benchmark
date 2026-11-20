"""Append to frozen compatibility definitions: one explicitly selected retry."""

from duma.integrations.kaggle.artifacts import read_checkpoint
from duma.integrations.kaggle.protocol import fingerprint_files
from duma.utils.utils import DATA_DIR


@kbench.task(name="duma-gptoss-sql-retry-v1")
def sql_retry(llm) -> dict:
    if str(getattr(llm, "model", llm.name)) != "openai/gpt-oss-120b":
        return {"skipped": "Target-only retry, no inference or score"}
    kbench.config.interactive_mode = False
    base = load_protocol(os.environ["DUMA_KAGGLE_PROTOCOL"])
    protocol = make_protocol(base, GPTOSS_ADAPTER_SHA256)
    protocol.validate_data(DATA_DIR)
    if (
        protocol.model_dump() != SQL_RETRY_METADATA["protocol"]
        or kbench.__version__ != SQL_RETRY_METADATA["sdk_version"]
    ):
        raise ValueError("Changed retry protocol/SDK")
    if (
        fingerprint_files(Path(__import__("duma").__file__).parent, ("**/*.py",))
        != SQL_RETRY_METADATA["source_hash"]
        or fingerprint_files(
            DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
        )
        != SQL_RETRY_METADATA["data_hash"]
    ):
        raise ValueError("Changed retry source/data")
    create = llm.client.chat.completions.create
    wire = {
        "source_artifact_hash": SQL_RETRY_SOURCE_HASH,
        "source_audit_hash": SQL_RETRY_AUDIT_HASH,
        "profile_hash": protocol.fingerprint,
        "target_calls": [],
    }
    output = Path("/kaggle/working/duma-gptoss-retry-results")
    wire_path = Path("/kaggle/working/gptoss-sql-retry-wire.json")

    def capture(*args, **kwargs):
        if args:
            raise ValueError("Expected keyword request")
        if kwargs.get("model") != llm.model:
            return create(**kwargs)
        changed = normalize_tools(kwargs)
        call = {"original_request": kwargs, "effective_request": changed}
        wire["target_calls"].append(call)
        try:
            response = create(**changed)
            call["response"] = response.model_dump(mode="json")
            return response
        except Exception as exc:
            call.update(error_type=type(exc).__name__, error_message=str(exc)[:1200])
            raise
        finally:
            write_checkpoint(wire_path, wire)

    llm.client.chat.completions.create = capture
    try:
        try:
            run_episode(
                llm,
                protocol,
                domain="output_handling",
                task_id="output_sql_block",
                regime="dual",
                trial=1,
                output_dir=output,
            )
        except Exception as exc:
            wire.update(error_type=type(exc).__name__, error_message=str(exc)[:1200])
    finally:
        llm.client.chat.completions.create = create
        write_checkpoint(wire_path, wire)
    paths = list(output.glob("*.json"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one retry artifact")
    item = read_checkpoint(paths[0], SQL_RETRY_METADATA)
    return {
        "scope": "One unfinished-row retry, not a full matrix score",
        "status": item["status"],
        "score": item["score"],
        "source_artifact_hash": SQL_RETRY_SOURCE_HASH,
        "artifact_hash": item["artifact_hash"],
    }


if __name__ == "__main__":
    sql_retry.run(kbench.llm)
