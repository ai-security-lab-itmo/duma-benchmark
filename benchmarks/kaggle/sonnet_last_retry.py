"""One remaining unfinished row, unchanged transport-v2 and original scoring."""

import json
from pathlib import Path

import kaggle_benchmarks as kbench

from duma.integrations.kaggle import runner
from duma.integrations.kaggle.artifacts import (
    checksum,
    read_checkpoint,
    write_checkpoint,
)


@kbench.task(name="duma-sonnet55-last-row-retry-v1")
def last_row_retry(llm) -> dict:
    if str(getattr(llm, "model", llm.name)) != "anthropic/claude-sonnet-5-5@default":
        return {"skipped": "Target-only single unfinished row; no inference"}
    if (
        checksum(LAST_SNAPSHOT) != LAST_SOURCE_HASH
        or LAST_SNAPSHOT["status"] != "technical_error"
        or LAST_SNAPSHOT["score"] is not None
    ):
        raise ValueError("Changed unfinished source row")
    meta = LAST_SNAPSHOT["metadata"]
    if (meta["domain"], meta["task_id"], meta["trial"], meta["seed"]) != (
        "collab",
        "collab_poisoning_token_exfil",
        1,
        301,
    ):
        raise ValueError("Unexpected retry row")
    protocol = SonnetProtocol.model_validate(meta["protocol"])
    if protocol.fingerprint != meta["protocol_hash"]:
        raise ValueError("Changed profile")
    kbench.config.interactive_mode = False
    original = runner.KaggleBackend
    create = llm.client.chat.completions.create
    wire = {"source_artifact_hash": LAST_SOURCE_HASH, "calls": []}
    output = Path("/kaggle/working/duma-sonnet-last-retry")
    wire_path = Path("/kaggle/working/sonnet-last-retry-wire.json")

    def observed_create(*args, **kwargs):
        raw = create(*args, **kwargs)
        if kwargs.get("model") == llm.model:
            wire["calls"].append(raw.model_dump(mode="json"))
            write_checkpoint(wire_path, wire)
        return raw

    runner.KaggleBackend = SonnetBackend
    llm.client.chat.completions.create = observed_create
    try:
        try:
            runner.run_episode(
                llm,
                protocol,
                domain="collab",
                task_id="collab_poisoning_token_exfil",
                regime="dual",
                trial=1,
                output_dir=output,
            )
        except Exception as exc:
            wire.update(error_type=type(exc).__name__, error_message=str(exc)[:1200])
    finally:
        runner.KaggleBackend = original
        llm.client.chat.completions.create = create
        write_checkpoint(wire_path, wire)
    paths = list(output.glob("*.json"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one retry artifact")
    item = read_checkpoint(paths[0], meta)
    print(
        json.dumps(
            {
                "status": item["status"],
                "score": item["score"],
                "artifact_hash": item["artifact_hash"],
            }
        ),
        flush=True,
    )
    return {
        "scope": "One new attempt, not a full matrix score",
        "status": item["status"],
        "score": item["score"],
    }


if __name__ == "__main__":
    last_row_retry.run(kbench.llm)
