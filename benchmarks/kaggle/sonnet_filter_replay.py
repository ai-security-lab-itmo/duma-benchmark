"""Exact failed-turn replay, not a replacement episode or benchmark score."""

import hashlib
import json
from pathlib import Path

import duma
import kaggle_benchmarks as kbench

from duma.data_model.message import SystemMessage
from duma.data_model.simulation import SimulationRun
from duma.integrations.kaggle.artifacts import checksum, write_checkpoint
from duma.integrations.kaggle.protocol import fingerprint_files
from duma.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from duma.registry import registry
from duma.runtime import get_tasks
from duma.utils.utils import DATA_DIR


def replay_messages(sim, task):
    if task.initial_state and task.initial_state.message_history:
        raise ValueError("This diagnostic supports only fresh dual episodes")
    history = list(sim.messages)
    if (
        not history
        or history[0].role != "assistant"
        or history[0].content != DEFAULT_FIRST_AGENT_MESSAGE.content
        or history[0].tool_calls
    ):
        raise ValueError("Unexpected initial greeting")
    # The orchestrator emits this to the user, never to fresh agent state.
    history = history[1:]
    if any(
        m.role == "tool" and getattr(m, "requestor", "assistant") != "assistant"
        for m in history
    ):
        raise ValueError("User tools require perspective projection")
    pending = set()
    for message in history:
        if message.role == "assistant":
            pending.update(c.id for c in message.tool_calls or [])
        elif message.role == "tool":
            if message.id not in pending:
                raise ValueError("Unmatched tool result")
            pending.remove(message.id)
    if pending or not history or history[-1].role not in {"user", "tool"}:
        raise ValueError("Not an exact failed agent turn")
    return [SystemMessage(role="system", content=sim.assistant_system_prompt)] + history


def is_filtered(raw):
    choices = raw.get("choices") or []
    return (
        len(choices) == 1
        and choices[0].get("finish_reason") == "content_filter"
        and choices[0].get("message") is None
    )


@kbench.task(name="duma-sonnet55-filter-replay-v1")
def filter_replay(llm) -> dict:
    if str(getattr(llm, "model", llm.name)) != "anthropic/claude-sonnet-5-5@default":
        return {"skipped": "Target-only replay; no inference or security score"}
    kbench.config.interactive_mode = False
    if len(FILTER_SNAPSHOTS) != 16 or kbench.__version__ != "0.6.2":
        raise ValueError("Unexpected selection or SDK")
    source = fingerprint_files(Path(duma.__file__).parent, ("**/*.py",))
    data = fingerprint_files(
        DATA_DIR / "duma", ("domains/**/*.*", "user_simulator/*.md")
    )
    prepared, seen = [], set()
    for snapshot in FILTER_SNAPSHOTS:
        if (
            checksum(snapshot) != snapshot["artifact_hash"]
            or snapshot["artifact_hash"] in seen
        ):
            raise ValueError("Changed or duplicate snapshot")
        seen.add(snapshot["artifact_hash"])
        meta = snapshot["metadata"]
        if (
            meta["source_hash"],
            meta["data_hash"],
            meta["target_model"],
            meta["regime"],
        ) != (source, data, llm.model, "dual"):
            raise ValueError("Snapshot/runtime identity mismatch")
        if snapshot["status"] != "technical_error" or snapshot["score"] is not None:
            raise ValueError("Cannot resample completed episode")
        task = get_tasks(meta["domain"], [meta["task_id"]])[0]
        sim = SimulationRun.model_validate(snapshot["simulation"])
        prepared.append((snapshot, replay_messages(sim, task)))
    create = llm.client.chat.completions.create
    current = {}

    def observed_create(*args, **kwargs):
        current["request"] = {k: v for k, v in kwargs.items() if k != "messages"}
        current["request_sha256"] = hashlib.sha256(
            json.dumps(kwargs, sort_keys=True).encode()
        ).hexdigest()
        response = create(*args, **kwargs)
        current["raw_response"] = response.model_dump(mode="json")
        return response

    report = {
        "version": "duma-sonnet-exact-filter-replay-v1",
        "scope": "Exact failed-turn replays only; historical cause not retroactively proven; no tools executed or security score",
        "cases": [],
    }
    output = Path("/kaggle/working/sonnet-filter-replay.json")
    llm.client.chat.completions.create = observed_create
    try:
        for snapshot, messages in prepared:
            meta = snapshot["metadata"]
            current = {
                "source_artifact_hash": snapshot["artifact_hash"],
                "task_id": meta["task_id"],
                "domain": meta["domain"],
                "seed": meta["seed"],
            }
            params = meta["protocol"]["agent"].copy()
            params.pop("model", None)
            params.pop("reasoning", None)
            params["seed"] = meta["seed"]
            try:
                response = SonnetBackend(llm, "target")(
                    messages=messages,
                    tools=registry.get_env_constructor(meta["domain"])().get_tools(),
                    **params,
                )
                current["parsed_response"] = response.model_dump(mode="json")
            except Exception as exc:
                current.update(
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:1200],
                    http_status=getattr(exc, "status_code", None),
                )
            current["outcome"] = (
                "provider_filtered"
                if is_filtered(current.get("raw_response", {}))
                else "unresolved"
            )
            report["cases"].append(current)
            write_checkpoint(output, report)
            print(
                json.dumps(
                    {
                        k: current.get(k)
                        for k in ("task_id", "seed", "outcome", "http_status")
                    }
                ),
                flush=True,
            )
            if current.get("http_status") in {401, 403, 404, 429}:
                break
    finally:
        llm.client.chat.completions.create = create
    return {
        "scope": report["scope"],
        "attempted": len(report["cases"]),
        "filtered": sum(c["outcome"] == "provider_filtered" for c in report["cases"]),
    }


if __name__ == "__main__":
    filter_replay.run(kbench.llm)
