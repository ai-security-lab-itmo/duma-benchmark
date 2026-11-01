"""Native Gemini transport profile over the unchanged frozen DUMA runtime."""

import os
from pathlib import Path
from typing import Literal

import kaggle_benchmarks as kbench

from duma.integrations.kaggle.artifacts import write_checkpoint
from duma.integrations.kaggle.protocol import KaggleProtocol, load_protocol
from duma.integrations.kaggle.runner import run_episode, run_matrix


class Gemini25NativeProtocol(KaggleProtocol):
    transport_profile: Literal["duma-gemini25-native-v1"] = "duma-gemini25-native-v1"
    target_api: Literal["genai"] = "genai"
    target_model: Literal["google/gemini-2.5-flash"] = "google/gemini-2.5-flash"
    base_protocol_hash: str
    native_adapter_sha256: str


def native_protocol(base, adapter_hash):
    return Gemini25NativeProtocol(
        **base.model_dump(),
        base_protocol_hash=base.fingerprint,
        native_adapter_sha256=adapter_hash,
    )


@kbench.task(name="__GEMINI25_TASK_NAME__")
def gemini25_native(llm) -> dict:
    from kaggle_benchmarks.kaggle.model_proxy import ModelProxy

    model = str(getattr(llm, "model", llm.name))
    if model != "google/gemini-2.5-flash":
        return {"skipped": "Gemini 2.5 Flash only; no inference or security score"}
    kbench.config.interactive_mode = False
    base = load_protocol(os.environ["DUMA_KAGGLE_PROTOCOL"])
    protocol = native_protocol(base, GEMINI25_ADAPTER_SHA256)
    native = ModelProxy(model=model, api="genai")
    output = Path("/kaggle/working/duma-kaggle-results")
    wire = Path("/kaggle/working/gemini25-native-wire")
    wire.mkdir(exist_ok=True)
    create = native.client.models.generate_content
    call_index = 0
    fatal = None

    def capture(*args, **kwargs):
        nonlocal call_index, fatal
        if fatal is not None:
            raise RuntimeError(f"Native endpoint unavailable: {fatal}")
        if args or set(kwargs) != {"model", "contents", "config"}:
            raise ValueError("Unexpected native SDK request shape")
        call_index += 1
        event = {
            "profile_hash": protocol.fingerprint,
            "model": kwargs["model"],
            "contents": [c.model_dump(mode="json") for c in kwargs["contents"]],
            "config": kwargs["config"].model_dump(mode="json"),
        }
        try:
            response = create(**kwargs)
            event["response"] = response.model_dump(mode="json")
            return response
        except Exception as exc:
            event.update(error_type=type(exc).__name__, error_message=str(exc)[:2000])
            code = getattr(exc, "code", getattr(exc, "status_code", None))
            if code in {401, 403, 404, 429}:
                fatal = code
            raise
        finally:
            write_checkpoint(wire / f"{call_index:06d}.json", event)

    native.client.models.generate_content = capture
    try:
        if GEMINI25_PHASE in {"solo", "dual"}:
            summary = run_matrix(
                native, protocol, regime=GEMINI25_PHASE, output_dir=output
            )
            return {"score": summary["score"], "profile_hash": protocol.fingerprint}
        if GEMINI25_PHASE != "preflight":
            raise ValueError("Unexpected native evaluation phase")
        rows = [
            ("collab", "collab_poisoning_logs", "solo"),
            ("collab", "collab_poisoning_logs", "dual"),
            ("output_handling", "output_xss_block", "dual"),
        ]
        outcomes = []
        for domain, task, regime in rows:
            try:
                result = run_episode(
                    native,
                    protocol,
                    domain=domain,
                    task_id=task,
                    regime=regime,
                    trial=0,
                    output_dir=output,
                    resume=False,
                )
                outcomes.append(
                    {
                        "row": [domain, task, regime],
                        "completed": True,
                        "score": float(result),
                    }
                )
            except Exception as exc:
                outcomes.append(
                    {
                        "row": [domain, task, regime],
                        "completed": False,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:2000],
                    }
                )
            if fatal is not None:
                break
        report = {
            "profile": protocol.model_dump(),
            "profile_hash": protocol.fingerprint,
            "scope": "Three compatibility episodes, not a leaderboard score",
            "outcomes": outcomes,
        }
        write_checkpoint(Path("/kaggle/working/gemini25-native-preflight.json"), report)
        return report
    finally:
        native.client.models.generate_content = create


if __name__ == "__main__":
    gemini25_native.run(kbench.llm)
