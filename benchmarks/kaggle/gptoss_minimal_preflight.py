"""Append to frozen 0.1.3 bootstrap. Explicit minimal-reasoning diagnostic."""

import json
import os
from pathlib import Path

import kaggle_benchmarks as kbench

from duma.integrations.kaggle.protocol import KaggleProtocol, load_protocol
from duma.integrations.kaggle.runner import run_episode
from duma.utils.utils import DATA_DIR


def diagnostic_protocol(base, reasoning="minimal"):
    if reasoning not in {"minimal", "low"}:
        raise ValueError("Unsupported diagnostic setting")
    if (
        base.fingerprint
        != "cc5dedcad4c423156f17d257129a77c846fb418680f23da7d97e00480e6e7325"
    ):
        raise ValueError("Expected frozen V4 base")
    data = base.model_dump()
    data["target_reasoning_overrides"]["openai/gpt-oss-120b"] = reasoning
    return KaggleProtocol.model_validate(data)


@kbench.task(name="duma-gptoss-minimal-preflight-v1")
def gptoss_preflight(llm) -> dict:
    if str(getattr(llm, "model", llm.name)) != "openai/gpt-oss-120b":
        return {"skipped": "Target-only diagnostic; no inference or score"}
    kbench.config.interactive_mode = False
    base = load_protocol(os.environ["DUMA_KAGGLE_PROTOCOL"])
    base.validate_data(DATA_DIR)
    reasoning = globals().get("GPTOSS_REASONING", "minimal")
    protocol = diagnostic_protocol(base, reasoning)
    output = Path("/kaggle/working/duma-kaggle-results")
    outcomes = []
    for domain, task, regime in (
        ("collab", "collab_poisoning_logs", "solo"),
        ("collab", "collab_poisoning_logs", "dual"),
        ("output_handling", "output_xss_block", "dual"),
    ):
        if reasoning == "low" and regime == "solo":
            continue  # Forced tool choice is already confirmed unsupported.
        row = {"domain": domain, "task_id": task, "regime": regime}
        try:
            run_episode(
                llm,
                protocol,
                domain=domain,
                task_id=task,
                regime=regime,
                trial=0,
                output_dir=output,
            )
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error_message=str(exc)[:1200])
            if getattr(exc, "status_code", None) in {401, 403, 404, 429}:
                outcomes.append(row)
                break
        outcomes.append(row)
        print(json.dumps(row), flush=True)
    return {
        "scope": f"Compatibility only, reasoning={reasoning}; low mode dual-only; different fingerprint from frozen V4, not a leaderboard score",
        "profile_hash": protocol.fingerprint,
        "outcomes": outcomes,
    }


if __name__ == "__main__":
    gptoss_preflight.run(kbench.llm)
