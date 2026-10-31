# %%
"""Private pilot entry point. Install DUMA and configure its data path first."""

import os
from pathlib import Path

import kaggle_benchmarks as kbench

from duma.integrations.kaggle.protocol import load_protocol
from duma.integrations.kaggle.runner import run_episode


# %%
@kbench.task(name="duma-pilot")
def duma_pilot(
    llm,
    domain: str = "collab",
    task_id: str = "collab_poisoning_logs",
    regime: str = "dual",
    trial: int = 0,
) -> bool:
    protocol_path = os.environ.get("DUMA_KAGGLE_PROTOCOL")
    if not protocol_path:
        raise ValueError("Set DUMA_KAGGLE_PROTOCOL to the pinned protocol JSON path")
    return run_episode(
        llm,
        load_protocol(protocol_path),
        domain=domain,
        task_id=task_id,
        regime=regime,
        trial=trial,
        output_dir=Path(os.environ.get("DUMA_KAGGLE_OUTPUT", "duma-kaggle-results")),
    )


# %%
if __name__ == "__main__":
    duma_pilot.run(kbench.llm)
