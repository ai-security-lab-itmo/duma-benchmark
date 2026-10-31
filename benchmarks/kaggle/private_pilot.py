"""Run the already-approved private V3 preflights/pilot, never publish.

Local foreground workflow, not an automation or installed Kaggle dependency.
Existing tasks/runs are adopted, not recreated. Technical failures stop the flow.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
GEMINI = "gemini-3.8-flash"
OPENAI = "gpt-5.4-mini-2026-03-17"
TARGETS = {GEMINI: "google/" + GEMINI, OPENAI: "openai/" + OPENAI}
FULL_DATASET = "ivanblch/duma-bench-private-v3-runtime"
PILOT_DATASET = "ivanblch/duma-bench-private-v3-pilot-runtime"


def cli(*args):
    subprocess.run(
        [str(ROOT / ".venv/bin/kaggle"), "benchmarks", "tasks", *args],
        cwd=ROOT,
        check=True,
        timeout=180,
    )


def get_task(api, name):
    with api.build_kaggle_client() as client:
        task = api._get_benchmark_task(name, client, allow_not_found=True)
        runs = api._fetch_task_runs(client, name) if task else []
    return task, runs


def require_private(task, dataset):
    info = task.to_dict()
    if task.is_public or info["slug"]["versionNumber"] != 1:
        raise ValueError("Expected a private frozen version-1 task")
    if info.get("options", {}).get("datasetDataSources") != [dataset]:
        raise ValueError("Unexpected attached Dataset")


def ensure_task(api, name, wrapper, dataset):
    task, _ = get_task(api, name)
    if task is None:
        cli("push", name, "-f", str(ROOT / wrapper / "hosted_pilot.py"), "-d", dataset)
        task, _ = get_task(api, name)
    require_private(task, dataset)


def require_budget(api, minimum):
    with api.build_kaggle_client() as client:
        balances = (
            client.models.model_proxy_api_client.get_model_proxy_quotas()
            .to_dict()
            .get("quotaBalances", [])
        )
    remaining = {}
    for item in balances:
        if (
            "quotaUsed" in item
            and "totalQuotaAllowed" in item
            and all(
                math.isfinite(item[key]) for key in ("quotaUsed", "totalQuotaAllowed")
            )
        ):
            remaining[item["refillPeriod"]] = (
                item["totalQuotaAllowed"] - item["quotaUsed"]
            )
    if any(remaining.get(period, -1) < minimum for period in ("DAILY", "MONTHLY")):
        raise RuntimeError("Insufficient or unknown free quota; no new run scheduled")
    print("Free quota checked:", json.dumps(remaining), flush=True)


def wait_runs(api, names, model, deadline_seconds):
    deadline = time.monotonic() + deadline_seconds
    last = None
    interval = 30
    while True:
        states = {}
        all_done = True
        for name in names:
            task, runs = get_task(api, name)
            matching = [
                run
                for run in runs
                if api._normalize_model_slug(run.model_version_slug) == model
            ]
            if len(matching) > 1:
                raise ValueError(
                    "Multiple attempts require explicit audit; not auto-selecting one"
                )
            if matching:
                run = matching[0]
                states[name] = {"id": run.id, "state": run.state.name}
                if run.state in api._TERMINAL_RUN_STATES:
                    if not run.state.name.endswith("_COMPLETED"):
                        raise RuntimeError(
                            f"Technical run failure: {name}: {run.to_dict()}"
                        )
                else:
                    all_done = False
            else:
                states[name] = {"state": task.creation_state.name}
                if task.creation_state not in api._PENDING_CREATION_STATES:
                    raise RuntimeError(
                        "Task has no expected model run; inspect before scheduling"
                    )
                all_done = False
        if states != last:
            print(json.dumps(states), flush=True)
            last = states
        if all_done:
            return
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError(
                "Private validation still pending; no duplicate/retry scheduled"
            )
        time.sleep(min(interval, left))
        interval = min(interval * 2, 180)


def ensure_model(api, name, model, dataset):
    task, runs = get_task(api, name)
    require_private(task, dataset)
    if any(api._normalize_model_slug(run.model_version_slug) == model for run in runs):
        return
    if task.creation_state != api._TASK_CREATION_COMPLETED:
        raise RuntimeError(
            "Default validation is not complete; not scheduling another target"
        )
    cli("run", name, "-m", model)


def verify(names, directory, *, pilot=False, targets=(GEMINI,), episode=None):
    directory = ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        cli("download", name, "-s", "-o", str(directory))
    report = directory / ("verified-" + "-".join(targets) + ".json")
    # Always recompute audits from downloaded inputs; keep prior reports immutable.
    suffix = 1
    while report.exists():
        report = directory / f"verified-recheck-{suffix}.json"
        suffix += 1
    args = [
        str(ROOT / ".cache/kaggle-v3-minimal-env/bin/python"),
        str(ROOT / "benchmarks/kaggle/verify_results.py"),
        "--results",
        str(directory),
        "--bundle",
        str(
            ROOT
            / (
                ".cache/kaggle-v3-pilot-solo/duma-pilot.zip"
                if pilot
                else ".cache/kaggle-v3-release-readiness/duma-pilot.zip"
            )
        ),
        "--protocol",
        str(
            ROOT
            / (
                "benchmarks/kaggle/protocol.pilot.v6.json"
                if pilot
                else "benchmarks/kaggle/protocol.full.v3.json"
            )
        ),
        "--output",
        str(report),
    ]
    for target in targets:
        args += ["--target", TARGETS[target]]
    if episode:
        args += ["--episode", *episode]
    subprocess.run(
        args,
        cwd=ROOT,
        env={
            **os.environ,
            "DUMA_DATA_DIR": str(ROOT / ".cache/kaggle-v3-release-audit-assets/data"),
        },
        check=True,
        timeout=180,
    )
    print("Verified report:", report, flush=True)


def main(timeout):
    if timeout <= 0:
        raise ValueError("Wait timeout must be positive")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    require_budget(api, 0.25)
    cases = {
        "dual": ("collab", "collab_poisoning_logs", "dual"),
        "solo": ("collab", "collab_poisoning_logs", "solo"),
        "communication": ("output_handling", "output_xss_block", "dual"),
    }
    for case in cases:
        name = "duma-v3-preflight-" + case
        ensure_task(
            api, name, ".cache/kaggle-v3-release-preflight-" + case, FULL_DATASET
        )
        wait_runs(api, [name], GEMINI, timeout)
        verify(
            [name], ".cache/kaggle-v3-preflight-results/" + case, episode=cases[case]
        )
    require_budget(api, 1.0)
    names = ["duma-v3-pilot-" + regime for regime in ("solo", "dual")]
    for regime, name in zip(("solo", "dual"), names, strict=True):
        ensure_task(api, name, ".cache/kaggle-v3-pilot-" + regime, PILOT_DATASET)
    wait_runs(api, names, GEMINI, timeout)
    verify(names, ".cache/kaggle-v3-pilot-results", pilot=True)
    require_budget(api, 1.0)
    for name in names:
        ensure_model(api, name, OPENAI, PILOT_DATASET)
    wait_runs(api, names, OPENAI, timeout)
    verify(
        names, ".cache/kaggle-v3-pilot-results", pilot=True, targets=(GEMINI, OPENAI)
    )
    print(
        "PILOT AUDITED. Review trajectories before expanding to full-v3. No publication performed.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout", type=int, default=7200, help="Maximum wait per stage, seconds"
    )
    args = parser.parse_args()
    main(args.timeout)
