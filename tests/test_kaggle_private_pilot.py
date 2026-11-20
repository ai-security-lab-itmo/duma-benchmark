from contextlib import nullcontext
from enum import Enum
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = runpy.run_path(str(ROOT / "benchmarks/kaggle/private_pilot.py"))
State = Enum("State", ["RUNNING", "BENCHMARK_TASK_RUN_STATE_COMPLETED", "ERRORED"])


def task(*, public=False, version=1, dataset="expected", state=State.RUNNING):
    return SimpleNamespace(
        is_public=public,
        creation_state=state,
        to_dict=lambda: {
            "slug": {"versionNumber": version},
            "options": {"datasetDataSources": [dataset]},
        },
    )


@pytest.mark.parametrize(
    "value", [task(public=True), task(version=2), task(dataset="wrong")]
)
def test_workflow_rejects_changed_visibility_versions_or_inputs(value):
    with pytest.raises(ValueError):
        WORKFLOW["require_private"](value, "expected")


def test_existing_task_is_not_reuploaded(monkeypatch):
    ensure = WORKFLOW["ensure_task"]
    monkeypatch.setitem(ensure.__globals__, "get_task", lambda *_: (task(), []))
    monkeypatch.setitem(
        ensure.__globals__, "cli", lambda *_: pytest.fail("Duplicate upload")
    )
    ensure(object(), "existing", "unused", "expected")


def fake_api():
    return SimpleNamespace(
        _normalize_model_slug=lambda value: value,
        _PENDING_CREATION_STATES={State.RUNNING},
        _TERMINAL_RUN_STATES={State.BENCHMARK_TASK_RUN_STATE_COMPLETED, State.ERRORED},
    )


@pytest.mark.parametrize(
    "state", [State.RUNNING, State.BENCHMARK_TASK_RUN_STATE_COMPLETED, State.ERRORED]
)
def test_existing_model_run_is_never_rescheduled(monkeypatch, state):
    ensure = WORKFLOW["ensure_model"]
    run = SimpleNamespace(model_version_slug="target", state=state)
    monkeypatch.setitem(ensure.__globals__, "get_task", lambda *_: (task(), [run]))
    monkeypatch.setitem(
        ensure.__globals__, "cli", lambda *_: pytest.fail("Duplicate model run")
    )
    ensure(fake_api(), "existing", "target", "expected")


def test_read_only_wait_completes_or_times_out_without_scheduling(monkeypatch):
    wait = WORKFLOW["wait_runs"]
    api = fake_api()
    run = SimpleNamespace(
        model_version_slug="target",
        state=State.BENCHMARK_TASK_RUN_STATE_COMPLETED,
        id=1,
    )
    monkeypatch.setitem(wait.__globals__, "get_task", lambda *_: (task(), [run]))
    wait(api, ["existing"], "target", 1)
    run.state = State.RUNNING
    with pytest.raises(TimeoutError):
        wait(api, ["existing"], "target", 0)


def test_multiple_attempts_are_not_auto_selected(monkeypatch):
    wait = WORKFLOW["wait_runs"]
    run = SimpleNamespace(model_version_slug="target")
    monkeypatch.setitem(wait.__globals__, "get_task", lambda *_: (task(), [run, run]))
    with pytest.raises(ValueError, match="Multiple attempts"):
        wait(fake_api(), ["existing"], "target", 0)


def test_budget_must_be_known_for_both_periods():
    balances = {
        "quotaBalances": [
            {"refillPeriod": "DAILY", "totalQuotaAllowed": 10, "quotaUsed": 1}
        ]
    }
    response = SimpleNamespace(to_dict=lambda: balances)
    client = SimpleNamespace(
        models=SimpleNamespace(
            model_proxy_api_client=SimpleNamespace(
                get_model_proxy_quotas=lambda: response
            )
        )
    )
    api = SimpleNamespace(build_kaggle_client=lambda: nullcontext(client))
    with pytest.raises(RuntimeError, match="unknown free quota"):
        WORKFLOW["require_budget"](api, 0.25)
    balances["quotaBalances"].append(
        {"refillPeriod": "MONTHLY", "totalQuotaAllowed": 100, "quotaUsed": 1}
    )
    WORKFLOW["require_budget"](api, 0.25)
