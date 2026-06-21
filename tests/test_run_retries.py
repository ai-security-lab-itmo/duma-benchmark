"""Tests for the run-level retry net (_run_with_retries)."""

from types import SimpleNamespace

import pytest

from duma import run as run_module
from duma.data_model.simulation import RunStatus, TerminationReason
from duma.data_model.tasks import Task, UserScenario
from duma.run import _make_errored_simulation, _run_with_retries


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(run_module.time, "sleep", lambda *_: None)


def _sim(status):
    return SimpleNamespace(run_status=status)


def test_no_retry_when_first_attempt_succeeds():
    calls = {"n": 0}

    def run_once():
        calls["n"] += 1
        return _sim(RunStatus.EVALUATED)

    result = _run_with_retries(run_once, max_retries=2)
    assert result.run_status == RunStatus.EVALUATED
    assert calls["n"] == 1


def test_retries_until_non_errored():
    statuses = [RunStatus.ERRORED, RunStatus.ERRORED, RunStatus.EVALUATED]
    calls = {"n": 0}

    def run_once():
        s = statuses[calls["n"]]
        calls["n"] += 1
        return _sim(s)

    result = _run_with_retries(run_once, max_retries=3)
    assert result.run_status == RunStatus.EVALUATED
    assert calls["n"] == 3


def test_returns_errored_after_exhausting_retries():
    calls = {"n": 0}

    def run_once():
        calls["n"] += 1
        return _sim(RunStatus.ERRORED)

    result = _run_with_retries(run_once, max_retries=2)
    assert result.run_status == RunStatus.ERRORED
    assert calls["n"] == 3  # initial + 2 retries


def test_reraises_after_all_attempts_raise():
    calls = {"n": 0}

    def run_once():
        calls["n"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _run_with_retries(run_once, max_retries=2)
    assert calls["n"] == 3


def test_recovers_after_transient_exception():
    calls = {"n": 0}

    def run_once():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return _sim(RunStatus.EVALUATED)

    result = _run_with_retries(run_once, max_retries=2)
    assert result.run_status == RunStatus.EVALUATED
    assert calls["n"] == 2


def test_make_errored_simulation():
    task = Task(id="t1", user_scenario=UserScenario(instructions="x"))
    sim = _make_errored_simulation(task, trial=2, seed=7, error=RuntimeError("boom"))
    assert sim.run_status == RunStatus.ERRORED
    assert sim.termination_reason == TerminationReason.TOO_MANY_ERRORS
    assert sim.reward_info.reward == 0.0
    assert sim.task_id == "t1"
    assert sim.trial == 2
    assert sim.seed == 7
    assert "boom" in sim.reward_info.info["note"]
    # Must be excluded from metrics like any other infra crash.
    from duma.metrics.agent_metrics import EXCLUDED_RUN_STATUSES

    assert sim.run_status.value in EXCLUDED_RUN_STATUSES
