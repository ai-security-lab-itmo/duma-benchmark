"""Tests that infra-crashed / inert runs are excluded from the pass@k denominator."""

import pandas as pd

from duma.metrics.agent_metrics import get_metrics_df, get_tasks_pass_hat_k


class _FakeResults:
    """Minimal stand-in exposing the to_df() that the metrics layer consumes."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_df(self) -> pd.DataFrame:
        return self._df.copy()


def _row(task_id, reward, run_status, num_trials=3):
    return {
        "task_id": task_id,
        "reward": reward,
        "run_status": run_status,
        "info_num_trials": num_trials,
        "task_num_agent_actions": 1,
        "task_num_user_actions": 0,
        "task_num_actions": 1,
    }


def test_errored_runs_dropped_from_denominator():
    # Task A: 1 success, 1 genuine fail, 1 infra crash. The crash must NOT count as a
    # third (failing) trial: pass^1 should be 1/2 = 0.5, not 1/3.
    df = pd.DataFrame(
        [
            _row("A", 1.0, "evaluated"),
            _row("A", 0.0, "evaluated"),
            _row("A", 0.0, "errored"),
        ]
    )
    metrics_df, max_k = get_metrics_df(_FakeResults(df))
    assert len(metrics_df) == 2
    # max_k stays at the configured trial count (3); per-task depth handles the
    # shortfall, so pass^1's denominator for task A is the 2 survivors, not 3.
    assert max_k == 3

    pass_hat = get_tasks_pass_hat_k(_FakeResults(df))
    assert pass_hat.loc["A", "pass^1"] == 0.5


def test_invalid_runs_dropped():
    df = pd.DataFrame(
        [
            _row("A", 1.0, "evaluated"),
            _row("A", 1.0, "invalid"),  # inert "pass" — must be excluded
        ]
    )
    metrics_df, _ = get_metrics_df(_FakeResults(df))
    assert len(metrics_df) == 1
    assert metrics_df.iloc[0]["run_status"] == "evaluated"


def test_incomplete_runs_are_kept():
    # MAX_STEPS / behavioural loops are scored by policy and stay in the denominator.
    df = pd.DataFrame(
        [
            _row("A", 1.0, "evaluated"),
            _row("A", 0.0, "incomplete"),
        ]
    )
    metrics_df, _ = get_metrics_df(_FakeResults(df))
    assert len(metrics_df) == 2


def test_legacy_none_status_is_kept():
    df = pd.DataFrame(
        [
            _row("A", 1.0, None),
            _row("A", 0.0, None),
        ]
    )
    metrics_df, _ = get_metrics_df(_FakeResults(df))
    assert len(metrics_df) == 2


def test_per_task_passk_depth_not_globally_truncated():
    # Task A keeps all 4 trials (so pass^4 is defined); task B loses 3 to exclusion
    # (only 1 survivor). The shortfall on B must NOT truncate A's pass^2..4 to NaN.
    rows = [_row("A", 1.0, "evaluated", num_trials=4) for _ in range(4)]
    rows += [_row("B", 1.0, "evaluated", num_trials=4)]
    rows += [_row("B", 0.0, "errored", num_trials=4) for _ in range(3)]
    pass_hat = get_tasks_pass_hat_k(_FakeResults(pd.DataFrame(rows)))
    # A retains 4 trials -> pass^4 computed (==1.0 since all succeeded)
    assert pass_hat.loc["A", "pass^4"] == 1.0
    # B has only 1 survivor -> pass^2 undefined (NaN), not a crash
    assert pd.isna(pass_hat.loc["B", "pass^2"])
