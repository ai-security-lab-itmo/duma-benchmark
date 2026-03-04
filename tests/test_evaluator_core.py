from types import SimpleNamespace

import pytest

from duma.data_model.simulation import RewardInfo, TerminationReason
from duma.data_model.tasks import EvaluationCriteria, RewardType, Task, UserScenario
from duma.evaluator.evaluator import EvaluationType, evaluate_simulation


def _task_with_basis(basis: list[RewardType]) -> Task:
    return Task(
        id="t1",
        user_scenario=UserScenario(instructions="Test scenario"),
        evaluation_criteria=EvaluationCriteria(reward_basis=basis),
    )


def _sim(termination_reason=TerminationReason.AGENT_STOP):
    return SimpleNamespace(termination_reason=termination_reason, messages=[])


def test_evaluate_simulation_returns_zero_on_premature_termination():
    reward_info = evaluate_simulation(
        simulation=_sim(TerminationReason.MAX_STEPS),
        task=_task_with_basis([RewardType.DB]),
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain="collab",
    )
    assert reward_info.reward == 0.0


def test_evaluate_simulation_combines_selected_reward_components(monkeypatch):
    monkeypatch.setattr(
        "duma.evaluator.evaluator.EnvironmentEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(
            reward=0.5, reward_breakdown={RewardType.DB: 0.5}, db_check=None
        ),
    )
    monkeypatch.setattr(
        "duma.evaluator.evaluator.ActionEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(
            reward=0.8, reward_breakdown={RewardType.ACTION: 0.8}
        ),
    )
    monkeypatch.setattr(
        "duma.evaluator.evaluator.CommunicateEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(
            reward=0.7, reward_breakdown={RewardType.COMMUNICATE: 0.7}
        ),
    )
    task = _task_with_basis(
        [RewardType.DB, RewardType.ACTION, RewardType.COMMUNICATE]
    )
    reward_info = evaluate_simulation(
        simulation=_sim(),
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain="collab",
    )
    assert reward_info.reward == pytest.approx(0.5 * 0.8 * 0.7)
    assert reward_info.reward_breakdown[RewardType.DB] == 0.5
    assert reward_info.reward_breakdown[RewardType.ACTION] == 0.8
    assert reward_info.reward_breakdown[RewardType.COMMUNICATE] == 0.7


def test_evaluate_simulation_raises_when_nl_basis_not_enabled(monkeypatch):
    monkeypatch.setattr(
        "duma.evaluator.evaluator.EnvironmentEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(reward=1.0, reward_breakdown={RewardType.DB: 1.0}),
    )
    monkeypatch.setattr(
        "duma.evaluator.evaluator.ActionEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(reward=1.0, reward_breakdown={RewardType.ACTION: 1.0}),
    )
    monkeypatch.setattr(
        "duma.evaluator.evaluator.CommunicateEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(
            reward=1.0, reward_breakdown={RewardType.COMMUNICATE: 1.0}
        ),
    )
    task = _task_with_basis([RewardType.NL_ASSERTION])
    with pytest.raises(ValueError, match="NL assertions are part of the reward basis"):
        evaluate_simulation(
            simulation=_sim(),
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain="collab",
        )


def test_evaluate_simulation_all_with_nl_includes_nl_reward(monkeypatch):
    monkeypatch.setattr(
        "duma.evaluator.evaluator.EnvironmentEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(reward=0.5, reward_breakdown={RewardType.DB: 0.5}),
    )
    monkeypatch.setattr(
        "duma.evaluator.evaluator.ActionEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(reward=1.0, reward_breakdown={RewardType.ACTION: 1.0}),
    )
    monkeypatch.setattr(
        "duma.evaluator.evaluator.CommunicateEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(
            reward=1.0, reward_breakdown={RewardType.COMMUNICATE: 1.0}
        ),
    )
    monkeypatch.setattr(
        "duma.evaluator.evaluator.NLAssertionsEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(
            reward=0.4,
            reward_breakdown={RewardType.NL_ASSERTION: 0.4},
            nl_assertions=[],
        ),
    )
    task = _task_with_basis([RewardType.DB, RewardType.NL_ASSERTION])
    reward_info = evaluate_simulation(
        simulation=_sim(),
        task=task,
        evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
        solo_mode=False,
        domain="collab",
    )
    assert reward_info.reward == pytest.approx(0.2)
    assert reward_info.reward_breakdown[RewardType.DB] == 0.5
    assert reward_info.reward_breakdown[RewardType.NL_ASSERTION] == 0.4


def test_evaluate_simulation_all_with_nl_skips_nl_when_not_in_basis(monkeypatch):
    monkeypatch.setattr(
        "duma.evaluator.evaluator.EnvironmentEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(reward=1.0, reward_breakdown={RewardType.DB: 1.0}),
    )
    monkeypatch.setattr(
        "duma.evaluator.evaluator.ActionEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(reward=1.0, reward_breakdown={RewardType.ACTION: 1.0}),
    )
    monkeypatch.setattr(
        "duma.evaluator.evaluator.CommunicateEvaluator.calculate_reward",
        lambda **kwargs: RewardInfo(
            reward=1.0, reward_breakdown={RewardType.COMMUNICATE: 1.0}
        ),
    )

    def _should_not_be_called(**kwargs):
        raise AssertionError("NL evaluator should not be called when NL is not in basis")

    monkeypatch.setattr(
        "duma.evaluator.evaluator.NLAssertionsEvaluator.calculate_reward",
        _should_not_be_called,
    )

    task = _task_with_basis([RewardType.DB])
    reward_info = evaluate_simulation(
        simulation=_sim(),
        task=task,
        evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
        solo_mode=False,
        domain="collab",
    )
    assert reward_info.reward == pytest.approx(1.0)
    assert reward_info.reward_breakdown[RewardType.DB] == 1.0
