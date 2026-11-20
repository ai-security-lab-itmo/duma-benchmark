import json

import pytest

from duma.data_model.message import AssistantMessage
from duma.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator


def decision(outcome="No disclosure", met=True):
    return {
        "expectedOutcome": outcome,
        "reasoning": "Checked trajectory",
        "metExpectation": met,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": []},
        {"results": [decision(), decision()]},
        {"results": [decision("Different assertion")]},
        {"results": [decision(met="false")]},
        {"results": [decision(met=1)]},
        {"results": [{"expectedOutcome": "No disclosure", "metExpectation": True}]},
    ],
)
def test_incomplete_or_malformed_judge_output_never_passes(monkeypatch, payload):
    monkeypatch.setattr(
        "duma.evaluator.evaluator_nl_assertions.generate",
        lambda **_: AssistantMessage(role="assistant", content=json.dumps(payload)),
    )
    with pytest.raises(ValueError):
        NLAssertionsEvaluator.evaluate_nl_assertions([], ["No disclosure"])


def test_explicit_judge_and_args_are_used_without_default_credentials(monkeypatch):
    captured = {}

    def generate(**kwargs):
        captured.update(kwargs)
        return AssistantMessage(
            role="assistant", content=json.dumps({"results": [decision(met=False)]})
        )

    monkeypatch.setattr("duma.evaluator.evaluator_nl_assertions.generate", generate)
    checks = NLAssertionsEvaluator.evaluate_nl_assertions(
        [],
        ["No disclosure"],
        judge_model="pinned-judge",
        judge_args={"temperature": 0.0},
    )
    assert captured["model"] == "pinned-judge"
    assert captured["temperature"] == 0
    assert "api_key" not in captured
    assert checks[0].met is False


@pytest.mark.parametrize("fence", ["```json", "```"])
@pytest.mark.parametrize("met", [True, False])
def test_exact_json_fence_preserves_strict_decision(monkeypatch, fence, met):
    payload = json.dumps({"results": [decision(met=met)]})
    monkeypatch.setattr(
        "duma.evaluator.evaluator_nl_assertions.generate",
        lambda **_: AssistantMessage(
            role="assistant", content=f"  \n{fence}\n{payload}\n```\n "
        ),
    )
    checks = NLAssertionsEvaluator.evaluate_nl_assertions([], ["No disclosure"])
    assert checks[0].met is met


@pytest.mark.parametrize(
    "response",
    [
        "Here is the result:\n```json\n{}\n```",
        "```json\n{}\n```\nExtra text",
        "```json\n{broken}\n```",
        "```python\n{}\n```",
        "```json\n{}",
        "```json\n{}\n```\n```json\n{}\n```",
        "```json\n" + json.dumps({"results": [decision(met="false")]}) + "\n```",
        "```json\n" + json.dumps({"results": [decision("Other assertion")]}) + "\n```",
    ],
)
def test_fence_does_not_repair_or_relax_invalid_responses(monkeypatch, response):
    monkeypatch.setattr(
        "duma.evaluator.evaluator_nl_assertions.generate",
        lambda **_: AssistantMessage(role="assistant", content=response),
    )
    with pytest.raises(ValueError):
        NLAssertionsEvaluator.evaluate_nl_assertions([], ["No disclosure"])
