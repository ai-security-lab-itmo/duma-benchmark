# Optional SDK imports must follow importorskip.
# ruff: noqa: E402
import json
from pathlib import Path

import pytest

kbench = pytest.importorskip("kaggle_benchmarks")

from kaggle_benchmarks.actors.llms import LLMResponse
from kaggle_benchmarks.actors.llms import GoogleGenAI
from kaggle_benchmarks.serializers.openai import OpenAICompletionSerializer

from duma.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from duma.environment.tool import as_tool
from duma.integrations.kaggle.backend import KaggleBackend
from duma.integrations.kaggle.protocol import KaggleProtocol, load_protocol
from duma.integrations.kaggle.runner import run_episode
from duma.runtime import get_tasks

ROOT = Path(__file__).parents[1]


class ScriptedLLM(kbench.LLMChat):
    def __init__(self, name, responses):
        super().__init__(name=name, support_temperature=True)
        self.responses = iter(responses)
        self.requests = []

    def invoke(self, messages, system=None, **kwargs):
        self.requests.append((list(messages), kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def tool_response(name, arguments=None):
    return LLMResponse(
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments or {})},
                "_thought_signature": b"signature",
            }
        ],
    )


@pytest.fixture
def protocol():
    data = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v3.json").model_dump()
    data["user"]["model"] = "fixed-user"
    data["judge"]["model"] = "fixed-judge"
    return KaggleProtocol.model_validate(data)


def services():
    return {
        "fixed-user": ScriptedLLM(
            "fixed-user",
            [
                LLMResponse(content="Please give a safe status update."),
                LLMResponse(content="###STOP###"),
            ],
        ),
        "fixed-judge": ScriptedLLM("fixed-judge", []),
    }


def test_one_step_tools_history_and_thought_signatures_are_preserved():
    executed = []

    def square(x: int) -> int:
        """Square a number."""
        executed.append(x)
        return x * x

    llm = ScriptedLLM(
        "target", [tool_response("square", {"x": 5}), LLMResponse(content="25")]
    )
    backend = KaggleBackend(llm, "target")
    history = [
        SystemMessage(role="system", content="Agent policy"),
        UserMessage(role="user", content="Square 5"),
    ]
    first = backend(messages=history, tools=[as_tool(square)])
    assert not executed  # Only DUMA is allowed to execute tools.
    backend(
        messages=history + [first, ToolMessage(role="tool", id="call-1", content="25")]
    )
    assert len(llm.requests) == 2
    messages = llm.requests[-1][0]
    assert (
        first.raw_data["tool_metadata"]["call-1"]["thought_signature"] == "c2lnbmF0dXJl"
    )
    replay_call = messages[-2].tool_calls[0]
    if not isinstance(replay_call, dict):
        assert replay_call.thought_signature == b"signature"
    wire = list(OpenAICompletionSerializer().dump_messages(messages))
    assert wire[-2]["tool_calls"][0]["id"] == wire[-1]["tool_call_id"] == "call-1"
    assert wire[-1]["role"] == "tool"


def test_histories_do_not_leak_between_service_roles():
    llm = ScriptedLLM(
        "shared-backbone", [LLMResponse(content="a"), LLMResponse(content="u")]
    )
    KaggleBackend(llm, "target")(messages=[UserMessage(role="user", content="public")])
    KaggleBackend(llm, "user")(
        messages=[SystemMessage(role="system", content="private persona")]
    )
    assert [m.content for m in llm.requests[0][0]] == ["public"]
    assert [m.content for m in llm.requests[1][0]] == ["private persona"]


@pytest.mark.parametrize("model", ["google/gemini-test", "anthropic/claude-haiku-test"])
def test_system_only_start_has_stable_empty_transport_turn(model):
    llm = ScriptedLLM(model, [LLMResponse(content="ready")])
    history = [SystemMessage(role="system", content="Unchanged solo task")]
    KaggleBackend(llm, "target")(messages=history)
    wire = list(OpenAICompletionSerializer().dump_messages(llm.requests[0][0]))
    assert wire == [
        {"role": "system", "content": "Unchanged solo task"},
        {"role": "user", "content": "\n"},
    ]
    assert len(history) == 1
    first = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="call-1", name="get_collab_note", arguments={})],
    )
    replay = ScriptedLLM(model, [LLMResponse(content="ready")])
    KaggleBackend(replay, "target")(
        messages=history
        + [first, ToolMessage(role="tool", id="call-1", content="note")]
    )
    replay_wire = list(
        OpenAICompletionSerializer().dump_messages(replay.requests[0][0])
    )
    assert replay_wire[:2] == wire
    assert [message["role"] for message in replay_wire] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]


@pytest.mark.parametrize(
    "model", ["google/gemini-test", "anthropic/claude-haiku-test", "openai/test"]
)
def test_normal_history_has_no_extra_transport_turn(model):
    llm = ScriptedLLM(model, [LLMResponse(content="ready")])
    history = [
        SystemMessage(role="system", content="Policy"),
        UserMessage(role="user", content="Request"),
    ]
    KaggleBackend(llm, "target")(messages=history)
    assert [message.content for message in llm.requests[0][0]] == ["Policy", "Request"]


def test_anthropic_dual_greeting_does_not_get_solo_prefix():
    llm = ScriptedLLM("anthropic/claude-haiku-test", [LLMResponse(content="ready")])
    history = [
        SystemMessage(role="system", content="Policy"),
        AssistantMessage(role="assistant", content="Hello"),
        UserMessage(role="user", content="Request"),
    ]
    KaggleBackend(llm, "target")(messages=history)
    assert [message.content for message in llm.requests[0][0]] == [
        "Policy",
        "Hello",
        "Request",
    ]


def test_model_specific_reasoning_is_recorded_and_forwarded(tmp_path):
    data = load_protocol(ROOT / "benchmarks/kaggle/protocol.full.v4.json").model_dump()
    data["user"]["model"] = "fixed-user"
    data["judge"]["model"] = "fixed-judge"
    current = KaggleProtocol.model_validate(data)
    target = ScriptedLLM(
        "zai/glm-5", [LLMResponse(content="I cannot share raw logs.")]
    )
    run_episode(target, current, domain="collab", task_id="collab_poisoning_logs",
                regime="dual", trial=0, output_dir=tmp_path, service_models=services())
    row = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert row["metadata"]["target_generation"] == current.target_generation_args("zai/glm-5")
    assert target.requests[0][1]["reasoning"] == "minimal"


def test_native_genai_preserves_tool_carriers_and_translates_limits(monkeypatch):
    llm = GoogleGenAI(client=None, model="offline-genai", support_temperature=True)
    responses = iter([tool_response("square", {"x": 5}), LLMResponse(content="25")])
    captured = []

    def call_api(contents, config):
        captured.append((contents, config))
        return next(responses)

    monkeypatch.setattr(llm, "_call_api", call_api)

    def square(x: int) -> int:
        """Square a number."""
        return x * x

    backend = KaggleBackend(llm, "target")
    history = [UserMessage(role="user", content="Square 5")]
    first = backend(
        messages=history, tools=[as_tool(square)], tool_choice="required", max_tokens=77
    )
    assert captured[0][1].max_output_tokens == 77
    assert captured[0][1].tool_config.function_calling_config.mode == "ANY"
    backend(
        messages=history + [first, ToolMessage(role="tool", id="call-1", content="25")]
    )
    parts = [
        part
        for content in captured[1][0]
        for part in content.parts
        if part.function_call
    ]
    assert parts[0].function_call.name == "square"
    assert parts[0].thought_signature == b"signature"


@pytest.mark.parametrize("regime", ["dual", "solo"])
def test_real_duma_episode_executes_tools_and_replays_state_checks(
    protocol, tmp_path, regime
):
    ending = (
        tool_response("done", {"resolution_note": "Kept logs internal"})
        if regime == "solo"
        else LLMResponse(content="I will not share logs or delete audit trails.")
    )
    target = ScriptedLLM(
        "target",
        [
            tool_response(
                "request_security_approval",
                {"reason": "Raw incident evidence requested"},
            ),
            ending,
        ],
    )
    fixed = services()
    passed = run_episode(
        target,
        protocol,
        domain="collab",
        task_id="collab_poisoning_logs",
        regime=regime,
        trial=0,
        output_dir=tmp_path,
        service_models=fixed,
    )
    assert passed
    artifact = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert artifact["status"] == "completed"
    assert artifact["score"] == 1
    assert artifact["simulation"]["reward_info"]["env_assertions"]
    assert len(artifact["usage_by_role"]["target"]) == 2
    assert not fixed["fixed-judge"].requests
    user_prompt = "\n".join(
        str(m.content) for messages, _ in fixed["fixed-user"].requests for m in messages
    )
    assert "collab_note" not in user_prompt
    with pytest.raises(FileExistsError):
        run_episode(
            target,
            protocol,
            domain="collab",
            task_id="collab_poisoning_logs",
            regime=regime,
            trial=0,
            output_dir=tmp_path,
            service_models=fixed,
            resume=False,
        )


def test_policy_violation_is_scored_as_failure_not_api_error(protocol, tmp_path):
    target = ScriptedLLM(
        "unsafe-target",
        [tool_response("share_logs_with_customer"), LLMResponse(content="Sent logs.")],
    )
    assert not run_episode(
        target,
        protocol,
        domain="collab",
        task_id="collab_poisoning_logs",
        regime="dual",
        trial=0,
        output_dir=tmp_path,
        service_models=services(),
    )
    artifact = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert artifact["status"] == "completed"
    assert artifact["score"] == 0


def test_api_error_is_saved_and_raised_without_becoming_attack_success(
    protocol, tmp_path
):
    target = ScriptedLLM("broken-target", [RuntimeError("provider unavailable")])
    with pytest.raises(RuntimeError, match="Episode did not finish"):
        run_episode(
            target,
            protocol,
            domain="collab",
            task_id="collab_poisoning_logs",
            regime="dual",
            trial=0,
            output_dir=tmp_path,
            service_models=services(),
        )
    artifact = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert artifact["status"] == "technical_error"
    assert artifact["score"] is None
    assert artifact["error_type"] == "RuntimeError"
    assert artifact["simulation"]["termination_reason"] == "too_many_errors"


def test_missing_service_model_fails_before_target_inference(protocol, tmp_path):
    target = ScriptedLLM("target", [])
    with pytest.raises(ValueError, match="unavailable"):
        run_episode(
            target,
            protocol,
            domain="collab",
            task_id="collab_poisoning_logs",
            regime="dual",
            trial=0,
            output_dir=tmp_path,
            service_models={},
        )
    assert not target.requests
    assert not list(tmp_path.iterdir())


def test_communication_assertions_use_the_fixed_judge(protocol, tmp_path):
    expected = get_tasks("output_handling", task_ids=["output_xss_block"])[
        0
    ].evaluation_criteria.nl_assertions
    target = ScriptedLLM(
        "target",
        [
            LLMResponse(
                content="Per policy we do not forward raw payloads, only sanitized responses."
            )
        ],
    )
    fixed = services()
    fixed["fixed-judge"] = ScriptedLLM(
        "fixed-judge",
        [
            LLMResponse(
                content=json.dumps(
                    {
                        "results": [
                            {
                                "expectedOutcome": outcome,
                                "metExpectation": True,
                                "reasoning": "Safe refusal",
                            }
                            for outcome in expected
                        ]
                    }
                )
            )
        ],
    )
    assert run_episode(
        target,
        protocol,
        domain="output_handling",
        task_id="output_xss_block",
        regime="dual",
        trial=0,
        output_dir=tmp_path,
        service_models=fixed,
    )
    assert len(fixed["fixed-judge"].requests) == 1
    artifact = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert len(artifact["usage_by_role"]["judge"]) == 1


@pytest.mark.parametrize("response", ["not JSON", '{"results": []}'])
def test_judge_error_retains_unscored_trajectory_and_raw_response(
    protocol, tmp_path, response
):
    target = ScriptedLLM("target", [LLMResponse(content="Per policy, sanitized only.")])
    fixed = services()
    fixed["fixed-judge"] = ScriptedLLM("fixed-judge", [LLMResponse(content=response)])
    with pytest.raises(ValueError):
        run_episode(
            target,
            protocol,
            domain="output_handling",
            task_id="output_xss_block",
            regime="dual",
            trial=0,
            output_dir=tmp_path,
            service_models=fixed,
        )
    artifact = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert artifact["status"] == "technical_error"
    assert artifact["score"] is None
    assert artifact["simulation"]["reward_info"] is None
    assert artifact["simulation"]["termination_reason"] == "user_stop"
    assert any(
        message.get("content") == "Per policy, sanitized only."
        for message in artifact["simulation"]["messages"]
    )
    assert artifact["judge_response"]["content"] == response
    assert artifact["judge_response"]["raw_data"]["model"] == "fixed-judge"
    assert len(artifact["usage_by_role"]["judge"]) == 1
    from duma.integrations.kaggle.artifacts import read_checkpoint

    read_checkpoint(next(tmp_path.glob("*.json")), artifact["metadata"])


def test_pilot_entry_point_runs_as_a_real_sdk_task(protocol, tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "protocol.json"
    manifest.write_text(protocol.model_dump_json())
    monkeypatch.setenv("DUMA_KAGGLE_PROTOCOL", str(manifest))
    monkeypatch.setenv("DUMA_KAGGLE_OUTPUT", str(tmp_path / "episodes"))
    monkeypatch.setattr(kbench, "llms", services())
    spec = importlib.util.spec_from_file_location(
        "duma_kaggle_pilot", ROOT / "benchmarks/kaggle/pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = ScriptedLLM(
        "target",
        [
            tool_response(
                "request_security_approval", {"reason": "Sensitive incident review"}
            ),
            LLMResponse(content="I will not share logs."),
        ],
    )
    result = module.duma_pilot.run(target)
    assert result.result is True
    assert len(list((tmp_path / "episodes").glob("*.json"))) == 1


def test_resume_uses_verified_completed_episode_without_model_calls(protocol, tmp_path):
    target = ScriptedLLM(
        "target",
        [
            tool_response(
                "request_security_approval", {"reason": "Sensitive evidence"}
            ),
            LLMResponse(content="I will keep logs internal."),
        ],
    )
    fixed = services()
    kwargs = dict(
        domain="collab",
        task_id="collab_poisoning_logs",
        regime="dual",
        trial=0,
        output_dir=tmp_path,
        service_models=fixed,
    )
    assert run_episode(target, protocol, **kwargs)
    requests = len(target.requests)
    assert run_episode(target, protocol, **kwargs)
    assert len(target.requests) == requests
    path = next(tmp_path.glob("*.json"))
    body = json.loads(path.read_text())
    body["score"] = 0
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="integrity"):
        run_episode(target, protocol, **kwargs)
    assert len(target.requests) == requests
    assert not list(tmp_path.glob("*.lock"))
    assert not list(tmp_path.glob("*.tmp"))


def test_incomplete_episode_has_no_score_and_can_be_retried(protocol, tmp_path):
    short = protocol.model_copy(update={"max_steps": 1})
    kwargs = dict(
        domain="collab",
        task_id="collab_poisoning_logs",
        regime="dual",
        trial=0,
        output_dir=tmp_path,
        service_models=services(),
    )
    with pytest.raises(RuntimeError, match="did not finish"):
        run_episode(ScriptedLLM("target", []), short, **kwargs)
    artifact = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert artifact["status"] == "incomplete"
    assert artifact["score"] is None
    kwargs["service_models"] = services()
    with pytest.raises(RuntimeError):
        run_episode(ScriptedLLM("target", []), short, **kwargs)
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 1


def test_matrix_aggregates_and_resumes_without_new_calls(protocol, tmp_path):
    from duma.integrations.kaggle.runner import run_matrix

    manifest = protocol.model_copy(
        update={
            "tasks": {"collab": ["collab_poisoning_logs"]},
            "trial_seeds": [300, 301],
        }
    )
    target = ScriptedLLM(
        "target",
        [
            tool_response(
                "request_security_approval", {"reason": "Sensitive evidence"}
            ),
            LLMResponse(content="No logs shared."),
        ]
        * 2,
    )
    fixed = services()
    fixed["fixed-user"] = ScriptedLLM(
        "fixed-user",
        [
            LLMResponse(content=text)
            for text in [
                "Please resolve safely.",
                "###STOP###",
                "Please resolve safely.",
                "###STOP###",
            ]
        ],
    )
    summary = run_matrix(
        target, manifest, regime="dual", output_dir=tmp_path, service_models=fixed
    )
    assert summary["score"] == 1
    assert summary["completed"] == 2
    assert len(target.requests) == 4
    assert (
        run_matrix(
            target, manifest, regime="dual", output_dir=tmp_path, service_models=fixed
        )["score"]
        == 1
    )
    assert len(target.requests) == 4


def test_matrix_refuses_to_return_score_after_provider_error(protocol, tmp_path):
    from duma.integrations.kaggle.runner import run_matrix

    manifest = protocol.model_copy(
        update={"tasks": {"collab": ["collab_poisoning_logs"]}, "trial_seeds": [300]}
    )
    with pytest.raises(RuntimeError, match="no leaderboard score"):
        run_matrix(
            ScriptedLLM("target", [RuntimeError("Recorded denial")]),
            manifest,
            regime="dual",
            output_dir=tmp_path,
            service_models=services(),
        )
    report = json.loads(next((tmp_path / "summaries").glob("*.json")).read_text())
    assert report["score"] is None and not report["complete"]
