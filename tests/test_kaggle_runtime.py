import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from duma.utils.pydantic_utils import update_pydantic_model_with_dict
from duma.runtime import run_task
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]


def test_all_35_hardened_tasks_match_versioned_contract():
    expected = json.loads(
        (ROOT / "tests/fixtures/kaggle-task-contract.v3.json").read_text()
    )
    actual = runpy.run_path(str(ROOT / "tests/kaggle_reference.py"))["task_contract"]()
    assert len(actual["tasks"]) == 35
    assert json.loads(json.dumps(actual, default=str)) == expected


def test_recorded_trajectories_and_scores_match_original_runner():
    from duma.run import run_task as legacy_run_task

    replay = runpy.run_path(str(ROOT / "tests/kaggle_replay.py"))["replays"]
    expected = replay(legacy_run_task)
    actual = replay(run_task)
    assert json.loads(json.dumps(actual)) == expected


def test_lightweight_import_does_not_load_legacy_dependencies():
    code = """
import sys
import importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"litellm", "langfuse", "fastapi", "pandas", "rich", "deepdiff", "dotenv", "addict", "yaml", "toml", "redis"} or fullname == "duma.run":
            raise AssertionError("Forbidden runtime import: " + fullname)
sys.meta_path.insert(0, Block())
from duma.runtime import get_tasks
from duma.integrations.kaggle.runner import run_episode
assert len(get_tasks("collab")) == 6
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_recursive_updates_preserve_addict_merge_semantics():
    class Model(BaseModel):
        state: dict

    initial = Model(state={"nested": {"keep": 1, "replace": 2}, "items": [1, 2]})
    updated = update_pydantic_model_with_dict(
        initial, {"state": {"nested": {"replace": 3}, "items": [5]}}
    )
    assert updated.state == {"nested": {"keep": 1, "replace": 3}, "items": [5]}
    assert initial.state["nested"]["replace"] == 2


def test_user_side_tools_and_replay_are_preserved(monkeypatch):
    from duma.data_model.message import AssistantMessage, ToolCall
    from duma.data_model.tasks import EnvAssertion
    from duma.environment.db import DB
    from duma.environment.toolkit import ToolKitBase, ToolType, is_tool
    from duma.registry import registry
    from duma.utils.generation import use_generation_backends

    class UserDB(DB):
        enabled: bool = False

    class UserTools(ToolKitBase):
        @is_tool(ToolType.WRITE)
        def enable(self) -> str:
            """Enable the user's setting."""
            self.db.enabled = True
            return "enabled"

        def enabled(self) -> bool:
            return self.db.enabled

    original = registry.get_env_constructor("collab")

    def environment(solo_mode=False):
        env = original(solo_mode=solo_mode)
        env.user_tools = UserTools(UserDB())
        return env

    monkeypatch.setitem(registry._domains, "recorded_user_tools", environment)
    task = registry.get_tasks_loader("collab")()[0].model_copy(deep=True)
    task.evaluation_criteria.env_assertions.append(
        EnvAssertion(
            env_type="user", func_name="enabled", arguments={}, assert_value=True
        )
    )
    responses = iter(
        [
            AssistantMessage(
                role="assistant",
                tool_calls=[ToolCall(id="user-call", name="enable", arguments={})],
            ),
            AssistantMessage(role="assistant", content="Please resolve safely."),
            AssistantMessage(role="assistant", content="###STOP###"),
        ]
    )
    target_responses = iter(
        [
            lambda: AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="approval",
                        name="request_security_approval",
                        arguments={"reason": "Sensitive evidence"},
                    )
                ],
            ),
            lambda: AssistantMessage(role="assistant", content="Kept logs internal."),
        ]
    )
    with use_generation_backends(
        {
            "test-user": lambda **_: next(responses),
            "test-target": lambda **_: next(target_responses)(),
        }
    ):
        result = run_task(
            domain="recorded_user_tools",
            task=task,
            agent="llm_agent",
            user="user_simulator",
            llm_agent="test-target",
            llm_user="test-user",
            seed=300,
        )
    assert result.reward_info.reward == 1
    assert any(
        getattr(message, "requestor", None) == "user" for message in result.messages
    )


def test_solo_uses_policy_from_solo_environment(monkeypatch):
    from duma.registry import registry
    from duma.data_model.message import AssistantMessage, ToolCall
    from duma.utils.generation import use_generation_backends

    original = registry.get_env_constructor("collab")

    def environment(solo_mode=False):
        env = original(solo_mode=solo_mode)
        env.policy = "solo-specific policy" if solo_mode else "dual-specific policy"
        return env

    monkeypatch.setitem(registry._domains, "mode_policy", environment)
    with use_generation_backends(
        {
            "solo-backend": lambda **_: AssistantMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="done",
                        name="done",
                        arguments={"resolution_note": "No changes"},
                    )
                ],
            )
        }
    ):
        result = run_task(
            domain="mode_policy",
            task=registry.get_tasks_loader("collab")()[0],
            agent="llm_agent_solo",
            user="dummy_user",
            llm_agent="solo-backend",
        )
    assert "solo-specific policy" in result.assistant_system_prompt
    assert "dual-specific policy" not in result.assistant_system_prompt
