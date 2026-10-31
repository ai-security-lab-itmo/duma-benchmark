"""Capture the pre-refactor task contract without calling a model."""

import hashlib
from pathlib import Path

from duma.agent.llm_agent import LLMAgent, LLMSoloAgent
from duma.orchestrator.orchestrator import Orchestrator
from duma.registry import registry
from duma.user.user_simulator import DummyUser, UserSimulator
from duma.utils.utils import DATA_DIR
from duma.utils.generation import use_generation_backends
from duma.data_model.message import AssistantMessage, ToolCall


def task_contract():
    assets = {
        path.relative_to(DATA_DIR).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted((DATA_DIR / "duma").rglob("*"))
        if path.is_file() and path.suffix in {".json", ".md"}
    }
    tasks = {}
    for domain in registry.get_info().domains:
        for task in registry.get_tasks_loader(domain)():
            modes = {}
            for solo in (False, True):
                env = registry.get_env_constructor(domain)(solo_mode=solo)
                tools = env.get_tools()
                if solo and env.user_tools:
                    tools += env.get_user_tools()
                cls = LLMSoloAgent if solo else LLMAgent
                kwargs = {"task": task} if solo else {}
                agent = cls(
                    tools=tools,
                    domain_policy=env.get_policy(),
                    llm="reference",
                    **kwargs,
                )
                user = (DummyUser if solo else UserSimulator)(
                    tools=env.get_user_tools() if env.user_tools else None,
                    instructions=str(task.user_scenario),
                    llm="reference",
                )
                engine = Orchestrator(
                    domain=domain,
                    agent=agent,
                    user=user,
                    environment=env,
                    task=task,
                    solo_mode=solo,
                    seed=300,
                )
                with use_generation_backends(
                    {
                        "reference": lambda **_: AssistantMessage(
                            role="assistant",
                            tool_calls=[
                                ToolCall(
                                    id="reference-call",
                                    name="done",
                                    arguments={
                                        "resolution_note": "Reference initialization"
                                    },
                                )
                            ],
                            cost=0.0,
                        )
                    }
                ):
                    engine.initialize()
                modes["solo" if solo else "dual"] = {
                    "agent_schema": [tool.openai_schema for tool in agent.tools],
                    "user_schema": [
                        tool.openai_schema
                        for tool in (getattr(user, "tools", None) or [])
                    ],
                    "agent_prompt": agent.system_prompt,
                    "user_prompt": getattr(user, "system_prompt", None),
                    "db_hash": env.get_db_hash(),
                    "user_db_hash": env.get_user_db_hash(),
                    "reward_basis": task.evaluation_criteria.reward_basis,
                }
            tasks[f"{domain}/{task.id}"] = modes
    return {"assets": assets, "tasks": tasks}
