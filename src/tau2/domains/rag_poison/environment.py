import json
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.rag_poison.data_model import RAGPoisonDB
from tau2.domains.rag_poison.tools import RAGPoisonTools
from tau2.domains.rag_poison.utils import (
    RAG_POISON_DB_PATH,
    RAG_POISON_POLICY_PATH,
    RAG_POISON_TASK_SET_PATH,
)
from tau2.environment.environment import Environment


def get_environment(db: Optional[RAGPoisonDB] = None, solo_mode: bool = False) -> Environment:
    if db is None:
        db = RAGPoisonDB.load(RAG_POISON_DB_PATH)
    tools = RAGPoisonTools(db)
    with open(RAG_POISON_POLICY_PATH, "r") as fp:
        policy = fp.read()
    env = Environment(
        domain_name="rag_poison",
        policy=policy,
        tools=tools,
    )
    if solo_mode:
        env.set_solo_mode(True)
    return env


def get_tasks() -> list[Task]:
    with open(RAG_POISON_TASK_SET_PATH, "r") as fp:
        tasks = json.load(fp)
    return [Task.model_validate(task) for task in tasks]


