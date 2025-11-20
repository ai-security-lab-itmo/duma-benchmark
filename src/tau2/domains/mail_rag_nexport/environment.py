import json
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.mail_rag_nexport.data_model import NexportRAGDB
from tau2.domains.mail_rag_nexport.tools import NexportRAGTools
from tau2.domains.mail_rag_nexport.utils import (
    MAIL_RAG_NEXPORT_DB_PATH,
    MAIL_RAG_NEXPORT_POLICY_PATH,
    MAIL_RAG_NEXPORT_TASK_SET_PATH,
    MAIL_RAG_NEXPORT_DESCRIPTION_PATH,
)
from tau2.environment.environment import Environment
from tau2.domains._task_utils import resolve_task_file_refs


def get_environment(db: Optional[NexportRAGDB] = None, solo_mode: bool = False) -> Environment:
    if db is None:
        db = NexportRAGDB.load(MAIL_RAG_NEXPORT_DB_PATH)
    tools = NexportRAGTools(db)
    with open(MAIL_RAG_NEXPORT_POLICY_PATH, "r") as fp:
        policy = fp.read()
    env = Environment(
        domain_name="mail_rag_nexport",
        policy=policy,
        tools=tools,
    )
    if solo_mode:
        env.set_solo_mode(True)
    return env


def get_tasks() -> list[Task]:
    with open(MAIL_RAG_NEXPORT_TASK_SET_PATH, "r") as fp:
        raw = json.load(fp)
    raw = resolve_task_file_refs(raw, MAIL_RAG_NEXPORT_TASK_SET_PATH)
    return [Task.model_validate(task) for task in raw]


