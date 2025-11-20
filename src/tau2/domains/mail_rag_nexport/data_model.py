from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from tau2.domains.mail_rag_nexport.utils import MAIL_RAG_NEXPORT_DB_PATH
from tau2.environment.db import DB


class Email(BaseModel):
    id: str
    from_addr: str
    to_addr: str
    subject: str
    body_text: str
    attachments_text: List[str] = Field(default_factory=list)


class NexportRAGState(BaseModel):
    emails: List[Email] = Field(default_factory=list)
    vector_index: List[Tuple[str, str]] = Field(default_factory=list)
    outbox: List[str] = Field(default_factory=list)


class NexportRAGDB(DB):
    state: NexportRAGState = Field(description="Shared mailbox + RAG index + outbox")


def get_db() -> NexportRAGDB:
    return NexportRAGDB.load(MAIL_RAG_NEXPORT_DB_PATH)


