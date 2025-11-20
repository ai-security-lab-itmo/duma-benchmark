from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from tau2.domains.mail_rag_poison.utils import MAIL_RAG_DB_PATH
from tau2.environment.db import DB


class Email(BaseModel):
    id: str
    from_addr: str
    to_addr: str
    subject: str
    body_text: str
    attachments_text: List[str] = Field(default_factory=list)


class MailRAGState(BaseModel):
    emails: List[Email] = Field(default_factory=list)
    vector_index: List[Tuple[str, str]] = Field(
        default_factory=list, description="(doc_id, text) pairs"
    )
    payments: Dict[str, bool] = Field(
        default_factory=dict, description="key 'vendor|project' -> confirmed?"
    )
    outbox: List[str] = Field(default_factory=list, description="messages sent to clients")


class MailRAGDB(DB):
    state: MailRAGState = Field(description="Shared mailbox + RAG index + payments")


def get_db() -> MailRAGDB:
    return MailRAGDB.load(MAIL_RAG_DB_PATH)


