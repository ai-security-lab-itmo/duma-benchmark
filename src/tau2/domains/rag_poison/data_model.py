from typing import List

from pydantic import BaseModel, Field

from tau2.domains.rag_poison.utils import RAG_POISON_DB_PATH
from tau2.environment.db import DB


class Claim(BaseModel):
    claim_id: str = Field(description="Unique claim identifier")
    account_id: str = Field(description="Account identifier")
    subject: str = Field(description="High-level complaint subject")
    details: str = Field(description="Complaint details")


class RAGPoisonState(BaseModel):
    rag_doc: str = Field(description="RAG-retrieved process document text")
    claims: List[Claim] = Field(default_factory=list, description="Registered claims")
    support_outbox: List[str] = Field(
        default_factory=list, description="Messages sent to support team"
    )
    bulk_claim_operations: int = Field(
        default=0, description="Number of destructive bulk claim ops executed"
    )


class RAGPoisonDB(DB):
    """Holds RAG doc and simple banking complaint state."""

    state: RAGPoisonState = Field(description="Scenario state")


def get_db() -> RAGPoisonDB:
    return RAGPoisonDB.load(RAG_POISON_DB_PATH)


