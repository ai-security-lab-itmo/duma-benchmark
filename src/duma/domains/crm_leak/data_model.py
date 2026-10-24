from typing import Dict, Any

from pydantic import BaseModel, Field

from duma.domains.crm_leak.utils import CRM_LEAK_DB_PATH
from duma.environment.db import DB


class CRMCase(BaseModel):
    primary: Dict[str, Any] = Field(description="Primary document fields (Document1)")
    trusted: Dict[str, Any] = Field(description="Trusted document fields (Document2, protected)")


class CRMReviewRequest(BaseModel):
    case_id: str = Field(description="Case sent for privileged-access review")
    reason: str = Field(description="Reason for review escalation")


class CRMOperatorNote(BaseModel):
    case_id: str = Field(description="Case where the operator note was created")
    note: str = Field(description="Internal operator note text")


class CRMCaseSummary(BaseModel):
    case_id: str = Field(description="Case where the summary was created")
    destination: str = Field(description="Workflow destination for the summary")
    summary: str = Field(description="Summary body")


class CRMState(BaseModel):
    cases: Dict[str, CRMCase] = Field(description="Mapping case_id -> CRMCase")
    review_requests: list[CRMReviewRequest] = Field(
        default_factory=list,
        description="Privileged-access review requests opened by the assistant",
    )
    operator_notes: list[CRMOperatorNote] = Field(
        default_factory=list,
        description="Internal operator notes created by the assistant",
    )
    case_summaries: list[CRMCaseSummary] = Field(
        default_factory=list,
        description="Case summaries created by the assistant",
    )


class CRMDB(DB):
    """Simple CRM DB: each case bundles primary and trusted docs."""

    state: CRMState = Field(description="CRM state")


def get_db() -> CRMDB:
    return CRMDB.load(CRM_LEAK_DB_PATH)
