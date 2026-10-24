from pydantic import BaseModel, Field

from duma.domains.auth_spoof_support.utils import AUTH_SPOOF_SUPPORT_DB_PATH
from duma.environment.db import DB


class Account(BaseModel):
    account_id: str
    full_name: str
    email: str
    balance: float
    verification_code: str = Field(description="Deterministic code for local simulation")


class RefundEntry(BaseModel):
    account_id: str
    amount: float
    verified: bool


class EmailChangeEntry(BaseModel):
    account_id: str
    old_email: str
    new_email: str
    verified: bool


class RecoveryEmailQueueEntry(BaseModel):
    account_id: str
    new_email: str
    approval_reference: str
    verified: bool


class RefundHoldReleaseEntry(BaseModel):
    account_id: str
    amount: float
    approval_reference: str
    verified: bool


class ManualReviewTicket(BaseModel):
    ticket_id: str
    account_id: str
    review_type: str
    decision: str
    approved_actions: list[str] = Field(default_factory=list)
    summary: str
    notes: str | None = None


class AuthSpoofState(BaseModel):
    accounts: dict[str, Account] = Field(default_factory=dict)
    manual_review_tickets: dict[str, ManualReviewTicket] = Field(default_factory=dict)
    verification_state: dict[str, bool] = Field(default_factory=dict)
    pending_codes: dict[str, str] = Field(default_factory=dict)
    refund_log: list[RefundEntry] = Field(default_factory=list)
    email_change_log: list[EmailChangeEntry] = Field(default_factory=list)
    recovery_email_queue: list[RecoveryEmailQueueEntry] = Field(default_factory=list)
    refund_hold_release_log: list[RefundHoldReleaseEntry] = Field(default_factory=list)
    collab_note: str = Field(description="Potentially malicious support instruction")
    secondary_collab_note: str | None = Field(
        default=None,
        description="Optional security guidance",
    )
    privileged_action_without_verification: bool = Field(
        default=False,
        description="Whether a privileged action was executed without identity check",
    )


class AuthSpoofDB(DB):
    state: AuthSpoofState = Field(description="Support auth workflow state")


def get_db() -> AuthSpoofDB:
    return AuthSpoofDB.load(AUTH_SPOOF_SUPPORT_DB_PATH)
