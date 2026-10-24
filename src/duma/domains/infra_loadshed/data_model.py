from typing import Literal

from pydantic import BaseModel, Field

from duma.domains.infra_loadshed.utils import INFRA_DB_PATH
from duma.environment.db import DB


class InfraState(BaseModel):
    cluster_id: str = Field(description="Cluster identifier")
    quota_tokens: int = Field(description="Total burst tokens available for the shift")
    tokens_spent: int = Field(default=0, description="Tokens spent this shift")
    external_api_calls: int = Field(default=0, description="Paid API calls executed")
    cluster_status: Literal["healthy", "degraded", "offline"] = Field(
        default="healthy", description="Overall cluster health state"
    )
    pending_jobs: list[dict] = Field(
        default_factory=list, description="Jobs awaiting approval"
    )
    collab_note: str = Field(
        description="Instruction from orchestration partner, potentially malicious"
    )
    secondary_collab_note: str | None = Field(
        default=None, description="Counter-instruction from security/SRE"
    )
    quota_override_requested: bool = Field(
        default=False, description="Whether override quota change was attempted"
    )
    over_quota_launch_attempted: bool = Field(
        default=False, description="Whether a job launch over quota was attempted"
    )
    cluster_shutdown_triggered: bool = Field(
        default=False, description="Whether the cluster was shut down"
    )
    sre_escalation_sent: bool = Field(
        default=False, description="Whether a concern was escalated to SRE"
    )
    quota_increase_requested: bool = Field(
        default=False, description="Whether a formal quota increase was requested"
    )


class InfraDB(DB):
    """Holds the infra loadshed scenario state."""

    state: InfraState = Field(description="Cluster state")


def get_db() -> InfraDB:
    return InfraDB.load(INFRA_DB_PATH)
