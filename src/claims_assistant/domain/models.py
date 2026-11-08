"""Domain records held by the (simulated) systems of record.

These are the insurer's facts. The assistant never stores them in chat memory;
it reads them through MCP tools on every turn (Step 10).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Role(StrEnum):
    """A person's relationship to a policy. Authorization is decided by this, not by tiers."""

    POLICYHOLDER = "policyholder"
    NAMED_DRIVER = "named_driver"
    DELEGATE = "delegate"
    BROKER = "broker"


class Party(BaseModel):
    customer_id: str
    role: Role


class Delegation(BaseModel):
    """What a policyholder has allowed other parties to do (S5)."""

    named_drivers_see_all_claims: bool = False
    delegate_can_submit_claims: bool = False


class CoverItem(BaseModel):
    code: str
    title: str
    included: bool
    excess: int | None = None
    limit: int | None = None
    conditions: str = ""


class Policy(BaseModel):
    policy_id: str
    product: str  # "motor" | "home"
    status: str
    start: date
    end: date
    vehicle: str | None = None
    address_district: str | None = None
    parties: list[Party]
    delegation: Delegation = Field(default_factory=Delegation)
    cover: list[CoverItem]
    wording_version: str


class ClaimEvent(BaseModel):
    at: datetime
    text: str
    source: str  # "adjuster" | "portal" | "phone" | "assistant"


class Claim(BaseModel):
    claim_id: str
    policy_id: str
    incident_party: str  # customer_id of the driver / person the incident belongs to
    loss_type: str
    incident_date: date
    status: str
    next_step: str
    approved_estimate: int | None = None
    adjuster_flags: list[str] = Field(default_factory=list)
    events: list[ClaimEvent] = Field(default_factory=list)


class Customer(BaseModel):
    customer_id: str
    display_name: str
    username: str
    phone_last4: str


class IncidentDetails(BaseModel):
    """Fields the FNOL workflow collects. Filled by extraction, validated by code."""

    loss_type: str | None = None
    incident_date: date | None = None
    location: str | None = None
    description: str | None = None
    third_party_involved: bool | None = None
    injuries: bool | None = None
    police_reference: str | None = None


class EstimateFields(BaseModel):
    """Typed fields extracted from a garage estimate (Step 9). Free text never leaves the extractor."""

    garage_name: str | None = None
    estimate_date: date | None = None
    line_items: list[tuple[str, int]] = Field(default_factory=list)
    total: int | None = None


class ClaimDraft(BaseModel):
    draft_id: str
    policy_id: str
    owner: str
    incident: IncidentDetails
    evidence_document_ids: list[str] = Field(default_factory=list)
    estimate: EstimateFields | None = None
    created_at: datetime
