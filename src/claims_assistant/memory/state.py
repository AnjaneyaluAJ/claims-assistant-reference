"""Conversation memory and case state (Step 10).

Four kinds of state, four owners:

* Conversation history (here): last turns verbatim, older turns as a running summary.
* Case state (here): a typed object shared by agents and the workflow.
* The system of record: reached through tools only, never cached here as fact.
* Long-term preferences (here): opt-in, visible to the customer, never claim facts.

Production stores these in a database with the insurer's retention rules; the interfaces
are what matter, and they are small on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ..domain.models import EstimateFields, IncidentDetails


class FnolState(BaseModel):
    state: str = "identify_policy"
    policy_id: str | None = None
    candidate_policies: list[dict] = Field(default_factory=list)
    incident: IncidentDetails = Field(default_factory=IncidentDetails)
    document_ids: list[str] = Field(default_factory=list)
    estimates: dict[str, EstimateFields] = Field(default_factory=dict)
    estimate_warnings: list[str] = Field(default_factory=list)
    draft_id: str | None = None
    idempotency_key: str | None = None
    validation_failures: int = 0
    claim_id: str | None = None


class PendingConfirmation(BaseModel):
    """A write waiting for the customer's explicit yes (and step-up)."""

    action: str  # "update_payout_account" | "submit_claim"
    args: dict[str, Any]
    readback: str


class CaseState(BaseModel):
    known_policy_ids: list[str] = Field(default_factory=list)
    claims_discussed: list[str] = Field(default_factory=list)
    workflow: FnolState | None = None
    pending: PendingConfirmation | None = None
    failed_attempts: int = 0
    handed_off: bool = False


@dataclass
class Conversation:
    conversation_id: str
    customer_id: str
    session_id: str
    turns: list[dict[str, str]] = field(default_factory=list)  # {"role": "user"|"assistant", "content": str}
    summary: str = ""
    case: CaseState = field(default_factory=CaseState)

    def history_for_model(self, verbatim_turns: int) -> list[dict[str, Any]]:
        recent = self.turns[-2 * verbatim_turns :]
        msgs: list[dict[str, Any]] = []
        if self.summary:
            msgs.append({"role": "user", "content": f"<conversation_summary>{self.summary}</conversation_summary>"})
            msgs.append({"role": "assistant", "content": "Understood."})
        msgs.extend({"role": m["role"], "content": m["content"]} for m in recent)
        return msgs

    def needs_summary(self, verbatim_turns: int) -> bool:
        return len(self.turns) > 2 * verbatim_turns + 4


@dataclass
class Preferences:
    """Opt-in only. Keys are limited so claim facts cannot be stored here by accident."""

    ALLOWED = frozenset({"preferred_contact", "language", "large_text"})
    values: dict[str, dict[str, str]] = field(default_factory=dict)

    def set(self, customer_id: str, key: str, value: str) -> None:
        if key not in self.ALLOWED:
            raise ValueError(f"{key} is not a preference")
        self.values.setdefault(customer_id, {})[key] = value

    def get(self, customer_id: str) -> dict[str, str]:
        return dict(self.values.get(customer_id, {}))


@dataclass
class ConversationStore:
    conversations: dict[str, Conversation] = field(default_factory=dict)

    def get_or_create(self, conversation_id: str, customer_id: str, session_id: str) -> Conversation:
        conv = self.conversations.get(conversation_id)
        if conv is None:
            conv = Conversation(conversation_id, customer_id, session_id)
            self.conversations[conversation_id] = conv
        elif conv.customer_id != customer_id:
            raise PermissionError("conversation belongs to another customer")
        conv.session_id = session_id
        return conv
