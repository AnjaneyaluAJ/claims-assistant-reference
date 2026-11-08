"""Handoff to a human adjuster (Step 12).

Triggers are decided in code, so the pressure to "contain" conversations cannot talk the
assistant out of escalating. The adjuster gets a structured case summary built from case
state, and the customer gets an honest message with the queue's real expected wait.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..memory.state import CaseState

_TRIGGERS: list[tuple[str, re.Pattern[str]]] = [
    (
        "injury_or_medical",
        re.compile(
            r"\b(injur\w*|hurt|hospital|a&e|whiplash|neck|back pain|broke (my|his|her) \w+|fracture\w*|ambulance|"
            r"concussion|doctor|physio\w*|medical)\b",
            re.I,
        ),
    ),
    (
        "complaint",
        re.compile(
            r"\b(complain\w*|ombudsman|unacceptable|disgrace\w*|i'?m done|fed up|nobody (calls|called|replies)|"
            r"no one (calls|called)|waiting (for )?(three|four|five|\d+) weeks|lawyer|solicitor)\b",
            re.I,
        ),
    ),
    (
        "vulnerability",
        re.compile(
            r"\b(bereave\w*|passed away|died|can'?t cope|struggling to cope|disabilit\w*|carer|"
            r"financial hardship|can'?t afford)\b",
            re.I,
        ),
    ),
    (
        "asked_for_person",
        re.compile(
            r"\b(speak|talk) (to|with) (a |an )?(human|person|someone|adjuster|agent|advisor)\b|\breal person\b", re.I
        ),
    ),
]


# "No one was hurt" must not trigger an injury handoff. Negated statements are removed first.
_NEGATED = re.compile(
    r"\b(no ?one|nobody|none of us|neither of us)\s+(was|were|got)?\s*(hurt|injured)\b|"
    r"\b(no|without any) injur\w*|\b(not|wasn'?t|weren'?t|isn'?t|aren'?t)\s+(hurt|injured)\b",
    re.I,
)


def detect_triggers(text: str) -> list[str]:
    cleaned = _NEGATED.sub(" ", text)
    return [name for name, pattern in _TRIGGERS if pattern.search(cleaned)]


@dataclass
class HandoffTicket:
    ticket_id: str
    customer_id: str
    reasons: list[str]
    summary: dict
    queue: str
    expected_wait_minutes: int
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class AdjusterQueue:
    """Stand-in for the contact-centre queue. ``expected_wait`` would come from the WFM system."""

    tickets: list[HandoffTicket] = field(default_factory=list)
    base_wait_minutes: dict[str, int] = field(default_factory=lambda: {"injury": 15, "complaints": 30, "claims": 20})

    def enqueue(
        self,
        customer_id: str,
        reasons: list[str],
        case: CaseState,
        last_messages: list[str],
        assistant_answers: list[str],
    ) -> HandoffTicket:
        queue = "injury" if "injury_or_medical" in reasons else "complaints" if "complaint" in reasons else "claims"
        summary = {
            "why_escalated": reasons,
            "customer_id": customer_id,
            "policies": case.known_policy_ids,
            "claims_discussed": case.claims_discussed,
            "open_workflow": case.workflow.state if case.workflow else None,
            "draft": case.workflow.draft_id if case.workflow else None,
            "customer_said": last_messages[-3:],
            "assistant_already_answered": assistant_answers[-3:],
        }
        waiting = sum(1 for t in self.tickets if t.queue == queue)
        ticket = HandoffTicket(
            ticket_id=f"HO-{uuid.uuid4().hex[:6].upper()}",
            customer_id=customer_id,
            reasons=reasons,
            summary=summary,
            queue=queue,
            expected_wait_minutes=self.base_wait_minutes[queue] + 5 * waiting,
        )
        self.tickets.append(ticket)
        return ticket


def customer_message(ticket: HandoffTicket) -> str:
    lead = {
        "injury": "I'm sorry you've been hurt. An injury claim needs one of our adjusters, so I've passed "
        "your case to the injury team",
        "complaints": "I'm sorry about how this has gone. I've passed your case to our customer care team",
        "claims": "I've passed your case to a claims adjuster",
    }[ticket.queue]
    return (
        f"{lead} with a summary of what we've discussed, so you won't need to repeat yourself. "
        f"The current expected wait for a call back is about {ticket.expected_wait_minutes} minutes. "
        f"Your reference is {ticket.ticket_id}."
    )
