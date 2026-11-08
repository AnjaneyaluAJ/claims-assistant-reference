"""Cost per conversation and per-customer token budgets (Step 15).

The useful unit is cost per resolved conversation: a cheap conversation that ends in a phone
call saved nothing. The ledger records cost per conversation and its outcome, so the report
can divide one by the other.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime


class BudgetExceeded(Exception):
    pass


@dataclass
class ConversationCost:
    customer_id: str
    usd: float = 0.0
    tokens: int = 0
    model_calls: int = 0
    outcome: str = "open"  # open | resolved | handoff | abandoned


@dataclass
class CostLedger:
    daily_token_budget: int
    conversations: dict[str, ConversationCost] = field(default_factory=dict)
    _daily: dict[tuple[str, date], int] = field(default_factory=lambda: defaultdict(int))

    def check_budget(self, customer_id: str) -> None:
        if self._daily[(customer_id, datetime.now(UTC).date())] >= self.daily_token_budget:
            raise BudgetExceeded(customer_id)

    def record(self, conversation_id: str, customer_id: str, tokens: int, usd: float) -> None:
        c = self.conversations.setdefault(conversation_id, ConversationCost(customer_id))
        c.usd += usd
        c.tokens += tokens
        c.model_calls += 1
        self._daily[(customer_id, datetime.now(UTC).date())] += tokens

    def set_outcome(self, conversation_id: str, outcome: str) -> None:
        if conversation_id in self.conversations:
            self.conversations[conversation_id].outcome = outcome

    def report(self) -> dict:
        total = sum(c.usd for c in self.conversations.values())
        resolved = [c for c in self.conversations.values() if c.outcome == "resolved"]
        return {
            "conversations": len(self.conversations),
            "resolved": len(resolved),
            "handoffs": sum(c.outcome == "handoff" for c in self.conversations.values()),
            "total_usd": round(total, 6),
            "usd_per_resolved_conversation": round(total / len(resolved), 6) if resolved else None,
            "model_calls": sum(c.model_calls for c in self.conversations.values()),
        }
