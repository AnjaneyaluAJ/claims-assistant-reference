"""Output checks (Step 9, layer 5).

Replies are scanned for promises the assistant may not make and for amounts that did not come
from a tool result in this turn. A failing reply is replaced with a safe one; the original is
kept on the trace for review. This is a backstop, not the main defence: the main defence is that
no tool can approve, deny or value a claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PROMISES = re.compile(
    r"\b(is|are|has been|have been|will be|been)\s+(pre-?)?approved\b|\byou will (receive|be paid|get)\b|"
    r"\b(guarantee|guaranteed)\b|\bclaim (is|has been) (accepted|settled)\b|\bwe will pay\b",
    re.IGNORECASE,
)
_AMOUNT = re.compile(r"(?:[$£€]\s?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?")
_FULL_ACCOUNT = re.compile(r"(?<![\w-])\d{8}(?![\w-])")

# Statements of fact from the claims system are allowed ("the approved estimate is $2,150")
# when the amount is grounded in a tool result. Only the promise forms above are blocked.
_GROUNDED_APPROVAL = re.compile(r"approved (repair )?estimate", re.IGNORECASE)


@dataclass
class OutputVerdict:
    ok: bool
    reasons: list[str]
    text: str


SAFE_REPLY = (
    "I can't confirm that. Decisions on claims and payments are made by our claims team, "
    "and I can share what the claims system shows or put you through to an adjuster."
)


def _amounts(text: str) -> set[int]:
    return {int(m.group(1).replace(",", "")) for m in _AMOUNT.finditer(text)}


def check_reply(reply: str, grounded_text: str) -> OutputVerdict:
    """``grounded_text`` is everything returned by tools this turn (and the workflow's own data)."""
    reasons = []
    stripped = _GROUNDED_APPROVAL.sub("", reply)
    if _PROMISES.search(stripped):
        reasons.append("promise_of_outcome")
    grounded = _amounts(grounded_text) | {int(x) for x in re.findall(r"\b\d+\b", grounded_text)}
    ungrounded = sorted(a for a in _amounts(reply) if a not in grounded)
    if ungrounded:
        reasons.append(f"ungrounded_amount:{ungrounded}")
    if _FULL_ACCOUNT.search(reply):
        reasons.append("full_account_number")
    if reasons:
        return OutputVerdict(False, reasons, SAFE_REPLY)
    return OutputVerdict(True, [], reply)
