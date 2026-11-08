"""Personal-data detector.

Pattern-based, which is enough to demonstrate the architecture and is fully testable.
A production deployment puts a trained detector behind the same ``detect`` function
(an open-source recognizer suite, or a managed service; Part 2 uses one) and keeps these
patterns as a fast first pass. Medical and injury content is not tokenized here: it is a
handoff trigger (Step 12) and is kept away from third-party models by routing (Step 11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    kind: str
    start: int
    end: int
    value: str


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("card_number", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")),
    ("sort_code", re.compile(r"\b\d{2}[- ]\d{2}[- ]\d{2}\b")),
    ("account_number", re.compile(r"(?<![\w-])\d{8}(?![\w-])")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"(?<!\w)(?:\+44\s?7\d{3}|07\d{3})\s?\d{3}\s?\d{3}(?!\w)")),
    ("ni_number", re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")),
]


def detect(text: str) -> list[Detection]:
    found: list[Detection] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            value = m.group(0)
            if kind == "card_number":
                digits = re.sub(r"\D", "", value)
                if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                    continue
            found.append(Detection(kind, s, e, value))
            taken.append((s, e))
    return sorted(found, key=lambda d: d.start)
