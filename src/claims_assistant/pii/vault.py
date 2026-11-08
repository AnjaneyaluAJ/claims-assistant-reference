"""Reversible PII tokenization (Step 11).

A detector finds personal data in every inbound message and document field and swaps
each value for a placeholder such as ``<pii:account_number:3f9a1c2b>``. The real value is
encrypted into a vault inside the insurer's boundary. Models, logs and traces only ever see
placeholders. Only services on the vault's allow list (the payments MCP server) can resolve
a placeholder, and only for the session it was created in.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass, field

from cryptography.fernet import Fernet

from .detector import Detection, detect

PLACEHOLDER = re.compile(r"<pii:([a-z_]+):([0-9a-f]{8})>")


class VaultAccessDenied(Exception):
    pass


@dataclass
class _Entry:
    kind: str
    ciphertext: bytes
    session_id: str


@dataclass
class TokenVault:
    resolvers: frozenset[str] = frozenset({"payments-mcp"})
    _fernet: Fernet = field(default_factory=lambda: Fernet(Fernet.generate_key()))
    _hmac_key: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    _entries: dict[str, _Entry] = field(default_factory=dict)

    def _token_id(self, session_id: str, kind: str, value: str) -> str:
        # Deterministic per session, so the same account number maps to the same placeholder
        # across turns, while different sessions get unrelated placeholders.
        mac = hmac.new(self._hmac_key, f"{session_id}|{kind}|{value}".encode(), hashlib.sha256)
        return mac.hexdigest()[:8]

    def tokenize(self, text: str, session_id: str) -> tuple[str, list[Detection]]:
        detections = detect(text)
        if not detections:
            return text, []
        out, last = [], 0
        for d in detections:
            tid = self._token_id(session_id, d.kind, d.value)
            self._entries.setdefault(tid, _Entry(d.kind, self._fernet.encrypt(d.value.encode()), session_id))
            out.append(text[last : d.start])
            out.append(f"<pii:{d.kind}:{tid}>")
            last = d.end
        out.append(text[last:])
        return "".join(out), detections

    def resolve(self, placeholder: str, *, service: str, session_id: str) -> str:
        if service not in self.resolvers:
            raise VaultAccessDenied(f"{service} may not resolve personal data")
        m = PLACEHOLDER.fullmatch(placeholder.strip())
        if not m:
            raise VaultAccessDenied("not a placeholder")
        entry = self._entries.get(m.group(2))
        if entry is None or entry.session_id != session_id or entry.kind != m.group(1):
            raise VaultAccessDenied("unknown placeholder for this session")
        return self._fernet.decrypt(entry.ciphertext).decode()

    def mask_for_display(self, text: str) -> str:
        """Render placeholders for the customer without revealing full values."""

        def _mask(m: re.Match) -> str:
            entry = self._entries.get(m.group(2))
            if entry is None:
                return "[redacted]"
            value = self._fernet.decrypt(entry.ciphertext).decode()
            digits = re.sub(r"\D", "", value)
            label = m.group(1).replace("_", " ")
            if digits and len(digits) >= 4:
                return f"{label} ending {digits[-4:]}"
            return f"[{label}]"

        return PLACEHOLDER.sub(_mask, text)


def redact(text: str) -> str:
    """One-way redaction for logs and traces: no vault, no way back."""
    detections = detect(text)
    if not detections:
        return text
    out, last = [], 0
    for d in detections:
        out.append(text[last : d.start])
        out.append(f"[{d.kind}]")
        last = d.end
    out.append(text[last:])
    return "".join(out)
