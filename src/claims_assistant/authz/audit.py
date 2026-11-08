"""Audit log for every permission decision (Step 8).

Append-only and hash-chained, so a deleted or edited line is detectable. Production
systems write this to a WORM store or a dedicated audit service; the chaining shown
here is the property that matters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class AuditLog:
    entries: list[dict] = field(default_factory=list)
    _last_hash: str = "0" * 64

    def write(
        self,
        *,
        subject: str,
        action: str,
        resource: str,
        allowed: bool,
        reason: str,
        service: str,
        session_id: str = "",
    ) -> dict:
        entry = {
            "at": datetime.now(UTC).isoformat(),
            "service": service,
            "subject": subject,
            "session": session_id,
            "action": str(action),
            "resource": resource,
            "allowed": allowed,
            "reason": reason,
            "prev": self._last_hash,
        }
        digest = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        entry["hash"] = digest
        self._last_hash = digest
        self.entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for e in self.entries:
            body = {k: v for k, v in e.items() if k != "hash"}
            if body["prev"] != prev:
                return False
            if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != e["hash"]:
                return False
            prev = e["hash"]
        return True

    def denials(self) -> list[dict]:
        return [e for e in self.entries if not e["allowed"]]
