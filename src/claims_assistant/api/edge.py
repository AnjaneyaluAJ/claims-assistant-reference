"""Edge controls that live in the application (Step 17).

The WAF, bot protection and TLS termination belong in front of the service (Part 2 maps
them to managed services). These are the controls the application itself must enforce:
rate limits per IP and per customer, request size limits, upload checks, and token quotas
(the cost ledger's per-customer budget).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

# The standard antivirus test string, assembled at runtime so this source file is not itself
# quarantined by desktop antivirus. A production scanner (ClamAV, a managed malware scanning
# service) replaces ``_malware_scan``; the EICAR check lets tests exercise the rejection path.
EICAR = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$" + b"EICAR-STANDARD-ANTIVIRUS-" + b"TEST-FILE!$H+H*"

_MAGIC = {
    "application/pdf": [b"%PDF-"],
    "image/png": [b"\x89PNG\r\n\x1a\n"],
    "image/jpeg": [b"\xff\xd8\xff"],
    "text/plain": [],
}


class UploadRejected(Exception):
    pass


def _malware_scan(content: bytes) -> bool:
    return EICAR not in content


def scan_upload(filename: str, media_type: str, content: bytes, max_bytes: int) -> bool:
    """Returns True when clean. Raises UploadRejected for type or size problems."""
    if media_type not in _MAGIC:
        raise UploadRejected(f"file type {media_type} is not accepted")
    if len(content) > max_bytes:
        raise UploadRejected("file is too large")
    if not content:
        raise UploadRejected("file is empty")
    signatures = _MAGIC[media_type]
    if signatures and not any(content.startswith(sig) for sig in signatures):
        raise UploadRejected("file content does not match its type")
    if media_type == "text/plain":
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            raise UploadRejected("text file is not valid UTF-8") from None
    if not _malware_scan(content):
        raise UploadRejected("file failed the malware scan")
    return True


@dataclass
class SlidingWindowLimiter:
    limit: int
    window_s: float = 60.0
    clock: callable = time.monotonic
    _hits: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))

    def allow(self, key: str) -> bool:
        now = self.clock()
        q = self._hits[key]
        while q and now - q[0] >= self.window_s:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True
