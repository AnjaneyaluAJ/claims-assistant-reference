"""Circuit breaker per backend (Step 16).

When a backend keeps failing, its tools return "unavailable" immediately instead of
piling up requests and holding conversations open until they time out.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    reset_after_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    failures: int = 0
    opened_at: float | None = None
    _half_open_probe: bool = field(default=False, repr=False)

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.reset_after_s:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        s = self.state
        if s == "closed":
            return True
        if s == "half_open" and not self._half_open_probe:
            self._half_open_probe = True  # let exactly one probe through
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None
        self._half_open_probe = False

    def record_failure(self) -> None:
        self.failures += 1
        self._half_open_probe = False
        if self.failures >= self.failure_threshold:
            self.opened_at = self.clock()
