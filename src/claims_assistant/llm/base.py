"""Provider-neutral model interface.

Agents talk to ``ModelClient``. Three implementations sit behind it:

* ``AnthropicBackend`` on the Claude API (first party)
* ``AnthropicBackend`` on Amazon Bedrock (same Messages API shape; Part 2 uses this)
* ``ScriptedModel``, a deterministic stand-in so tests and the evaluation gate run in CI
  without a key. It checks the architecture, not model quality (see README).

Messages and content blocks use the Messages API shape (plain dicts), so the history an
agent keeps can be sent unchanged to the real API, thinking blocks included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __iadd__(self, other: Usage) -> Usage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        return self

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens


@dataclass
class ModelResponse:
    content: list[dict[str, Any]]
    stop_reason: str
    usage: Usage
    model: str
    request_id: str | None = None
    provider: str = ""

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]


@dataclass
class ModelRequest:
    route: str  # "coordinator", "claims_agent", "extractor", ... (used by the router and in traces)
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_tokens: int = 4096
    output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelUnavailable(Exception):
    """The model endpoint failed after its own retries; the router may try a fallback."""


class ModelClient(Protocol):
    name: str

    async def create(self, request: ModelRequest, *, model: str, effort: str) -> ModelResponse: ...
