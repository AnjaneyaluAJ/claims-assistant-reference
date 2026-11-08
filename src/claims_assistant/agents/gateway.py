"""The agent runtime's path to the MCP servers (Steps 5, 7 and 16).

This is the separate path from the diagram in Step 5. The model proposes a tool call; the
gateway attaches a service token that the model never sees, minted by token exchange for that
one server and only the scope that tool needs. It also owns timeouts, safe retries and the
per-backend circuit breakers.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from opentelemetry.trace import Status, StatusCode, Tracer

from ..config import READ_ONLY_TOOLS, TOOL_SCOPES
from ..identity.idp import DevIdentityProvider
from ..observability import tracing as t
from ..reliability.breaker import CircuitBreaker


@dataclass
class ToolOutcome:
    tool: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    latency_ms: float = 0.0

    def for_model(self) -> str:
        if self.ok:
            return json.dumps(self.data, default=str)
        return json.dumps({"error": {"code": self.error_code, "message": self.error_message}})


@dataclass
class ToolCatalog:
    """Tool definitions discovered from the MCP servers at startup, by server."""

    by_server: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def server_of(self) -> dict[str, str]:
        return {tool["name"]: srv for srv, tools in self.by_server.items() for tool in tools}

    def for_servers(self, *servers: str, only: set[str] | None = None) -> list[dict[str, Any]]:
        out = []
        for s in servers:
            for tool in self.by_server.get(s, []):
                if only is None or tool["name"] in only:
                    out.append(tool)
        return out


ConnectFn = Callable[..., AbstractAsyncContextManager[Client]]


def http_connector(apps: dict[str, Any] | None, urls: dict[str, str]) -> ConnectFn:
    """Connect to an MCP server with a bearer token. In-process mode uses an ASGI transport
    (same HTTP stack, no sockets); otherwise real HTTP to the server's URL."""

    def connect(server: str, token: str, statuses: list[int] | None = None) -> AbstractAsyncContextManager[Client]:
        url = urls[server]
        base = url.rsplit("/mcp", 1)[0]
        transport = httpx2.ASGITransport(app=apps[server]) if apps else None

        async def record(response: httpx2.Response) -> None:
            if statuses is not None:
                statuses.append(response.status_code)

        http = httpx2.AsyncClient(
            base_url=base,
            transport=transport,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx2.Timeout(30.0),
            event_hooks={"response": [record]},
        )
        return Client(streamable_http_client(url, http_client=http))

    return connect


class McpGateway:
    def __init__(
        self,
        *,
        idp: DevIdentityProvider,
        session_token: str,
        catalog: ToolCatalog,
        urls: dict[str, str],
        connect: ConnectFn,
        breakers: dict[str, CircuitBreaker],
        tracer: Tracer,
        span_attrs: dict[str, str],
        tool_timeout_s: float = 8.0,
        deadline: float | None = None,
    ) -> None:
        self.idp = idp
        self.session_token = session_token
        self.catalog = catalog
        self.urls = urls
        self.connect = connect
        self.breakers = breakers
        self.tracer = tracer
        self.span_attrs = span_attrs
        self.tool_timeout_s = tool_timeout_s
        self.deadline = deadline  # monotonic time by which the turn must finish
        self._tokens: dict[tuple[str, str], str] = {}
        self.calls: list[ToolOutcome] = []

    def _token(self, server: str, scope: str) -> str:
        key = (server, scope)
        if key not in self._tokens:
            self._tokens[key] = self.idp.exchange(self.session_token, self.urls[server], {scope})
        return self._tokens[key]

    async def call(self, tool: str, args: dict[str, Any], *, call_id: str = "", agent: str = "") -> ToolOutcome:
        server = self.catalog.server_of.get(tool)
        with self.tracer.start_as_current_span(f"execute_tool {tool}") as span:
            for k, v in self.span_attrs.items():
                span.set_attribute(k, v)
            span.set_attribute(t.OP, "execute_tool")
            span.set_attribute(t.TOOL_NAME, tool)
            span.set_attribute(t.TOOL_CALL_ID, call_id)
            span.set_attribute("app.tool.arguments", t.as_json(args))
            span.set_attribute("app.agent", agent)
            if server is None:
                outcome = ToolOutcome(tool, False, error_code="unknown_tool", error_message="No such tool.")
            else:
                span.set_attribute("app.mcp.server", server)
                outcome = await self._call_with_policy(server, tool, args)
            span.set_attribute("app.tool.ok", outcome.ok)
            span.set_attribute("app.tool.latency_ms", outcome.latency_ms)
            if not outcome.ok:
                span.set_attribute("app.tool.error_code", outcome.error_code or "")
                if outcome.error_code in {"unavailable", "timeout", "circuit_open"}:
                    span.set_status(Status(StatusCode.ERROR, outcome.error_code))
            else:
                span.set_attribute("app.tool.result", t.as_json(outcome.data))
        self.calls.append(outcome)
        return outcome

    async def _call_with_policy(self, server: str, tool: str, args: dict[str, Any]) -> ToolOutcome:
        breaker = self.breakers[server]
        # Retry only what is safe to repeat: reads, and writes that carry an idempotency key.
        safe = tool in READ_ONLY_TOOLS or "idempotency_key" in args
        attempts = 3 if safe else 1
        outcome = ToolOutcome(tool, False, error_code="unavailable", error_message="Service unavailable.")
        for attempt in range(attempts):
            if not breaker.allow():
                return ToolOutcome(
                    tool,
                    False,
                    error_code="circuit_open",
                    error_message=f"The {server.split('-')[0]} system is temporarily unavailable.",
                )
            budget = self.tool_timeout_s
            if self.deadline is not None:
                budget = min(budget, self.deadline - time.monotonic())
            if budget <= 0:
                return ToolOutcome(tool, False, error_code="timeout", error_message="Turn time budget exhausted.")
            outcome = await self._call_once(server, tool, args, budget)
            if outcome.ok or outcome.error_code not in {"unavailable", "timeout"}:
                breaker.record_success()
                return outcome
            breaker.record_failure()
            if attempt + 1 < attempts:
                await anyio.sleep(0.05 * 2**attempt)
        return outcome

    async def _call_once(self, server: str, tool: str, args: dict[str, Any], timeout_s: float) -> ToolOutcome:
        start = time.monotonic()
        token = self._token(server, TOOL_SCOPES.get(tool, ""))
        statuses: list[int] = []
        try:
            with anyio.fail_after(timeout_s):
                async with self.connect(server, token, statuses) as client:
                    result = await client.call_tool(tool, args)
        except TimeoutError:
            return ToolOutcome(
                tool,
                False,
                error_code="timeout",
                error_message="The request timed out.",
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            latency = (time.monotonic() - start) * 1000
            # An auth rejection is a bug or an attack, not an outage: it must not trip the breaker
            # or be retried, and it must be visible as what it is.
            if any(code in (401, 403) for code in statuses):
                return ToolOutcome(
                    tool,
                    False,
                    error_code="unauthorized",
                    error_message="This action is not available in the current session.",
                    latency_ms=latency,
                )
            return ToolOutcome(
                tool, False, error_code="unavailable", error_message=type(exc).__name__, latency_ms=latency
            )
        latency = (time.monotonic() - start) * 1000
        text = "".join(getattr(c, "text", "") for c in result.content)
        if result.is_error:
            try:
                err = json.loads(text)["error"]
                return ToolOutcome(
                    tool, False, error_code=err.get("code"), error_message=err.get("message"), latency_ms=latency
                )
            except (ValueError, KeyError, TypeError):
                # Argument validation errors from the MCP SDK arrive as plain text.
                return ToolOutcome(
                    tool, False, error_code="invalid_argument", error_message=text[:300], latency_ms=latency
                )
        data = result.structured_content or (json.loads(text) if text.strip().startswith("{") else {"text": text})
        if isinstance(data, dict) and set(data) == {"result"} and isinstance(data["result"], dict):
            data = data["result"]
        return ToolOutcome(tool, True, data=data, latency_ms=latency)
