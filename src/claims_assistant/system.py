"""Wires every component together. One ``System`` is one running deployment.

In-process mode (the default, used by the CLI, tests and the eval gate) runs the four MCP servers
as ASGI apps inside this process and reaches them through the same HTTP client stack, with real
bearer-token authentication, just without sockets. Setting ``CLAIMS_MCP_MODE=remote`` points the
gateway at the URLs in ``Settings.service_urls`` instead; that requires the servers to be deployed
separately and to trust the identity provider's signing key, which this repository does not set up
(Part 2 does, on AWS).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from contextlib import AsyncExitStack
from datetime import date
from typing import Any

from .agents.gateway import ToolCatalog, http_connector
from .agents.loop import ModelCaller
from .agents.prompts import EXTRACT_ESTIMATE
from .authz.audit import AuditLog
from .config import SERVICE_SCOPES, Settings
from .cost.ledger import CostLedger
from .domain.models import EstimateFields
from .guardrails.untrusted import wrap
from .handoff.handoff import AdjusterQueue
from .identity.idp import DevIdentityProvider
from .llm.base import ModelRequest
from .llm.router import ModelRouter, build_router
from .mcp_servers import claims_server, documents_server, payments_server, policy_server
from .mcp_servers.common import ServerContext
from .memory.state import ConversationStore, Preferences
from .observability.tracing import Tracing
from .pii.vault import TokenVault
from .reliability.breaker import CircuitBreaker
from .service import ChatService
from .systems.backends import Backends

DEV_PASSWORD = "demo"  # development identity provider only

ESTIMATE_SCHEMA = {
    "type": "object",
    "properties": {
        "garage_name": {"type": ["string", "null"]},
        "estimate_date": {"type": ["string", "null"], "description": "ISO date"},
        "line_items": {
            "type": "array",
            "items": {"type": "array", "prefixItems": [{"type": "string"}, {"type": "integer"}]},
        },
        "total": {"type": ["integer", "null"]},
    },
    "required": ["garage_name", "estimate_date", "line_items", "total"],
    "additionalProperties": False,
}


class System:
    def __init__(
        self, settings: Settings | None = None, *, router: ModelRouter | None = None, today: date | None = None
    ) -> None:
        self.settings = settings or Settings()
        self.today = today or date.today()
        self.backends = Backends()
        self.otp_outbox: dict[str, str] = {}
        self.idp = DevIdentityProvider(
            users={c.username: (c.customer_id, DEV_PASSWORD) for c in self.backends.customers._by_id.values()},
            audience_scopes={self.settings.service_urls[k]: v for k, v in SERVICE_SCOPES.items()},
            otp_sender=lambda customer_id, code: self.otp_outbox.__setitem__(customer_id, code),
        )
        self.audit = AuditLog()
        self.vault = TokenVault()
        self.tracing = Tracing()
        self.router = router or build_router(self.settings)
        self.ledger = CostLedger(daily_token_budget=self.settings.daily_token_budget_per_customer)
        self.queue = AdjusterQueue()
        self.conversations = ConversationStore()
        self.preferences = Preferences()
        self.breakers = {
            name: CircuitBreaker(self.settings.breaker_failure_threshold, self.settings.breaker_reset_s)
            for name in SERVICE_SCOPES
        }
        self.agent_tool_allowlist = {
            "claims_agent": {"list_my_claims", "get_claim_status"},  # no write tools for the claims agent
            "policy_agent": {"list_my_policies", "get_cover_summary", "search_policy_wording"},
        }
        self._pseudonym_key = secrets.token_bytes(16)
        self.server_ctx = ServerContext(
            backends=self.backends,
            idp=self.idp,
            audit=self.audit,
            vault=self.vault,
            extractor=self._extract_estimate,
            urls=self.settings.service_urls,
        )
        self.servers = {
            "policy-mcp": policy_server.build(self.server_ctx),
            "claims-mcp": claims_server.build(self.server_ctx),
            "documents-mcp": documents_server.build(self.server_ctx),
            "payments-mcp": payments_server.build(self.server_ctx),
        }
        self.apps: dict[str, Any] = {}
        self.catalog = ToolCatalog()
        self.connect = None
        self.chat = ChatService(self)
        self._stack: AsyncExitStack | None = None

    def pseudonym(self, customer_id: str) -> str:
        return "p-" + hmac.new(self._pseudonym_key, customer_id.encode(), hashlib.sha256).hexdigest()[:12]

    async def _extract_estimate(self, text: str, session_id: str) -> EstimateFields:
        """The documents server's extractor: a model call with no tools and a fixed schema."""
        # Book the cost (and the token budget) to the conversation that uploaded the document.
        convs = self.conversations.conversations.values()
        conv = next((c for c in reversed(convs) if c.session_id == session_id), None)
        caller = ModelCaller(
            router=self.router,
            tracer=self.tracing.tracer,
            ledger=self.ledger,
            conversation_id=conv.conversation_id if conv else f"extract:{session_id}",
            customer_id=conv.customer_id if conv else "system:extractor",
            span_attrs={"app.session": session_id},
        )
        resp = await caller(
            ModelRequest(
                route="extract_estimate",
                system=EXTRACT_ESTIMATE,
                tools=[],
                output_schema=ESTIMATE_SCHEMA,
                messages=[{"role": "user", "content": wrap("uploaded_document", text)}],
            ),
            "estimate_extractor",
        )
        try:
            data = json.loads(resp.text)
            data["line_items"] = [tuple(x) for x in data.get("line_items") or [] if len(x) == 2]
            return EstimateFields.model_validate(data)
        except (ValueError, TypeError):
            return EstimateFields()

    async def __aenter__(self) -> System:
        self._stack = AsyncExitStack()
        if self.settings.in_process_mcp:
            for name, srv in self.servers.items():
                self.apps[name] = srv.streamable_http_app(stateless_http=True, json_response=True)
                await self._stack.enter_async_context(srv.session_manager.run())
            self.connect = http_connector(self.apps, self.settings.service_urls)
        else:
            self.connect = http_connector(None, self.settings.service_urls)
        await self._discover_tools()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    async def _discover_tools(self) -> None:
        """Read tool definitions from each MCP server, as a model would see them."""
        login = self.idp.login("maya", DEV_PASSWORD)  # any valid session can list tools
        for name, scopes in SERVICE_SCOPES.items():
            token = self.idp.exchange(login, self.settings.service_urls[name], set(scopes))
            async with self.connect(name, token) as client:
                listed = await client.list_tools()
            self.catalog.by_server[name] = [
                {"name": tool.name, "description": (tool.description or "").strip(), "input_schema": tool.input_schema}
                for tool in listed.tools
            ]

    # Convenience for tests, the CLI and the evaluation runner -------------------------------------
    def login(self, username: str) -> str:
        return self.idp.login(username, DEV_PASSWORD)

    def step_up(self, session_token: str) -> str:
        claims = self.idp.decode(session_token, "chat-api")
        self.idp.start_step_up(session_token)
        return self.idp.complete_step_up(session_token, self.otp_outbox[claims["sub"]])

    def upload(self, session_token: str, filename: str, media_type: str, content: bytes) -> str:
        from .api.edge import scan_upload

        claims = self.idp.decode(session_token, "chat-api")
        clean = scan_upload(filename, media_type, content, self.settings.max_upload_bytes)
        return self.backends.documents.put(claims["sub"], filename, media_type, content, scanned_clean=clean)
