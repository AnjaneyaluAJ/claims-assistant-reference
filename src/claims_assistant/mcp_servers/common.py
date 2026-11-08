"""Shared plumbing for the domain MCP servers (Step 5).

The rule this module enforces: identity comes from the verified bearer token that the
agent runtime attaches to the HTTP request, never from tool arguments. The model chooses
the tool and, say, the claim ID. The token decides whose data comes back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp_types import CallToolResult, TextContent

from ..authz.audit import AuditLog
from ..authz.policy import Caller, Decision
from ..identity.idp import ISSUER, AuthError, DevIdentityProvider
from ..pii.vault import TokenVault
from ..systems.backends import Backends


class JwtTokenVerifier:
    """MCP ``TokenVerifier`` that accepts only tokens minted for this server's audience."""

    def __init__(self, idp: DevIdentityProvider, resource_url: str) -> None:
        self.idp = idp
        self.resource_url = resource_url

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            claims = self.idp.decode(token, audience=self.resource_url)
        except AuthError:
            return None
        return AccessToken(
            token=token,
            client_id=claims.get("act", {}).get("sub", "unknown"),
            scopes=claims.get("scope", "").split(),
            expires_at=claims["exp"],
            resource=self.resource_url,
            subject=claims["sub"],
            claims=claims,
        )


def auth_settings(resource_url: str) -> AuthSettings:
    """The bearer middleware checks issuer, signature, expiry and audience for every request.
    Scopes are checked per tool (``require_scope``), so each call needs only its own scope."""
    return AuthSettings(issuer_url=ISSUER, resource_server_url=resource_url, validate_token_resource=True)


@dataclass
class ServerContext:
    backends: Backends
    idp: DevIdentityProvider
    audit: AuditLog
    vault: TokenVault
    extractor: Any = None  # documents server only: callable that extracts typed fields
    urls: dict[str, str] = field(default_factory=dict)


def current_caller(ctx: ServerContext) -> Caller:
    token = get_access_token()
    if token is None or token.claims is None:  # the bearer middleware should make this unreachable
        raise PermissionError("unauthenticated")
    claims = token.claims
    return Caller(
        customer_id=claims["sub"],
        scopes=frozenset(token.scopes),
        stepped_up=ctx.idp.has_recent_step_up(claims),
        session_id=claims.get("sid", ""),
    )


def tool_error(code: str, message: str, **extra: Any) -> CallToolResult:
    """A structured error the model can explain. Never a stack trace, never internal detail."""
    payload = {"error": {"code": code, "message": message, **extra}}
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))], is_error=True)


def require_scope(caller: Caller, scope: str) -> CallToolResult | None:
    if scope not in caller.scopes:
        return tool_error("insufficient_scope", "This action is not available in the current session.")
    return None


def audited(
    ctx: ServerContext, service: str, caller: Caller, action: str, resource: str, decision: Decision
) -> Decision:
    ctx.audit.write(
        subject=caller.customer_id,
        action=action,
        resource=resource,
        allowed=decision.allowed,
        reason=decision.reason,
        service=service,
        session_id=caller.session_id,
    )
    return decision


NOT_LINKED = "That record isn't linked to your account."  # does not confirm whether it exists
UNAVAILABLE = "The {system} system is temporarily unavailable."
