"""Policy MCP server: the customer's policies, cover summaries and policy wording."""

from __future__ import annotations

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..authz.policy import Action, check
from ..systems.backends import BackendUnavailable, NotFound
from .common import (
    NOT_LINKED,
    UNAVAILABLE,
    JwtTokenVerifier,
    ServerContext,
    audited,
    auth_settings,
    current_caller,
    require_scope,
    tool_error,
)

SERVICE = "policy-mcp"


def build(ctx: ServerContext) -> MCPServer:
    url = ctx.urls[SERVICE]
    srv = MCPServer(
        SERVICE,
        instructions="Read-only access to the signed-in customer's insurance policies.",
        token_verifier=JwtTokenVerifier(ctx.idp, url),
        auth=auth_settings(url),
    )
    policies = ctx.backends.policies

    @srv.tool()
    async def list_my_policies() -> dict:
        """List the policies the signed-in customer is a party to, with their role on each.
        Takes no arguments: the customer is identified by the session, never by a policy number."""
        caller = current_caller(ctx)
        if err := require_scope(caller, "policy:read"):
            return err
        try:
            found = await policies.policies_for(caller.customer_id)
        except BackendUnavailable:
            return tool_error("unavailable", UNAVAILABLE.format(system="policy"))
        out = []
        for p in found:
            role = next(x.role for x in p.parties if x.customer_id == caller.customer_id)
            out.append(
                {
                    "policy_id": p.policy_id,
                    "product": p.product,
                    "status": p.status,
                    "insured": p.vehicle or p.address_district,
                    "role": role,
                    "renews": p.end.isoformat(),
                }
            )
        return {"policies": out}

    @srv.tool()
    async def get_cover_summary(
        policy_id: Annotated[str, Field(description="A policy_id returned by list_my_policies")],
    ) -> dict:
        """What the policy covers: each cover item, whether it is included, its excess and conditions."""
        caller = current_caller(ctx)
        if err := require_scope(caller, "policy:read"):
            return err
        try:
            policy = await policies.get(policy_id)
        except NotFound:
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        except BackendUnavailable:
            return tool_error("unavailable", UNAVAILABLE.format(system="policy"))
        decision = audited(
            ctx,
            SERVICE,
            caller,
            Action.POLICY_READ_SUMMARY,
            policy_id,
            check(caller, Action.POLICY_READ_SUMMARY, policy),
        )
        if not decision.allowed:
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        return {
            "policy_id": policy.policy_id,
            "product": policy.product,
            "cover": [c.model_dump() for c in policy.cover],
        }

    @srv.tool()
    async def search_policy_wording(
        policy_id: Annotated[str, Field(description="A policy_id returned by list_my_policies")],
        query: Annotated[str, Field(description="What the customer wants to know, e.g. 'cracked windscreen'")],
        max_sections: Annotated[int, Field(ge=1, le=5)] = 3,
    ) -> dict:
        """Return the sections of the policy wording most relevant to the query, with section numbers
        to cite. Use this to answer coverage questions; never answer coverage from memory."""
        caller = current_caller(ctx)
        if err := require_scope(caller, "policy:read"):
            return err
        try:
            policy = await policies.get(policy_id)
            decision = audited(
                ctx, SERVICE, caller, Action.POLICY_READ, policy_id, check(caller, Action.POLICY_READ, policy)
            )
            if not decision.allowed:
                return tool_error("not_found_or_not_permitted", NOT_LINKED)
            sections = await policies.wording(policy.wording_version)
        except NotFound:
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        except BackendUnavailable:
            return tool_error("unavailable", UNAVAILABLE.format(system="policy"))
        terms = {t for t in query.lower().replace("-", " ").split() if len(t) > 2}
        scored = []
        for number, title, text in sections:
            hay = f"{title} {text}".lower()
            score = sum(hay.count(t) for t in terms) + 3 * sum(t in title.lower() for t in terms)
            if score:
                scored.append((score, number, title, text))
        scored.sort(reverse=True)
        return {
            "wording_version": policy.wording_version,
            "sections": [{"section": n, "title": t, "text": x} for _, n, t, x in scored[:max_sections]],
        }

    return srv
