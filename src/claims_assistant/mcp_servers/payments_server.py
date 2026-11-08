"""Payments MCP server: the only component allowed to resolve PII placeholders (Step 11).

The model passes placeholders such as ``<pii:account_number:3f9a1c2b>``. This server,
inside the trust boundary, resolves them from the vault and calls the payments system.
Real bank details never appear in a prompt, a model response, a log line or a trace.
"""

from __future__ import annotations

import re
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..authz.policy import Action, check
from ..pii.vault import PLACEHOLDER, VaultAccessDenied
from ..systems.backends import BackendUnavailable
from .common import (
    UNAVAILABLE,
    JwtTokenVerifier,
    ServerContext,
    audited,
    auth_settings,
    current_caller,
    require_scope,
    tool_error,
)

SERVICE = "payments-mcp"
Placeholder = Annotated[
    str, Field(pattern=PLACEHOLDER.pattern, description="The placeholder that appears in the conversation, as is")
]


def build(ctx: ServerContext) -> MCPServer:
    url = ctx.urls[SERVICE]
    srv = MCPServer(
        SERVICE,
        instructions="Payout account changes for the signed-in policyholder.",
        token_verifier=JwtTokenVerifier(ctx.idp, url),
        auth=auth_settings(url),
    )

    @srv.tool()
    async def update_payout_account(sort_code: Placeholder, account_number: Placeholder) -> dict:
        """Set the account that claim settlements are paid into. Pass the placeholders exactly as they
        appear in the conversation. Requires recent step-up authentication and the policyholder role."""
        caller = current_caller(ctx)
        if err := require_scope(caller, "payout_account:update"):
            return err
        policies = await ctx.backends.policies.policies_for(caller.customer_id)
        # Policyholder on at least one policy; the engine also enforces step-up.
        decisions = [check(caller, Action.PAYOUT_UPDATE, p) for p in policies]
        decision = next((d for d in decisions if d.allowed), decisions[0] if decisions else None)
        if decision is None:
            return tool_error("not_permitted", "Only a policyholder can change the payout account.")
        audited(ctx, SERVICE, caller, Action.PAYOUT_UPDATE, caller.customer_id, decision)
        if not decision.allowed:
            if decision.needs_step_up:
                return tool_error("step_up_required", "Please confirm it's you with the code we send.")
            return tool_error("not_permitted", "Only a policyholder can change the payout account.")
        try:
            sc = ctx.vault.resolve(sort_code, service=SERVICE, session_id=caller.session_id)
            acct = ctx.vault.resolve(account_number, service=SERVICE, session_id=caller.session_id)
        except VaultAccessDenied:
            return tool_error("invalid_argument", "Please type the account details again.")
        if not re.fullmatch(r"\d{2}[- ]?\d{2}[- ]?\d{2}", sc) or not re.fullmatch(r"\d{8}", acct):
            return tool_error("invalid_argument", "That doesn't look like a valid sort code and account number.")
        try:
            await ctx.backends.payments.set_payout_account(caller.customer_id, sc, acct)
        except BackendUnavailable:
            return tool_error("unavailable", UNAVAILABLE.format(system="payments"))
        return {"updated": True, "account_ending": acct[-4:]}

    return srv
