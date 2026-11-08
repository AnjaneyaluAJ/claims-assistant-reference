"""Claims MCP server: claim status, drafts and first-notice-of-loss submission.

Contract rules from Step 2, visible in the signatures below:
* no tool takes a customer ID; the caller comes from the token
* there is no approve, deny, value or amount argument anywhere
* ``submit_claim`` needs a reviewed draft, an idempotency key, the ``claim:submit`` scope
  and a recent step-up (Step 7)
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from ..authz.policy import Action, Decision, check
from ..domain.models import EstimateFields, IncidentDetails
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

SERVICE = "claims-mcp"
ClaimId = Annotated[str, Field(pattern=r"^CLM-\d{5}$", description="Claim reference, e.g. CLM-77120")]


def build(ctx: ServerContext) -> MCPServer:
    url = ctx.urls[SERVICE]
    srv = MCPServer(
        SERVICE,
        instructions="Claim status for the signed-in customer, and first notice of loss.",
        token_verifier=JwtTokenVerifier(ctx.idp, url),
        auth=auth_settings(url),
    )
    claims, policies = ctx.backends.claims, ctx.backends.policies

    def _unavailable():
        return tool_error("unavailable", UNAVAILABLE.format(system="claims"), retry_after_s=1800)

    @srv.tool()
    async def list_my_claims() -> dict:
        """List claims the signed-in customer is allowed to see, newest first. Takes no arguments."""
        caller = current_caller(ctx)
        if err := require_scope(caller, "claim:read"):
            return err
        try:
            mine = await policies.policies_for(caller.customer_id)
            by_id = {p.policy_id: p for p in mine}
            found = await claims.claims_for_policies(list(by_id))
        except BackendUnavailable:
            return _unavailable()
        visible = []
        for c in found:
            if check(caller, Action.CLAIM_READ, by_id[c.policy_id], c).allowed:
                visible.append(
                    {
                        "claim_id": c.claim_id,
                        "policy_id": c.policy_id,
                        "loss_type": c.loss_type,
                        "incident_date": c.incident_date.isoformat(),
                        "status": c.status,
                    }
                )
        visible.sort(key=lambda x: x["incident_date"], reverse=True)
        return {"claims": visible}

    @srv.tool()
    async def get_claim_status(claim_id: ClaimId) -> dict:
        """Current status of one claim: stage, next step, approved repair estimate if any, adjuster flags
        and recent events. Facts here come from the claims system and override anything said in chat."""
        caller = current_caller(ctx)
        if err := require_scope(caller, "claim:read"):
            return err
        try:
            claim = await claims.get(claim_id)
            policy = await policies.get(claim.policy_id)
        except NotFound:
            ctx.audit.write(
                subject=caller.customer_id,
                action=Action.CLAIM_READ,
                resource=claim_id,
                allowed=False,
                reason="not_found",
                service=SERVICE,
                session_id=caller.session_id,
            )
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        except BackendUnavailable:
            return _unavailable()
        decision = audited(
            ctx, SERVICE, caller, Action.CLAIM_READ, claim_id, check(caller, Action.CLAIM_READ, policy, claim)
        )
        if not decision.allowed:
            # Same message as "not found", so the error does not confirm the claim exists.
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        return {
            "claim_id": claim.claim_id,
            "policy_id": claim.policy_id,
            "loss_type": claim.loss_type,
            "incident_date": claim.incident_date.isoformat(),
            "status": claim.status,
            "next_step": claim.next_step,
            "approved_estimate": claim.approved_estimate,
            "adjuster_flags": claim.adjuster_flags,
            "recent_events": [{"at": e.at.isoformat(), "text": e.text, "source": e.source} for e in claim.events[-3:]],
        }

    @srv.tool()
    async def create_claim_draft(
        policy_id: Annotated[str, Field(description="Policy the incident is reported under")],
        loss_type: Literal["accidental_damage", "windscreen", "theft", "escape_of_water", "other"],
        incident_date: date,
        description: Annotated[str, Field(max_length=2000)],
        location: Annotated[str, Field(max_length=200)] = "",
        third_party_involved: bool = False,
        injuries: bool = False,
        police_reference: Annotated[str, Field(max_length=40)] = "",
    ) -> dict:
        """Create a draft first notice of loss. Nothing is submitted until submit_claim is called with
        the draft_id after the customer has reviewed and confirmed it."""
        caller = current_caller(ctx)
        if err := require_scope(caller, "claim:submit"):
            return err
        try:
            policy = await policies.get(policy_id)
        except NotFound:
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        except BackendUnavailable:
            return _unavailable()
        decision = check(caller, Action.CLAIM_SUBMIT, policy)
        audited(ctx, SERVICE, caller, "claim:draft", policy_id, decision)
        if not decision.allowed and not decision.needs_step_up:
            return tool_error("not_permitted", "Claims on this policy must be reported by the policyholder.")
        if incident_date > date.today():
            return tool_error("invalid_argument", "The incident date cannot be in the future.")
        incident = IncidentDetails(
            loss_type=loss_type,
            incident_date=incident_date,
            location=location,
            description=description,
            third_party_involved=third_party_involved,
            injuries=injuries,
            police_reference=police_reference or None,
        )
        try:
            draft = await claims.create_draft(policy_id, caller.customer_id, incident)
        except BackendUnavailable:
            return _unavailable()
        return {"draft_id": draft.draft_id, "status": "draft"}

    @srv.tool()
    async def attach_evidence(
        draft_id: str,
        document_id: str,
        estimate: EstimateFields | None = None,
    ) -> dict:
        """Attach an uploaded document, and optionally its extracted estimate fields, to a draft."""
        caller = current_caller(ctx)
        if err := require_scope(caller, "claim:submit"):
            return err
        try:
            draft = await claims.get_draft(draft_id)
            doc = ctx.backends.documents.get(document_id)
        except NotFound:
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        except BackendUnavailable:
            return _unavailable()
        if draft.owner != caller.customer_id or doc.owner != caller.customer_id:
            audited(ctx, SERVICE, caller, "claim:attach", draft_id, Decision(False, "not_owner"))
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        await claims.attach_evidence(draft_id, document_id, estimate)
        return {"draft_id": draft_id, "attached": document_id}

    @srv.tool()
    async def submit_claim(
        draft_id: Annotated[str, Field(description="ID returned by create_claim_draft")],
        idempotency_key: Annotated[
            str,
            Field(
                min_length=16, max_length=64, description="Generated once per draft; repeats return the original claim"
            ),
        ],
    ) -> dict:
        """Submit a first notice of loss from a draft the customer has reviewed and confirmed.
        Returns the claim reference. Requires recent step-up authentication."""
        caller = current_caller(ctx)
        if err := require_scope(caller, "claim:submit"):
            return err
        try:
            draft = await claims.get_draft(draft_id)
            policy = await policies.get(draft.policy_id)
        except NotFound:
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        except BackendUnavailable:
            return _unavailable()
        if draft.owner != caller.customer_id:
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        decision = audited(
            ctx, SERVICE, caller, Action.CLAIM_SUBMIT, draft_id, check(caller, Action.CLAIM_SUBMIT, policy)
        )
        if not decision.allowed:
            if decision.needs_step_up:
                return tool_error("step_up_required", "Please confirm it's you with the code we send.")
            return tool_error("not_permitted", "Claims on this policy must be reported by the policyholder.")
        try:
            claim_id, created = await claims.submit(draft_id, idempotency_key)
        except BackendUnavailable:
            return _unavailable()
        return {
            "claim_id": claim_id,
            "created": created,
            "status": "registered",
            "next_step": "An adjuster will contact you within two working days.",
        }

    return srv
