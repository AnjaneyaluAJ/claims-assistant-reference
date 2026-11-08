"""Relationship-based authorization (Step 8).

Access is decided by the caller's relationship to the policy and the claim, evaluated
inside the MCP server before any data leaves it. The table in the blog is encoded here
once, and every server calls ``check``:

    Action                  Policyholder        Named driver                 Delegate                  Broker
    claim:read              yes                 own incidents only*          yes                       yes
    claim:submit            yes, step-up        own incidents, step-up       if delegation, step-up    no
    payout_account:update   yes, step-up        no                           no                        no
    policy:read             yes                 cover summary only           yes                       yes

    * unless the policyholder has enabled ``named_drivers_see_all_claims``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..domain.models import Claim, Policy, Role


class Action(StrEnum):
    CLAIM_READ = "claim:read"
    CLAIM_SUBMIT = "claim:submit"
    PAYOUT_UPDATE = "payout_account:update"
    POLICY_READ = "policy:read"
    POLICY_READ_SUMMARY = "policy:read_summary"


@dataclass(frozen=True)
class Caller:
    """Identity taken from a verified service token. Never built from tool arguments."""

    customer_id: str
    scopes: frozenset[str]
    stepped_up: bool
    session_id: str


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    needs_step_up: bool = False


def _role(caller: Caller, policy: Policy) -> Role | None:
    for party in policy.parties:
        if party.customer_id == caller.customer_id:
            return party.role
    return None


def check(caller: Caller, action: Action, policy: Policy, claim: Claim | None = None) -> Decision:
    role = _role(caller, policy)
    if role is None:
        return Decision(False, "no_relationship")
    if claim is not None and claim.policy_id != policy.policy_id:
        return Decision(False, "claim_policy_mismatch")

    match action:
        case Action.POLICY_READ_SUMMARY:
            return Decision(True, f"{role}:summary")

        case Action.POLICY_READ:
            if role == Role.NAMED_DRIVER:
                return Decision(False, "named_driver_summary_only")
            return Decision(True, str(role))

        case Action.CLAIM_READ:
            if claim is None:
                return Decision(False, "claim_required")
            if role in (Role.POLICYHOLDER, Role.DELEGATE, Role.BROKER):
                return Decision(True, str(role))
            if claim.incident_party == caller.customer_id:
                return Decision(True, "named_driver:own_incident")
            if policy.delegation.named_drivers_see_all_claims:
                return Decision(True, "named_driver:delegated")
            return Decision(False, "named_driver:not_own_incident")

        case Action.CLAIM_SUBMIT:
            if role == Role.BROKER:
                return Decision(False, "broker_cannot_submit")
            if role == Role.DELEGATE and not policy.delegation.delegate_can_submit_claims:
                return Decision(False, "delegation_excludes_claims")
            if not caller.stepped_up:
                return Decision(False, "step_up_required", needs_step_up=True)
            return Decision(True, str(role))

        case Action.PAYOUT_UPDATE:
            if role != Role.POLICYHOLDER:
                return Decision(False, "policyholder_only")
            if not caller.stepped_up:
                return Decision(False, "step_up_required", needs_step_up=True)
            return Decision(True, str(role))

    return Decision(False, "unknown_action")
