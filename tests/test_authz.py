"""The authorization table from Step 8, one assertion per cell."""

from datetime import date

import pytest

from claims_assistant.authz.policy import Action, Caller, check
from claims_assistant.domain.models import Claim, CoverItem, Delegation, Party, Policy, Role


def policy(delegation=None) -> Policy:
    return Policy(
        policy_id="MTR-1",
        product="motor",
        status="active",
        start=date(2026, 1, 1),
        end=date(2026, 12, 31),
        parties=[
            Party(customer_id="holder", role=Role.POLICYHOLDER),
            Party(customer_id="driver", role=Role.NAMED_DRIVER),
            Party(customer_id="delegate", role=Role.DELEGATE),
            Party(customer_id="broker", role=Role.BROKER),
        ],
        delegation=delegation or Delegation(),
        cover=[CoverItem(code="x", title="X", included=True)],
        wording_version="v",
    )


def claim(party="holder") -> Claim:
    return Claim(
        claim_id="CLM-00001",
        policy_id="MTR-1",
        incident_party=party,
        loss_type="accidental_damage",
        incident_date=date(2026, 5, 1),
        status="registered",
        next_step="",
    )


def caller(who: str, stepped_up=False) -> Caller:
    return Caller(customer_id=who, scopes=frozenset(), stepped_up=stepped_up, session_id="s")


@pytest.mark.parametrize(
    "who,incident,allowed",
    [
        ("holder", "holder", True),
        ("holder", "driver", True),
        ("driver", "driver", True),
        ("driver", "holder", False),
        ("delegate", "holder", True),
        ("broker", "holder", True),
        ("stranger", "holder", False),
    ],
)
def test_claim_read(who, incident, allowed):
    assert check(caller(who), Action.CLAIM_READ, policy(), claim(incident)).allowed is allowed


def test_named_driver_sees_all_claims_when_delegated():
    p = policy(Delegation(named_drivers_see_all_claims=True))
    assert check(caller("driver"), Action.CLAIM_READ, p, claim("holder")).allowed


@pytest.mark.parametrize(
    "who,stepped_up,allowed,needs_step_up",
    [
        ("holder", False, False, True),
        ("holder", True, True, False),
        ("driver", True, True, False),
        ("delegate", True, False, False),
        ("broker", True, False, False),
    ],
)
def test_claim_submit(who, stepped_up, allowed, needs_step_up):
    d = check(caller(who, stepped_up), Action.CLAIM_SUBMIT, policy())
    assert (d.allowed, d.needs_step_up) == (allowed, needs_step_up)


def test_delegate_submit_when_delegation_covers_claims():
    p = policy(Delegation(delegate_can_submit_claims=True))
    assert check(caller("delegate", True), Action.CLAIM_SUBMIT, p).allowed


@pytest.mark.parametrize(
    "who,stepped_up,allowed",
    [
        ("holder", True, True),
        ("holder", False, False),
        ("driver", True, False),
        ("delegate", True, False),
        ("broker", True, False),
    ],
)
def test_payout_update(who, stepped_up, allowed):
    assert check(caller(who, stepped_up), Action.PAYOUT_UPDATE, policy()).allowed is allowed


@pytest.mark.parametrize("who,allowed", [("holder", True), ("driver", False), ("delegate", True), ("broker", True)])
def test_policy_read_full_wording(who, allowed):
    assert check(caller(who), Action.POLICY_READ, policy()).allowed is allowed


def test_named_driver_gets_cover_summary():
    assert check(caller("driver"), Action.POLICY_READ_SUMMARY, policy()).allowed


def test_claim_from_another_policy_is_refused():
    other = claim()
    other.policy_id = "MTR-2"
    assert not check(caller("holder"), Action.CLAIM_READ, policy(), other).allowed
