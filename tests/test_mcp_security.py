"""Steps 2, 5, 6, 7 and 8 against the real MCP servers, over HTTP, with real tokens."""

import uuid

from conftest import gateway_for

IDENTITY_OR_MONEY_FIELDS = {
    "customer_id",
    "user_id",
    "customer",
    "policyholder",
    "email",
    "amount",
    "approved",
    "approve",
    "payout_amount",
    "settlement_amount",
    "decision",
    "value",
}


def _all_properties(schema: dict) -> set[str]:
    props = set(schema.get("properties", {}))
    for sub in schema.get("properties", {}).values():
        if isinstance(sub, dict):
            props |= _all_properties(sub)
    for sub in schema.get("$defs", {}).values():
        props |= _all_properties(sub)
    return props


async def test_tool_contracts_take_no_identity_and_no_money_arguments(system):
    """The rule that closes the Step 6 hole, checked over every tool the servers expose."""
    for server, tools in system.catalog.by_server.items():
        for tool in tools:
            bad = _all_properties(tool["input_schema"]) & IDENTITY_OR_MONEY_FIELDS
            assert not bad, f"{server}.{tool['name']} accepts {bad}"
            assert "approve" not in tool["name"] and "deny" not in tool["name"]


async def test_requests_without_a_valid_token_are_rejected(system):
    gw = gateway_for(system, system.login("maya"))
    gw._tokens[("claims-mcp", "claim:read")] = "not-a-token"
    out = await gw.call("list_my_claims", {})
    assert out.error_code == "unauthorized"
    assert system.breakers["claims-mcp"].state == "closed"  # an auth failure is not an outage


async def test_token_for_one_server_is_useless_against_another(system):
    session = system.login("maya")
    payments_token = system.idp.exchange(
        session, system.settings.service_urls["payments-mcp"], {"payout_account:update"}
    )
    gw = gateway_for(system, session)
    gw._tokens[("claims-mcp", "claim:read")] = payments_token
    out = await gw.call("get_claim_status", {"claim_id": "CLM-77120"})
    assert out.error_code == "unauthorized"


async def test_other_customers_claim_is_refused_and_audited(system):
    gw = gateway_for(system, system.login("maya"))
    out = await gw.call("get_claim_status", {"claim_id": "CLM-88213"})
    assert out.error_code == "not_found_or_not_permitted"
    missing = await gw.call("get_claim_status", {"claim_id": "CLM-11111"})
    assert missing.error_message == out.error_message  # does not reveal whether the claim exists
    denial = system.audit.denials()[-2]
    assert (denial["subject"], denial["resource"], denial["reason"]) == ("cust-1001", "CLM-88213", "no_relationship")
    assert system.audit.verify_chain()


async def test_list_my_claims_only_returns_own_claims(system):
    out = await gateway_for(system, system.login("tom")).call("list_my_claims", {})
    assert [c["claim_id"] for c in out.data["claims"]] == ["CLM-88213"]


async def test_submit_requires_step_up_and_is_idempotent(system):
    session = system.login("maya")
    gw = gateway_for(system, session)
    draft = await gw.call(
        "create_claim_draft",
        {
            "policy_id": "MTR-5530217",
            "loss_type": "windscreen",
            "incident_date": "2026-11-05",
            "description": "Stone chip cracked the windscreen.",
        },
    )
    key = uuid.uuid4().hex
    refused = await gw.call("submit_claim", {"draft_id": draft.data["draft_id"], "idempotency_key": key})
    assert refused.error_code == "step_up_required"

    gw = gateway_for(system, system.step_up(session))
    first = await gw.call("submit_claim", {"draft_id": draft.data["draft_id"], "idempotency_key": key})
    again = await gw.call("submit_claim", {"draft_id": draft.data["draft_id"], "idempotency_key": key})
    assert first.ok and again.ok
    assert first.data["claim_id"] == again.data["claim_id"]
    assert (first.data["created"], again.data["created"]) == (True, False)
    assert system.backends.claims.submissions == 1


async def test_cannot_submit_someone_elses_draft(system):
    gw = gateway_for(system, system.login("maya"))
    draft = await gw.call(
        "create_claim_draft",
        {
            "policy_id": "MTR-5530217",
            "loss_type": "theft",
            "incident_date": "2026-11-01",
            "description": "Car stolen overnight.",
        },
    )
    tom = gateway_for(system, system.step_up(system.login("tom")))
    out = await tom.call("submit_claim", {"draft_id": draft.data["draft_id"], "idempotency_key": uuid.uuid4().hex})
    assert out.error_code == "not_found_or_not_permitted"


async def test_cannot_draft_on_a_policy_you_are_not_on(system):
    gw = gateway_for(system, system.login("maya"))
    out = await gw.call(
        "create_claim_draft",
        {
            "policy_id": "MTR-4471902",
            "loss_type": "theft",
            "incident_date": "2026-11-01",
            "description": "Not my policy at all.",
        },
    )
    assert not out.ok


async def test_documents_are_private_to_their_uploader(system):
    maya = system.login("maya")
    doc = system.upload(maya, "e.txt", "text/plain", b"Riverside Motors\nTOTAL .... 100\n")
    out = await gateway_for(system, system.login("tom")).call("extract_estimate", {"document_id": doc})
    assert out.error_code == "not_found_or_not_permitted"


async def test_payout_requires_policyholder_and_resolves_placeholders_server_side(system):
    session = system.step_up(system.login("maya"))
    claims = system.idp.decode(session, "chat-api")
    text, _ = system.vault.tokenize("41926630 20-45-77", claims["sid"])
    acct, sc = text.split()
    out = await gateway_for(system, session).call("update_payout_account", {"sort_code": sc, "account_number": acct})
    assert out.ok and out.data == {"updated": True, "account_ending": "6630"}
    assert system.backends.payments.payout_accounts["cust-1001"] == ("20-45-77", "41926630")


async def test_raw_bank_details_are_rejected_by_the_contract(system):
    session = system.step_up(system.login("maya"))
    out = await gateway_for(system, session).call(
        "update_payout_account", {"sort_code": "20-45-77", "account_number": "41926630"}
    )
    assert out.error_code == "invalid_argument"
    assert "cust-1001" not in system.backends.payments.payout_accounts
