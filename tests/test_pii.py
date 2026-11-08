"""Step 11: detection, reversible tokenization, and who may resolve a placeholder."""

import pytest

from claims_assistant.pii.detector import detect
from claims_assistant.pii.vault import TokenVault, VaultAccessDenied, redact


@pytest.mark.parametrize(
    "text,kinds",
    [
        ("account 41926630, sort code 20-45-77", ["account_number", "sort_code"]),
        ("card 4111 1111 1111 1111", ["card_number"]),
        ("card 4111 1111 1111 1112", []),  # fails the Luhn check
        ("call me on 07700 900123", ["phone"]),
        ("maya@example.com", ["email"]),
        ("policy MTR-5530217 and claim CLM-77120", []),  # our own references are not personal data
        ("estimate total 1,540 on 2026-11-05", []),
    ],
)
def test_detector(text, kinds):
    assert [d.kind for d in detect(text)] == kinds


def test_tokenize_and_resolve_only_by_payments_in_same_session():
    vault = TokenVault()
    text, found = vault.tokenize("pay into 41926630 sort 20-45-77", "sess-1")
    assert "41926630" not in text and "20-45-77" not in text
    acct = next(p for p in text.split() if p.startswith("<pii:account_number"))
    assert vault.resolve(acct, service="payments-mcp", session_id="sess-1") == "41926630"
    with pytest.raises(VaultAccessDenied):
        vault.resolve(acct, service="claims-mcp", session_id="sess-1")
    with pytest.raises(VaultAccessDenied):
        vault.resolve(acct, service="payments-mcp", session_id="sess-2")


def test_same_value_same_placeholder_within_a_session_only():
    vault = TokenVault()
    a, _ = vault.tokenize("41926630", "s1")
    b, _ = vault.tokenize("41926630", "s1")
    c, _ = vault.tokenize("41926630", "s2")
    assert a == b and a != c


def test_mask_for_display_shows_last_four_only():
    vault = TokenVault()
    text, _ = vault.tokenize("41926630", "s1")
    assert vault.mask_for_display(text) == "account number ending 6630"


def test_one_way_redaction_for_logs():
    assert redact("acct 41926630 email a@b.co") == "acct [account_number] email [email]"
