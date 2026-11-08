"""Step 7: token exchange binds tokens to one service; step-up is time-limited."""

import pytest

from claims_assistant.identity.idp import STEP_UP_VALID_S, AuthError, DevIdentityProvider


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def idp():
    sent = {}
    clock = Clock()
    p = DevIdentityProvider(
        users={"maya": ("cust-1", "pw")},
        audience_scopes={"claims": {"claim:read", "claim:submit"}, "payments": {"pay"}},
        otp_sender=lambda sub, code: sent.__setitem__(sub, code),
        clock=clock,
    )
    p.sent, p.test_clock = sent, clock
    return p


def test_login_rejects_bad_password(idp):
    with pytest.raises(AuthError):
        idp.login("maya", "wrong")


def test_exchanged_token_is_bound_to_its_audience(idp):
    session = idp.login("maya", "pw")
    token = idp.exchange(session, "claims", {"claim:read"})
    assert idp.decode(token, "claims")["sub"] == "cust-1"
    with pytest.raises(AuthError):
        idp.decode(token, "payments")
    with pytest.raises(AuthError):
        idp.decode(token, "chat-api")


def test_exchange_refuses_scopes_outside_the_audience(idp):
    session = idp.login("maya", "pw")
    with pytest.raises(AuthError):
        idp.exchange(session, "claims", {"pay"})


def test_exchange_requires_a_chat_session(idp):
    session = idp.login("maya", "pw")
    service_token = idp.exchange(session, "claims", {"claim:read"})
    with pytest.raises(AuthError):
        idp.exchange(service_token, "payments", {"pay"})


def test_step_up_flow_and_expiry(idp):
    session = idp.login("maya", "pw")
    idp.start_step_up(session)
    with pytest.raises(AuthError):
        idp.complete_step_up(session, "000000" if idp.sent["cust-1"] != "000000" else "111111")
    stepped = idp.complete_step_up(session, idp.sent["cust-1"])
    token = idp.exchange(stepped, "claims", {"claim:submit"})
    assert idp.has_recent_step_up(idp.decode(token, "claims"))
    idp.test_clock.now += STEP_UP_VALID_S + 1
    assert not idp.has_recent_step_up(idp.decode(stepped, "chat-api") | {"exp": 0})


def test_otp_attempts_are_limited(idp):
    session = idp.login("maya", "pw")
    idp.start_step_up(session)
    wrong = "000000" if idp.sent["cust-1"] != "000000" else "111111"
    for _ in range(3):
        with pytest.raises(AuthError):
            idp.complete_step_up(session, wrong)
    with pytest.raises(AuthError):
        idp.complete_step_up(session, idp.sent["cust-1"])  # challenge burned after too many attempts


def test_tokens_expire(idp):
    session = idp.login("maya", "pw")
    idp.test_clock.now += 31 * 60
    with pytest.raises(AuthError):
        idp.exchange(session, "claims", {"claim:read"})
