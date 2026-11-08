"""A small development identity provider (Step 7).

It stands in for the insurer's real OpenID Connect provider and implements the three
things the architecture depends on:

1. Login: a session token for the chat API (audience ``chat-api``).
2. Token exchange (shaped after RFC 8693): the chat API trades the session token for a
   short-lived token scoped to one downstream MCP server. A token for the claims server
   is useless against the payments server.
3. Step-up: a one-time code to the registered phone raises the session's assurance level
   for a few minutes. Write tools reject tokens without a recent step-up.

Do not use this in production. Use your IdP's token exchange and MFA features.
"""

from __future__ import annotations

import hmac
import secrets
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = "https://idp.example-insurer.test"
CHAT_AUDIENCE = "chat-api"
STEP_UP_ACR = "urn:example-insurer:acr:step-up"
SESSION_TTL_S = 30 * 60
SERVICE_TOKEN_TTL_S = 5 * 60
STEP_UP_VALID_S = 5 * 60
OTP_TTL_S = 5 * 60


class AuthError(Exception):
    pass


@dataclass
class _PendingOtp:
    code: str
    expires_at: float
    attempts: int = 0


@dataclass
class DevIdentityProvider:
    """``users`` maps username to (customer_id, password). ``audience_scopes`` maps each
    downstream audience to the scopes the chat API may request for it."""

    users: dict[str, tuple[str, str]]
    audience_scopes: dict[str, set[str]]
    otp_sender: Callable[[str, str], None] = lambda customer_id, code: None
    clock: Callable[[], float] = time.time
    _key: rsa.RSAPrivateKey = field(
        default_factory=lambda: rsa.generate_private_key(public_exponent=65537, key_size=2048)
    )
    _pending_otp: dict[str, _PendingOtp] = field(default_factory=dict)
    _kid: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # -- keys -------------------------------------------------------------------------------------
    @property
    def public_key(self):
        return self._key.public_key()

    def _sign(self, claims: dict) -> str:
        return jwt.encode(claims, self._key, algorithm="RS256", headers={"kid": self._kid})

    def decode(self, token: str, audience: str) -> dict:
        try:
            claims = jwt.decode(
                token,
                self.public_key,
                algorithms=["RS256"],
                audience=audience,
                issuer=ISSUER,
                options={"require": ["exp", "iat", "sub", "aud", "iss"], "verify_exp": False, "verify_iat": False},
            )
        except jwt.PyJWTError as exc:
            raise AuthError(f"invalid token: {exc}") from exc
        # Time checks use the provider's clock, so expiry is testable without sleeping.
        if claims["exp"] <= self.clock():
            raise AuthError("invalid token: expired")
        return claims

    # -- login ------------------------------------------------------------------------------------
    def login(self, username: str, password: str) -> str:
        record = self.users.get(username)
        if record is None or not hmac.compare_digest(record[1], password):
            raise AuthError("invalid credentials")
        now = int(self.clock())
        return self._sign(
            {
                "iss": ISSUER,
                "sub": record[0],
                "aud": CHAT_AUDIENCE,
                "iat": now,
                "exp": now + SESSION_TTL_S,
                "auth_time": now,
                "acr": "urn:example-insurer:acr:password",
                "sid": uuid.uuid4().hex,
            }
        )

    # -- step-up ----------------------------------------------------------------------------------
    def start_step_up(self, session_token: str) -> None:
        claims = self.decode(session_token, CHAT_AUDIENCE)
        code = f"{secrets.randbelow(10**6):06d}"
        self._pending_otp[claims["sid"]] = _PendingOtp(code=code, expires_at=self.clock() + OTP_TTL_S)
        self.otp_sender(claims["sub"], code)

    def complete_step_up(self, session_token: str, code: str) -> str:
        claims = self.decode(session_token, CHAT_AUDIENCE)
        pending = self._pending_otp.get(claims["sid"])
        if pending is None or self.clock() > pending.expires_at:
            raise AuthError("no active step-up challenge")
        pending.attempts += 1
        if pending.attempts > 3:
            del self._pending_otp[claims["sid"]]
            raise AuthError("too many attempts")
        if not hmac.compare_digest(pending.code, code):
            raise AuthError("incorrect code")
        del self._pending_otp[claims["sid"]]
        now = int(self.clock())
        return self._sign({**claims, "iat": now, "exp": now + SESSION_TTL_S, "acr": STEP_UP_ACR, "step_up_at": now})

    # -- token exchange ---------------------------------------------------------------------------
    def exchange(self, session_token: str, audience: str, scopes: set[str]) -> str:
        """Trade a chat session token for a short-lived, audience-bound service token."""
        claims = self.decode(session_token, CHAT_AUDIENCE)
        allowed = self.audience_scopes.get(audience)
        if allowed is None:
            raise AuthError(f"unknown audience {audience}")
        if not scopes <= allowed:
            raise AuthError(f"scopes {sorted(scopes - allowed)} not allowed for {audience}")
        now = int(self.clock())
        out = {
            "iss": ISSUER,
            "sub": claims["sub"],
            "aud": audience,
            "iat": now,
            "exp": min(now + SERVICE_TOKEN_TTL_S, claims["exp"]),
            "scope": " ".join(sorted(scopes)),
            "act": {"sub": CHAT_AUDIENCE},  # the chat API acts on behalf of the customer
            "acr": claims.get("acr"),
            "sid": claims["sid"],
            "jti": uuid.uuid4().hex,
        }
        if "step_up_at" in claims:
            out["step_up_at"] = claims["step_up_at"]
        return self._sign(out)

    def has_recent_step_up(self, claims: dict) -> bool:
        at = claims.get("step_up_at")
        return claims.get("acr") == STEP_UP_ACR and at is not None and self.clock() - at <= STEP_UP_VALID_S
