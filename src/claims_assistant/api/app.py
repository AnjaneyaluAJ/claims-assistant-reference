"""The chat API (FastAPI). Run with ``claims-assistant serve``.

Endpoints
  POST /auth/login             dev identity provider login (stand-in for the OIDC redirect)
  POST /auth/step-up/start     send a one-time code to the registered phone
  POST /auth/step-up/verify    exchange the code for a stepped-up session
  POST /documents              upload with type, size and malware checks
  POST /chat                   one turn
  GET  /healthz
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import Settings
from ..identity.idp import CHAT_AUDIENCE, AuthError
from ..system import System
from .edge import SlidingWindowLimiter, UploadRejected


class LoginIn(BaseModel):
    username: str
    password: str


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    attachments: list[str] = Field(default_factory=list, max_length=5)


class CodeIn(BaseModel):
    code: str = Field(pattern=r"^\d{6}$")


def create_app(system: System | None = None) -> FastAPI:
    holder: dict[str, System] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = system or System()
        async with s:
            holder["s"] = s
            yield

    app = FastAPI(title="Claims assistant", lifespan=lifespan)

    def sys_() -> System:
        return holder["s"]

    settings = system.settings if system else Settings()
    ip_limiter = SlidingWindowLimiter(settings.rate_limit_per_ip_per_min)
    customer_limiter = SlidingWindowLimiter(settings.rate_limit_per_customer_per_min)

    @app.middleware("http")
    async def edge(request: Request, call_next):
        if int(request.headers.get("content-length") or 0) > 6 * 1024 * 1024:
            return JSONResponse({"detail": "request too large"}, status_code=413)
        ip = request.client.host if request.client else "unknown"
        if not ip_limiter.allow(ip):
            return JSONResponse({"detail": "rate limited"}, status_code=429, headers={"Retry-After": "60"})
        return await call_next(request)

    def session(authorization: str = Header(...), s: System = Depends(sys_)) -> tuple[str, dict]:
        token = authorization.removeprefix("Bearer ").strip()
        try:
            claims = s.idp.decode(token, CHAT_AUDIENCE)
        except AuthError:
            raise HTTPException(401, "invalid session") from None
        if not customer_limiter.allow(claims["sub"]):
            raise HTTPException(429, "rate limited", headers={"Retry-After": "60"})
        return token, claims

    @app.get("/healthz")
    async def healthz(s: System = Depends(sys_)):
        return {"ok": True, "breakers": {k: b.state for k, b in s.breakers.items()}}

    @app.post("/auth/login")
    async def login(body: LoginIn, s: System = Depends(sys_)):
        try:
            return {"session_token": s.idp.login(body.username, body.password)}
        except AuthError:
            raise HTTPException(401, "invalid credentials") from None

    @app.post("/auth/step-up/start")
    async def step_up_start(sess=Depends(session), s: System = Depends(sys_)):
        token, claims = sess
        s.idp.start_step_up(token)
        out = {"sent_to": f"phone ending {s.backends.customers.get(claims['sub']).phone_last4}"}
        if os.getenv("CLAIMS_DEV_SHOW_OTP") == "1":  # development only: no SMS gateway here
            out["dev_code"] = s.otp_outbox[claims["sub"]]
        return out

    @app.post("/auth/step-up/verify")
    async def step_up_verify(body: CodeIn, sess=Depends(session), s: System = Depends(sys_)):
        try:
            return {"session_token": s.idp.complete_step_up(sess[0], body.code)}
        except AuthError as exc:
            raise HTTPException(401, str(exc)) from None

    @app.post("/documents")
    async def upload(file: UploadFile = File(...), sess=Depends(session), s: System = Depends(sys_)):
        content = await file.read(s.settings.max_upload_bytes + 1)
        try:
            doc_id = s.upload(sess[0], file.filename or "upload", file.content_type or "", content)
        except UploadRejected as exc:
            raise HTTPException(422, str(exc)) from None
        return {"document_id": doc_id}

    @app.post("/chat")
    async def chat(body: ChatIn, sess=Depends(session), s: System = Depends(sys_)):
        try:
            result = await s.chat.handle_turn(sess[0], body.conversation_id, body.message, body.attachments)
        except PermissionError:
            raise HTTPException(403, "not your conversation") from None
        return {
            "reply": result.reply,
            "conversation_id": result.conversation_id,
            "turn_id": result.turn_id,
            "step_up_required": result.step_up_required,
            "handoff_ticket": result.handoff_ticket,
            "degraded": result.degraded,
        }

    return app
