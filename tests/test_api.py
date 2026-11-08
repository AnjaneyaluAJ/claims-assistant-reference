"""The chat API end to end: login, chat, upload, step-up, over HTTP."""

from pathlib import Path

from fastapi.testclient import TestClient

from claims_assistant.api.app import create_app
from claims_assistant.api.edge import EICAR
from claims_assistant.config import Settings
from claims_assistant.system import System

FIXTURES = Path(__file__).resolve().parents[1] / "evals" / "fixtures"


def test_full_payout_flow_over_http(monkeypatch):
    monkeypatch.setenv("CLAIMS_DEV_SHOW_OTP", "1")
    with TestClient(create_app(System(Settings(model_provider="scripted")))) as c:
        assert c.post("/auth/login", json={"username": "maya", "password": "nope"}).status_code == 401
        token = c.post("/auth/login", json={"username": "maya", "password": "demo"}).json()["session_token"]
        h = {"Authorization": f"Bearer {token}"}

        r = c.post("/chat", headers=h, json={"message": "Pay the settlement into 41926630, sort code 20-45-77."})
        body = r.json()
        assert "ending 6630" in body["reply"] and "41926630" not in r.text
        cid = body["conversation_id"]

        r = c.post("/chat", headers=h, json={"message": "yes", "conversation_id": cid}).json()
        assert r["step_up_required"]

        code = c.post("/auth/step-up/start", headers=h).json()["dev_code"]
        stepped = c.post("/auth/step-up/verify", headers=h, json={"code": code}).json()["session_token"]
        r = c.post(
            "/chat", headers={"Authorization": f"Bearer {stepped}"}, json={"message": "yes", "conversation_id": cid}
        ).json()
        assert "ending 6630" in r["reply"]


def test_conversation_cannot_be_used_by_another_customer():
    with TestClient(create_app(System(Settings(model_provider="scripted")))) as c:
        maya = c.post("/auth/login", json={"username": "maya", "password": "demo"}).json()["session_token"]
        tom = c.post("/auth/login", json={"username": "tom", "password": "demo"}).json()["session_token"]
        cid = c.post(
            "/chat", headers={"Authorization": f"Bearer {maya}"}, json={"message": "What's the status of my claim?"}
        ).json()["conversation_id"]
        r = c.post(
            "/chat",
            headers={"Authorization": f"Bearer {tom}"},
            json={"message": "what did I just ask?", "conversation_id": cid},
        )
        assert r.status_code == 403


def test_upload_is_scanned():
    with TestClient(create_app(System(Settings(model_provider="scripted")))) as c:
        token = c.post("/auth/login", json={"username": "maya", "password": "demo"}).json()["session_token"]
        h = {"Authorization": f"Bearer {token}"}
        ok = c.post(
            "/documents",
            headers=h,
            files={"file": ("e.txt", (FIXTURES / "estimate_clean.txt").read_bytes(), "text/plain")},
        )
        assert ok.status_code == 200 and ok.json()["document_id"].startswith("DOC-")
        bad = c.post("/documents", headers=h, files={"file": ("e.txt", EICAR, "text/plain")})
        assert bad.status_code == 422


def test_requests_need_a_session():
    with TestClient(create_app(System(Settings(model_provider="scripted")))) as c:
        assert c.post("/chat", headers={"Authorization": "Bearer junk"}, json={"message": "hi"}).status_code == 401
