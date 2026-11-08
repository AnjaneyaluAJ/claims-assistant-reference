"""Configuration. Everything that varies by environment is here, read from env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Route:
    model: str
    effort: str
    max_tokens: int = 4096


# Model routing (Step 11). The starting point is the strongest model everywhere, with effort
# set per task. Move a route to a smaller model only when the evaluation set shows it holds
# quality on that route: `claims-assistant eval --route extractor=claude-haiku-5-5`.
DEFAULT_ROUTES: dict[str, Route] = {
    "coordinator": Route("claude-opus-5-5", "medium"),
    "claims_agent": Route("claude-opus-5-5", "medium"),
    "policy_agent": Route("claude-opus-5-5", "medium"),
    "extract_incident": Route("claude-opus-5-5", "low", 2048),
    "extract_estimate": Route("claude-opus-5-5", "low", 2048),
    "summarizer": Route("claude-opus-5-5", "low", 1024),
    "judge": Route("claude-opus-5-5", "high", 2048),
}

# USD per million tokens (Claude API list prices, 2026-10). Cache writes are 1.25x input.
PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01},
    "scripted": {"input": 0.0, "output": 0.0, "cache_read": 0.0},
}

SERVICE_SCOPES: dict[str, set[str]] = {
    "policy-mcp": {"policy:read"},
    "claims-mcp": {"claim:read", "claim:submit"},
    "documents-mcp": {"document:read"},
    "payments-mcp": {"payout_account:update"},
}

# Least privilege per call: the gateway requests only the scope a tool needs.
TOOL_SCOPES: dict[str, str] = {
    "list_my_policies": "policy:read",
    "get_cover_summary": "policy:read",
    "search_policy_wording": "policy:read",
    "list_my_claims": "claim:read",
    "get_claim_status": "claim:read",
    "create_claim_draft": "claim:submit",
    "attach_evidence": "claim:submit",
    "submit_claim": "claim:submit",
    "extract_estimate": "document:read",
    "update_payout_account": "payout_account:update",
}

READ_ONLY_TOOLS = {
    "list_my_policies",
    "get_cover_summary",
    "search_policy_wording",
    "list_my_claims",
    "get_claim_status",
    "extract_estimate",
}


@dataclass
class Settings:
    model_provider: str = field(default_factory=lambda: os.getenv("CLAIMS_MODEL_PROVIDER", "scripted"))
    fallback_provider: str | None = field(default_factory=lambda: os.getenv("CLAIMS_FALLBACK_PROVIDER") or None)
    aws_region: str | None = field(default_factory=lambda: os.getenv("AWS_REGION"))
    routes: dict[str, Route] = field(default_factory=lambda: dict(DEFAULT_ROUTES))

    # Base URLs of the MCP servers. In-process mode uses these as identifiers and audiences only.
    service_urls: dict[str, str] = field(
        default_factory=lambda: {
            "policy-mcp": os.getenv("POLICY_MCP_URL", "http://127.0.0.1:9101/mcp"),
            "claims-mcp": os.getenv("CLAIMS_MCP_URL", "http://127.0.0.1:9102/mcp"),
            "documents-mcp": os.getenv("DOCUMENTS_MCP_URL", "http://127.0.0.1:9103/mcp"),
            "payments-mcp": os.getenv("PAYMENTS_MCP_URL", "http://127.0.0.1:9104/mcp"),
        }
    )
    in_process_mcp: bool = field(default_factory=lambda: os.getenv("CLAIMS_MCP_MODE", "in_process") == "in_process")

    # Step 15 and 16 limits.
    max_agent_steps: int = 8
    tool_timeout_s: float = 8.0
    turn_budget_s: float = 45.0
    daily_token_budget_per_customer: int = 400_000
    breaker_failure_threshold: int = 3
    breaker_reset_s: float = 30.0

    # Step 10.
    verbatim_turns: int = 6

    # Step 17.
    max_message_chars: int = 4000
    max_upload_bytes: int = 5 * 1024 * 1024
    rate_limit_per_ip_per_min: int = 60
    rate_limit_per_customer_per_min: int = 20
