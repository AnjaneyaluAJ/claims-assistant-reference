"""Model routing and endpoint fallback (Steps 11 and 16)."""

from __future__ import annotations

import logging

from ..config import PRICES, Route, Settings
from .base import ModelClient, ModelRequest, ModelResponse, ModelUnavailable, Usage

log = logging.getLogger(__name__)


def cost_usd(model: str, usage: Usage) -> float:
    key = model.removeprefix("anthropic.")
    price = next((v for k, v in PRICES.items() if key.startswith(k)), None)
    if price is None:
        return 0.0
    return (
        usage.input_tokens * price["input"]
        + usage.output_tokens * price["output"]
        + usage.cache_read_input_tokens * price["cache_read"]
        + usage.cache_creation_input_tokens * price["input"] * 1.25
    ) / 1_000_000


class ModelRouter:
    """Picks the model for a route and fails over to a second endpoint when the first is down.

    The fallback endpoint must already have passed the evaluation gate; failing over to an
    untested model is how outages turn into incidents.
    """

    def __init__(self, primary: ModelClient, routes: dict[str, Route], fallback: ModelClient | None = None):
        self.primary = primary
        self.fallback = fallback
        self.routes = routes

    def route(self, name: str) -> Route:
        return self.routes.get(name) or self.routes["coordinator"]

    async def create(self, request: ModelRequest) -> ModelResponse:
        r = self.route(request.route)
        request.max_tokens = min(request.max_tokens, r.max_tokens) if request.max_tokens else r.max_tokens
        try:
            return await self.primary.create(request, model=r.model, effort=r.effort)
        except ModelUnavailable:
            if self.fallback is None:
                raise
            log.warning("primary model endpoint unavailable on route %s; using fallback", request.route)
            return await self.fallback.create(request, model=r.model, effort=r.effort)


def build_router(settings: Settings) -> ModelRouter:
    def make(provider: str) -> ModelClient:
        if provider == "scripted":
            from .scripted import ScriptedModel

            return ScriptedModel()
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(provider, aws_region=settings.aws_region)

    primary = make(settings.model_provider)
    fallback = make(settings.fallback_provider) if settings.fallback_provider else None
    return ModelRouter(primary, settings.routes, fallback)
