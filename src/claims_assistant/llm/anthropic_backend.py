"""Claude through the official Anthropic SDK, on the Claude API or on Amazon Bedrock.

Production details handled here:

* Prompt caching. Tool definitions and the system prompt are identical on every turn, so
  they form a cached prefix. Nothing volatile (timestamps, IDs) goes into the system prompt.
* Refusal fallback. On the Claude API, ``fallbacks: "default"`` lets the API re-run a declined
  request on Anthropic's recommended fallback model. Bedrock does not take this parameter.
* Errors. The SDK retries 429/5xx itself; whatever is left becomes ``ModelUnavailable``, which
  the router turns into a fallback to a second endpoint (Step 16). 400-class errors are bugs
  and are raised as they are.
"""

from __future__ import annotations

from typing import Any

import anthropic

from .base import ModelRequest, ModelResponse, ModelUnavailable, Usage

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-fable-5-1", "claude-opus-5", "claude-sonnet-5-5"}


class AnthropicBackend:
    def __init__(
        self,
        provider: str = "anthropic",
        *,
        aws_region: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.provider = provider
        self.name = provider
        if provider == "anthropic":
            # Resolves credentials from the environment (ANTHROPIC_API_KEY or an `ant auth login` profile).
            self.client: Any = anthropic.AsyncAnthropic(timeout=timeout_s, max_retries=max_retries)
        elif provider == "bedrock":
            self.client = anthropic.AsyncAnthropicBedrockMantle(
                aws_region=aws_region, timeout=timeout_s, max_retries=max_retries
            )
        else:
            raise ValueError(f"unknown provider {provider}")

    def _model_id(self, model: str) -> str:
        return f"anthropic.{model}" if self.provider == "bedrock" and not model.startswith("anthropic.") else model

    async def create(self, request: ModelRequest, *, model: str, effort: str) -> ModelResponse:
        tools = [dict(t) for t in request.tools]
        params: dict[str, Any] = {
            "model": self._model_id(model),
            "max_tokens": request.max_tokens,
            # One stable system block with a cache breakpoint: tools + system are cached together.
            "system": [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}],
            "messages": request.messages,
            "output_config": {"effort": effort},
            # Automatic caching of the growing conversation prefix.
            "cache_control": {"type": "ephemeral"},
        }
        if tools:
            params["tools"] = tools
        if request.output_schema is not None:
            params["output_config"]["format"] = {"type": "json_schema", "schema": request.output_schema}

        use_fallback = self.provider == "anthropic" and model in FALLBACK_MODELS
        try:
            if use_fallback:
                resp = await self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **params)
            else:
                resp = await self.client.messages.create(**params)
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as exc:
            raise ModelUnavailable(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ModelUnavailable(str(exc)) from exc
            raise

        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        # Keep every block (thinking included) so the history can be sent back unchanged.
        content = [b.model_dump(mode="json", exclude_none=True) for b in resp.content]
        return ModelResponse(
            content=content,
            stop_reason=resp.stop_reason or "end_turn",
            usage=usage,
            model=resp.model,
            request_id=getattr(resp, "_request_id", None),
            provider=self.provider,
        )
