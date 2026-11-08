"""The agent loop: model proposes, code decides (Step 2), with Step 14-16 controls built in.

* Every model call is a span with tokens, cost, model and finish reason.
* A step limit stops runaway loops (Step 15).
* A per-customer token budget is checked before every model call (Step 15).
* If a tool reports its backend unavailable, the loop stops and returns a degraded result;
  the reply is then written by code, so the model cannot improvise claim facts (Step 16).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.trace import Tracer

from ..cost.ledger import CostLedger
from ..llm.base import ModelRequest, ModelResponse
from ..llm.router import ModelRouter, cost_usd
from ..observability import tracing as t
from .gateway import ToolOutcome

DEGRADED_CODES = {"unavailable", "circuit_open", "timeout"}
Executor = Callable[[str, dict[str, Any], str], Awaitable[ToolOutcome]]


@dataclass
class AgentRun:
    text: str
    outcomes: list[ToolOutcome] = field(default_factory=list)
    degraded: str | None = None
    stop_reason: str = "end_turn"


@dataclass
class ModelCaller:
    router: ModelRouter
    tracer: Tracer
    ledger: CostLedger
    conversation_id: str
    customer_id: str
    span_attrs: dict[str, str]

    async def __call__(self, request: ModelRequest, agent: str) -> ModelResponse:
        self.ledger.check_budget(self.customer_id)
        route = self.router.route(request.route)
        with self.tracer.start_as_current_span(f"chat {route.model}") as span:
            for k, v in self.span_attrs.items():
                span.set_attribute(k, v)
            span.set_attribute(t.OP, "chat")
            span.set_attribute(t.REQ_MODEL, route.model)
            span.set_attribute(t.AGENT_NAME, agent)
            span.set_attribute("app.route", request.route)
            span.set_attribute("app.effort", route.effort)
            span.set_attribute("app.prompt.chars", len(request.system))
            resp = await self.router.create(request)
            usd = cost_usd(resp.model, resp.usage)
            span.set_attribute(t.PROVIDER, resp.provider)
            span.set_attribute(t.RESP_MODEL, resp.model)
            span.set_attribute(t.IN_TOK, resp.usage.input_tokens)
            span.set_attribute(t.OUT_TOK, resp.usage.output_tokens)
            span.set_attribute("gen_ai.usage.cache_read.input_tokens", resp.usage.cache_read_input_tokens)
            span.set_attribute("gen_ai.usage.cache_creation.input_tokens", resp.usage.cache_creation_input_tokens)
            span.set_attribute(t.FINISH, [resp.stop_reason])
            span.set_attribute("app.cost_usd", usd)
            if resp.request_id:
                span.set_attribute("app.request_id", resp.request_id)
            span.set_attribute("app.output.text", resp.text[:2000])
        self.ledger.record(self.conversation_id, self.customer_id, resp.usage.total, usd)
        return resp


async def run_agent(
    *,
    call_model: ModelCaller,
    tracer: Tracer,
    span_attrs: dict[str, str],
    agent: str,
    route: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    execute: Executor,
    max_steps: int,
    stop_after_tools: Callable[[], bool] | None = None,
) -> AgentRun:
    run = AgentRun(text="")
    with tracer.start_as_current_span(f"invoke_agent {agent}") as span:
        for k, v in span_attrs.items():
            span.set_attribute(k, v)
        span.set_attribute(t.OP, "invoke_agent")
        span.set_attribute(t.AGENT_NAME, agent)
        for step in range(max_steps):  # noqa: B007 (step is reported on the span after the loop)
            resp = await call_model(ModelRequest(route=route, system=system, messages=messages, tools=tools), agent)
            # Append the whole response, thinking blocks included: history stays append-only.
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "refusal":
                run.text, run.stop_reason = "", "refusal"
                break
            if resp.stop_reason == "pause_turn":
                continue
            calls = resp.tool_calls
            if not calls:
                run.text, run.stop_reason = resp.text, resp.stop_reason
                break
            results = []
            for call in calls:
                outcome = await execute(call["name"], call.get("input") or {}, call["id"])
                run.outcomes.append(outcome)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call["id"],
                        "content": outcome.for_model(),
                        "is_error": not outcome.ok,
                    }
                )
                if not outcome.ok and outcome.error_code in DEGRADED_CODES:
                    run.degraded = outcome.tool
            # All results for one assistant message go back in a single user message.
            messages.append({"role": "user", "content": results})
            if run.degraded:
                run.stop_reason = "degraded"
                break
            if stop_after_tools is not None and stop_after_tools():
                run.stop_reason = "handed_to_code"
                break
        else:
            run.stop_reason = "max_steps"
        span.set_attribute("app.agent.steps", step + 1)
        span.set_attribute("app.agent.stop_reason", run.stop_reason)
        if run.degraded:
            span.set_attribute("app.degraded_tool", run.degraded)
    return run
