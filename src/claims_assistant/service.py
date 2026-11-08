"""One customer turn, end to end. This is the "chat API" box in the final architecture.

input guardrails (PII tokenization, handoff triggers)
  -> pending confirmation?   -> code handles yes/no (+ step-up) and calls the write tool
  -> claim report running?   -> FNOL workflow (code, model only for extraction)
  -> otherwise               -> coordinator agent -> domain agents -> MCP servers
output guardrails (promise and amount checks) -> mask placeholders -> reply
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .agents import prompts
from .agents.gateway import McpGateway, ToolOutcome
from .agents.loop import ModelCaller, run_agent
from .cost.ledger import BudgetExceeded
from .guardrails.output import check_reply
from .handoff.handoff import customer_message, detect_triggers
from .identity.idp import CHAT_AUDIENCE, AuthError
from .llm.base import ModelRequest
from .memory.state import Conversation, PendingConfirmation
from .observability import tracing as t
from .pii.vault import PLACEHOLDER
from .workflow.fnol import YES, FnolWorkflow, WorkflowReply

_Q = {
    "type": "object",
    "properties": {"question": {"type": "string", "description": "Self-contained question for the specialist"}},
    "required": ["question"],
    "additionalProperties": False,
}
COORDINATOR_TOOLS = [
    {
        "name": "ask_claims_agent",
        "description": "Ask the claims specialist about the customer's existing claims.",
        "input_schema": _Q,
    },
    {
        "name": "ask_policy_agent",
        "description": "Ask the policy specialist what the customer's policy covers.",
        "input_schema": _Q,
    },
    {
        "name": "start_claim_report",
        "description": "Start the fixed process for reporting a new incident or loss.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "propose_payout_account_change",
        "description": "Prepare a change to the account settlements are paid into. Pass the placeholders exactly "
        "as they appear. The customer confirms the change afterwards.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sort_code": {"type": "string", "pattern": PLACEHOLDER.pattern},
                "account_number": {"type": "string", "pattern": PLACEHOLDER.pattern},
            },
            "required": ["sort_code", "account_number"],
            "additionalProperties": False,
        },
    },
    {
        "name": "request_human_handoff",
        "description": "Pass the conversation to a human adjuster.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
]

DEGRADED_REPLY = (
    "Our {system} system is unavailable right now, so I can't check that for you, and I won't guess. "
    "I can ask the team to call you back once it's available, or you can try again later."
)
BUDGET_REPLY = (
    "You've reached today's limit for the assistant. I've noted your question; "
    "you can reach the claims team by phone, or try again tomorrow."
)


@dataclass
class TurnResult:
    reply: str
    conversation_id: str
    turn_id: str
    step_up_required: bool = False
    handoff_ticket: str | None = None
    degraded: bool = False
    blocked_reasons: list[str] = field(default_factory=list)


class ChatService:
    def __init__(self, system) -> None:  # system: claims_assistant.system.System (avoid import cycle)
        self.s = system

    async def handle_turn(
        self, session_token: str, conversation_id: str | None, message: str, attachments: list[str] | None = None
    ) -> TurnResult:
        s = self.s
        try:
            claims = s.idp.decode(session_token, CHAT_AUDIENCE)
        except AuthError:
            raise PermissionError("invalid session") from None
        customer_id, sid = claims["sub"], claims["sid"]
        stepped_up = s.idp.has_recent_step_up(claims)
        conversation_id = conversation_id or f"conv-{uuid.uuid4().hex[:10]}"
        conv = s.conversations.get_or_create(conversation_id, customer_id, sid)
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        attrs = {
            t.CONVERSATION_ID: conversation_id,
            "app.turn.id": turn_id,
            "app.customer.pseudonym": s.pseudonym(customer_id),
        }

        with s.tracing.tracer.start_as_current_span("chat.turn") as span:
            for k, v in attrs.items():
                span.set_attribute(k, v)
            result = await self._turn(conv, session_token, stepped_up, message, attachments or [], attrs, turn_id)
            span.set_attribute("app.turn.handoff", bool(result.handoff_ticket))
            span.set_attribute("app.turn.degraded", result.degraded)
            span.set_attribute("app.turn.step_up_required", result.step_up_required)
            if result.blocked_reasons:
                span.set_attribute("app.turn.output_blocked", ",".join(result.blocked_reasons))
        try:
            await self.summarize_if_needed(conv)
        except BudgetExceeded:
            pass  # keep the verbatim history; the next turn will hit the budget reply anyway
        return result

    async def _turn(
        self,
        conv: Conversation,
        session_token: str,
        stepped_up: bool,
        message: str,
        attachments: list[str],
        attrs: dict[str, str],
        turn_id: str,
    ) -> TurnResult:
        s = self.s
        tracer = s.tracing.tracer
        result = TurnResult(reply="", conversation_id=conv.conversation_id, turn_id=turn_id)

        # 1. Input guardrails: tokenize personal data before anything else sees the message.
        with tracer.start_as_current_span("guardrail.input") as span:
            for k, v in attrs.items():
                span.set_attribute(k, v)
            tokenized, detections = s.vault.tokenize(message[: s.settings.max_message_chars], conv.session_id)
            span.set_attribute("app.pii.kinds", sorted({d.kind for d in detections}))
            triggers = detect_triggers(message)
            span.set_attribute("app.handoff.triggers", triggers)

        gateway = McpGateway(
            idp=s.idp,
            session_token=session_token,
            catalog=s.catalog,
            urls=s.settings.service_urls,
            connect=s.connect,
            breakers=s.breakers,
            tracer=tracer,
            span_attrs=attrs,
            tool_timeout_s=s.settings.tool_timeout_s,
            deadline=time.monotonic() + s.settings.turn_budget_s,
        )
        call_model = ModelCaller(
            router=s.router,
            tracer=tracer,
            ledger=s.ledger,
            conversation_id=conv.conversation_id,
            customer_id=conv.customer_id,
            span_attrs=attrs,
        )
        workflow = FnolWorkflow(gateway=gateway, call_model=call_model, tracer=tracer, span_attrs=attrs, today=s.today)
        grounded: list[str] = []

        try:
            if triggers and not conv.case.handed_off:
                return self._handoff(conv, triggers, tokenized, result)

            # 2. A write waiting for the customer's yes.
            if conv.case.pending is not None:
                reply = await self._handle_pending(conv, tokenized, stepped_up, gateway, result)
                if reply is not None:
                    return self._finish(conv, tokenized, reply, result, grounded=[reply])

            # 3. A claim report in progress.
            if conv.case.workflow is not None and conv.case.workflow.state != "submitted":
                wf_reply = await workflow.handle(conv.case, tokenized, attachments, stepped_up)
                return self._from_workflow(conv, tokenized, wf_reply, result)

            # 4. The coordinator.
            flags: dict[str, Any] = {}
            run = await self._coordinate(conv, tokenized, gateway, call_model, attrs, flags, grounded)
        except BudgetExceeded:
            return self._finish(conv, tokenized, BUDGET_REPLY, result, outcome="abandoned")

        if flags.get("start_claim_report"):
            wf_reply = await workflow.start(conv.case, tokenized)
            return self._from_workflow(conv, tokenized, wf_reply, result)
        if flags.get("handoff"):
            return self._handoff(conv, ["agent_requested:" + flags["handoff"]], tokenized, result)
        if conv.case.pending is not None and flags.get("pending_created"):
            return self._finish(conv, tokenized, conv.case.pending.readback, result, grounded=[])
        if run.degraded:
            result.degraded = True
            system = s.catalog.server_of.get(run.degraded, "claims-mcp").split("-")[0]
            return self._finish(conv, tokenized, DEGRADED_REPLY.format(system=system), result, outcome="degraded")
        if run.stop_reason in {"refusal", "max_steps"} or not run.text.strip():
            conv.case.failed_attempts += 1
            if conv.case.failed_attempts >= 2:
                return self._handoff(conv, ["repeated_failure"], tokenized, result)
            return self._finish(
                conv,
                tokenized,
                "Sorry, I couldn't work that out. Could you put it another way?",
                result,
                outcome="open",
            )

        # 5. Output guardrails.
        verdict = check_reply(run.text, "\n".join(grounded))
        with tracer.start_as_current_span("guardrail.output") as span:
            for k, v in attrs.items():
                span.set_attribute(k, v)
            span.set_attribute("app.output.ok", verdict.ok)
            span.set_attribute("app.output.reasons", verdict.reasons)
            if not verdict.ok:
                span.set_attribute("app.output.original", run.text[:2000])
        if not verdict.ok:
            result.blocked_reasons = verdict.reasons
        conv.case.failed_attempts = 0
        return self._finish(conv, tokenized, verdict.text, result)

    # -- coordinator ---------------------------------------------------------------------------------
    async def _coordinate(
        self,
        conv: Conversation,
        tokenized: str,
        gateway: McpGateway,
        call_model: ModelCaller,
        attrs: dict[str, str],
        flags: dict[str, Any],
        grounded: list[str],
    ):
        s = self.s
        tracer = s.tracing.tracer

        async def domain_agent(name: str, question: str, call_id: str) -> ToolOutcome:
            route, system, servers = {
                "ask_claims_agent": ("claims_agent", prompts.CLAIMS_AGENT, ("claims-mcp",)),
                "ask_policy_agent": ("policy_agent", prompts.POLICY_AGENT, ("policy-mcp",)),
            }[name]
            tools = s.catalog.for_servers(*servers, only=s.agent_tool_allowlist[route])

            async def execute(tool: str, args: dict[str, Any], tool_call_id: str) -> ToolOutcome:
                if tool not in s.agent_tool_allowlist[route]:
                    return ToolOutcome(tool, False, error_code="unknown_tool", error_message="Not available.")
                out = await gateway.call(tool, args, call_id=tool_call_id, agent=route)
                if out.ok:
                    grounded.append(json.dumps(out.data, default=str))
                    _remember(conv, out)
                return out

            run = await run_agent(
                call_model=call_model,
                tracer=tracer,
                span_attrs=attrs,
                agent=route,
                route=route,
                system=system,
                messages=[{"role": "user", "content": question}],
                tools=tools,
                execute=execute,
                max_steps=s.settings.max_agent_steps,
            )
            if run.degraded:
                return ToolOutcome(
                    name,
                    False,
                    error_code="unavailable",
                    error_message=f"{servers[0].split('-')[0]} system unavailable",
                    data={"degraded_tool": run.degraded},
                )
            facts = [o.data for o in run.outcomes if o.ok]
            return ToolOutcome(name, True, data={"answer": run.text, "facts": facts})

        async def execute(tool: str, args: dict[str, Any], call_id: str) -> ToolOutcome:
            if tool in ("ask_claims_agent", "ask_policy_agent"):
                out = await domain_agent(tool, str(args.get("question", tokenized)), call_id)
                if not out.ok and out.data.get("degraded_tool"):
                    flags["degraded_tool"] = out.data["degraded_tool"]
                return out
            if tool == "start_claim_report":
                flags["start_claim_report"] = True
                return ToolOutcome(tool, True, data={"status": "workflow_started"})
            if tool == "propose_payout_account_change":
                sc, acct = args.get("sort_code", ""), args.get("account_number", "")
                ok = (
                    PLACEHOLDER.fullmatch(sc)
                    and PLACEHOLDER.fullmatch(acct)
                    and PLACEHOLDER.fullmatch(sc).group(1) == "sort_code"
                    and PLACEHOLDER.fullmatch(acct).group(1) == "account_number"
                )
                if not ok:
                    return ToolOutcome(
                        tool,
                        False,
                        error_code="invalid_argument",
                        error_message="Ask the customer to type the sort code and account number.",
                    )
                readback = s.vault.mask_for_display(
                    f"I'll change your payout account to {acct} with {sc}. "
                    "Reply YES to confirm. We'll send a code to your phone to check it's you."
                )
                conv.case.pending = PendingConfirmation(
                    action="update_payout_account", args={"sort_code": sc, "account_number": acct}, readback=readback
                )
                flags["pending_created"] = True
                return ToolOutcome(tool, True, data={"status": "awaiting_customer_confirmation"})
            if tool == "request_human_handoff":
                flags["handoff"] = str(args.get("reason", "requested"))[:80]
                return ToolOutcome(tool, True, data={"status": "handoff_requested"})
            return ToolOutcome(tool, False, error_code="unknown_tool", error_message="No such tool.")

        messages = conv.history_for_model(s.settings.verbatim_turns) + [{"role": "user", "content": tokenized}]
        run = await run_agent(
            call_model=call_model,
            tracer=tracer,
            span_attrs=attrs,
            agent="coordinator",
            route="coordinator",
            system=prompts.COORDINATOR,
            messages=messages,
            tools=COORDINATOR_TOOLS,
            execute=execute,
            max_steps=s.settings.max_agent_steps,
            stop_after_tools=lambda: bool(
                flags.get("start_claim_report")
                or flags.get("handoff")
                or flags.get("pending_created")
                or flags.get("degraded_tool")
            ),
        )
        if flags.get("degraded_tool"):
            run.degraded = flags["degraded_tool"]
        return run

    # -- confirmations, handoff, workflow -------------------------------------------------------------
    async def _handle_pending(
        self, conv: Conversation, tokenized: str, stepped_up: bool, gateway: McpGateway, result: TurnResult
    ) -> str | None:
        pending = conv.case.pending
        if not YES.search(tokenized):
            conv.case.pending = None
            return "Okay, I haven't changed anything." if tokenized.strip().lower().startswith("no") else None
        if not stepped_up:
            result.step_up_required = True
            return "To keep your account safe, I've sent a code to your phone. Enter it to confirm the change."
        out = await gateway.call(pending.action, pending.args, agent="confirmation")
        if not out.ok:
            if out.error_code == "step_up_required":
                result.step_up_required = True
                return "Please confirm it's you with the code we've sent."
            conv.case.pending = None
            return out.error_message or "I couldn't make that change."
        conv.case.pending = None
        return f"Done. Settlements will now be paid into the account ending {out.data['account_ending']}."

    def _handoff(self, conv: Conversation, reasons: list[str], tokenized: str, result: TurnResult) -> TurnResult:
        s = self.s
        with s.tracing.tracer.start_as_current_span("handoff") as span:
            span.set_attribute(t.CONVERSATION_ID, conv.conversation_id)
            span.set_attribute("app.turn.id", result.turn_id)
            span.set_attribute("app.handoff.reasons", reasons)
            customer_said = [m["content"] for m in conv.turns if m["role"] == "user"] + [tokenized]
            answered = [m["content"] for m in conv.turns if m["role"] == "assistant"]
            ticket = s.queue.enqueue(conv.customer_id, reasons, conv.case, customer_said, answered)
            span.set_attribute("app.handoff.ticket", ticket.ticket_id)
        conv.case.handed_off = True
        result.handoff_ticket = ticket.ticket_id
        return self._finish(conv, tokenized, customer_message(ticket), result, outcome="handoff")

    def _from_workflow(self, conv: Conversation, tokenized: str, wf: WorkflowReply, result: TurnResult) -> TurnResult:
        if wf.handoff_reasons:
            return self._handoff(conv, wf.handoff_reasons, tokenized, result)
        result.step_up_required = wf.step_up_required
        outcome = "resolved" if wf.finished else "open"
        return self._finish(conv, tokenized, wf.text, result, outcome=outcome, grounded=wf.grounded)

    def _finish(
        self,
        conv: Conversation,
        tokenized: str,
        reply: str,
        result: TurnResult,
        *,
        outcome: str = "resolved",
        grounded: list[str] | None = None,
    ) -> TurnResult:
        s = self.s
        display = s.vault.mask_for_display(reply)
        conv.turns.append({"role": "user", "content": tokenized})
        conv.turns.append({"role": "assistant", "content": reply})
        s.ledger.set_outcome(conv.conversation_id, outcome)
        result.reply = display
        return result

    async def summarize_if_needed(self, conv: Conversation) -> None:
        """Roll older turns into a running summary so long conversations don't grow without bound."""
        s = self.s
        if not conv.needs_summary(s.settings.verbatim_turns):
            return
        keep = 2 * s.settings.verbatim_turns
        older, conv.turns = conv.turns[:-keep], conv.turns[-keep:]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in older)
        call_model = ModelCaller(
            router=s.router,
            tracer=s.tracing.tracer,
            ledger=s.ledger,
            conversation_id=conv.conversation_id,
            customer_id=conv.customer_id,
            span_attrs={t.CONVERSATION_ID: conv.conversation_id},
        )
        resp = await call_model(
            ModelRequest(
                route="summarizer",
                system=prompts.SUMMARIZER,
                messages=[{"role": "user", "content": f"{conv.summary}\n{text}"}],
            ),
            "summarizer",
        )
        conv.summary = resp.text.strip()[:1500]


def _remember(conv: Conversation, out: ToolOutcome) -> None:
    """Case state keeps references (which claims and policies were discussed), never claim facts."""
    if out.tool == "get_claim_status" and "claim_id" in out.data:
        if out.data["claim_id"] not in conv.case.claims_discussed:
            conv.case.claims_discussed.append(out.data["claim_id"])
    if out.tool == "list_my_policies":
        ids = [p["policy_id"] for p in out.data.get("policies", [])]
        conv.case.known_policy_ids = sorted({*conv.case.known_policy_ids, *ids})
