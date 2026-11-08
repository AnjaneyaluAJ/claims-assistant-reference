"""First notice of loss as a workflow, not an agent (Step 4).

Insurers have a fixed process: identify the policy, collect what happened, collect evidence,
validate, confirm, submit. If an agent improvises this sequence it will sometimes skip a step,
so it runs as a state machine in code. A model is used in exactly two places, both to fill
typed fields: incident extraction (here) and estimate extraction (in the documents server).
Order, validation, confirmation, step-up and submission are code.

    identify_policy -> collect_incident -> collect_evidence -> validate -> confirm_with_customer -> submitted
                              |                                   |               |
                              +----> handoff (injury)             +-> handoff     +-> back to collect_incident
                                                                     (2 failures)     (customer wants changes)
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta

from opentelemetry.trace import Tracer

from ..agents.gateway import McpGateway
from ..agents.loop import ModelCaller
from ..agents.prompts import EXTRACT_INCIDENT
from ..domain.models import IncidentDetails
from ..guardrails.untrusted import wrap
from ..llm.base import ModelRequest
from ..memory.state import CaseState, FnolState

YES = re.compile(r"^\s*(yes|yep|yeah|confirm|correct|submit|go ahead)\b", re.I)
NO = re.compile(r"^\s*(no|nope|change|edit|wrong|not right)\b", re.I)
DONE = re.compile(r"\b(done|no (more )?(documents|photos|files)|that'?s all|nothing else|skip)\b", re.I)
HOME_WORDS = re.compile(r"\b(home|house|flat|leak|water|ceiling|pipe|kitchen|bathroom)\b", re.I)
MOTOR_WORDS = re.compile(r"\b(car|vehicle|drive|driving|reversed|crash|collision|windscreen|bumper|parked)\b", re.I)

INCIDENT_SCHEMA = {
    "type": "object",
    "properties": {
        "loss_type": {
            "type": ["string", "null"],
            "enum": ["accidental_damage", "windscreen", "theft", "escape_of_water", "other", None],
        },
        "incident_date": {"type": ["string", "null"], "description": "ISO date"},
        "location": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
        "third_party_involved": {"type": ["boolean", "null"]},
        "injuries": {"type": ["boolean", "null"]},
        "police_reference": {"type": ["string", "null"]},
    },
    "required": [
        "loss_type",
        "incident_date",
        "location",
        "description",
        "third_party_involved",
        "injuries",
        "police_reference",
    ],
    "additionalProperties": False,
}


@dataclass
class WorkflowReply:
    text: str
    step_up_required: bool = False
    handoff_reasons: list[str] = field(default_factory=list)
    finished: bool = False
    grounded: list[str] = field(default_factory=list)


class FnolWorkflow:
    def __init__(
        self,
        *,
        gateway: McpGateway,
        call_model: ModelCaller,
        tracer: Tracer,
        span_attrs: dict[str, str],
        today: date | None = None,
    ) -> None:
        self.gw = gateway
        self.call_model = call_model
        self.tracer = tracer
        self.span_attrs = span_attrs
        self.today = today or date.today()

    def _step(self, name: str):
        span = self.tracer.start_as_current_span(f"workflow.step {name}")
        return _StepSpan(span, name, self.span_attrs)

    # -- entry points ---------------------------------------------------------------------------------
    async def start(self, case: CaseState, first_message: str) -> WorkflowReply:
        case.workflow = FnolState()
        with self._step("identify_policy"):
            out = await self.gw.call("list_my_policies", {}, agent="fnol_workflow")
            if not out.ok:
                return self._unavailable(case)
            policies = [p for p in out.data.get("policies", []) if p["status"] == "active"]
            case.known_policy_ids = sorted({*case.known_policy_ids, *(p["policy_id"] for p in policies)})
            wf = case.workflow
            wf.candidate_policies = policies
            product = (
                "home" if HOME_WORDS.search(first_message) else "motor" if MOTOR_WORDS.search(first_message) else None
            )
            matches = [p for p in policies if product is None or p["product"] == product]
            if len(matches) == 1:
                wf.policy_id = matches[0]["policy_id"]
            elif not matches:
                case.workflow = None
                return WorkflowReply(
                    "I can't find an active policy to report this under. "
                    "I can put you through to the claims team if you'd like."
                )
        if case.workflow.policy_id is None:
            options = "\n".join(f"{i}. {p['product'].title()}: {p['insured']}" for i, p in enumerate(matches, 1))
            case.workflow.candidate_policies = matches
            return WorkflowReply(f"I'm sorry this has happened. Which policy is this about?\n{options}")
        case.workflow.state = "collect_incident"
        return await self._collect_incident(case, first_message, attachments=[], stepped_up=False)

    async def handle(self, case: CaseState, message: str, attachments: list[str], stepped_up: bool) -> WorkflowReply:
        wf = case.workflow
        assert wf is not None
        if re.search(r"\b(cancel|stop|forget it)\b", message, re.I) and wf.state != "submitted":
            case.workflow = None
            return WorkflowReply("No problem, I've stopped the claim report. Nothing was submitted.", finished=True)
        if wf.state == "identify_policy":
            return await self._choose_policy(case, message)
        if wf.state == "collect_incident":
            return await self._collect_incident(case, message, attachments, stepped_up)
        if wf.state == "collect_evidence":
            return await self._collect_evidence(case, message, attachments)
        if wf.state == "confirm_with_customer":
            return await self._confirm(case, message, stepped_up)
        return WorkflowReply("This claim has already been submitted.", finished=True)

    # -- states ---------------------------------------------------------------------------------------
    async def _choose_policy(self, case: CaseState, message: str) -> WorkflowReply:
        wf = case.workflow
        with self._step("identify_policy"):
            choice = None
            m = re.search(r"\b(\d)\b", message)
            if m and 1 <= int(m.group(1)) <= len(wf.candidate_policies):
                choice = wf.candidate_policies[int(m.group(1)) - 1]
            else:
                for p in wf.candidate_policies:
                    if p["product"] in message.lower() or (p["insured"] or "").lower().split(",")[0] in message.lower():
                        choice = p
            if choice is None:
                return WorkflowReply("Please reply with the number of the policy.")
            wf.policy_id = choice["policy_id"]
            wf.state = "collect_incident"
        return WorkflowReply(
            "Thanks. Tell me what happened: when and where it was, what was damaged, "
            "and whether anyone else was involved or hurt."
        )

    async def _extract_incident(self, text: str) -> IncidentDetails:
        req = ModelRequest(
            route="extract_incident",
            system=EXTRACT_INCIDENT,
            tools=[],
            output_schema=INCIDENT_SCHEMA,
            messages=[
                {"role": "user", "content": f"Today is {self.today.isoformat()}.\n{wrap('customer_message', text)}"}
            ],
        )
        resp = await self.call_model(req, "fnol_extractor")
        try:
            data = json.loads(resp.text)
            return IncidentDetails.model_validate({k: v for k, v in data.items() if v is not None})
        except (ValueError, TypeError):
            return IncidentDetails()

    async def _collect_incident(
        self, case: CaseState, message: str, attachments: list[str], stepped_up: bool
    ) -> WorkflowReply:
        wf = case.workflow
        with self._step("collect_incident") as step:
            found = await self._extract_incident(message)
            merged = wf.incident.model_dump()
            for k, v in found.model_dump().items():
                if v is not None:
                    merged[k] = v
            wf.incident = IncidentDetails.model_validate(merged)
            step.set("app.workflow.fields", ",".join(k for k, v in merged.items() if v is not None))
            if wf.incident.injuries:
                return WorkflowReply("", handoff_reasons=["injury_or_medical"])
            missing = [
                label
                for key, label in (
                    ("incident_date", "the date it happened"),
                    ("description", "what happened"),
                    ("loss_type", "what was damaged"),
                )
                if getattr(wf.incident, key) in (None, "")
            ]
            if missing:
                return WorkflowReply(f"Thanks. Could you also tell me {' and '.join(missing)}?")
            wf.state = "collect_evidence"
        if attachments:
            return await self._collect_evidence(case, "", attachments)
        return WorkflowReply(
            "Got it. If you have photos or a garage estimate, upload them now. If not, just say 'no documents'."
        )

    async def _collect_evidence(self, case: CaseState, message: str, attachments: list[str]) -> WorkflowReply:
        wf = case.workflow
        grounded: list[str] = []
        with self._step("collect_evidence"):
            for doc_id in attachments:
                out = await self.gw.call("extract_estimate", {"document_id": doc_id}, agent="fnol_workflow")
                if not out.ok:
                    if out.error_code in {"unavailable", "circuit_open", "timeout"}:
                        return self._unavailable(case)
                    return WorkflowReply("I couldn't read that file. Please upload a PDF, photo or text file.")
                wf.document_ids.append(doc_id)
                from ..domain.models import EstimateFields

                wf.estimates[doc_id] = EstimateFields.model_validate(out.data["fields"])
                wf.estimate_warnings.extend(out.data.get("warnings", []))
                grounded.append(json.dumps(out.data["fields"]))
        if attachments and not DONE.search(message):
            return WorkflowReply(
                f"Thanks, I've added {len(attachments)} document(s). Upload more, or say 'done' to continue.",
                grounded=grounded,
            )
        return await self._validate_and_confirm(case, grounded)

    def _validation_errors(self, incident: IncidentDetails) -> list[str]:
        errors = []
        if incident.incident_date and incident.incident_date > self.today:
            errors.append("The date can't be in the future.")
        if incident.incident_date and incident.incident_date < self.today - timedelta(days=365):
            errors.append("Incidents more than a year old need to be reported to the claims team directly.")
        if not incident.description or len(incident.description) < 10:
            errors.append("Please describe what happened in a sentence or two.")
        return errors

    async def _validate_and_confirm(self, case: CaseState, grounded: list[str]) -> WorkflowReply:
        wf = case.workflow
        with self._step("validate"):
            errors = self._validation_errors(wf.incident)
            if errors:
                wf.validation_failures += 1
                wf.state = "collect_incident"
                if wf.validation_failures >= 2:
                    return WorkflowReply("", handoff_reasons=["workflow_validation_failed"])
                return WorkflowReply(" ".join(errors))
        with self._step("create_draft"):
            inc = wf.incident
            out = await self.gw.call(
                "create_claim_draft",
                {
                    "policy_id": wf.policy_id,
                    "loss_type": inc.loss_type,
                    "incident_date": inc.incident_date.isoformat(),
                    "description": inc.description,
                    "location": inc.location or "",
                    "third_party_involved": bool(inc.third_party_involved),
                    "injuries": bool(inc.injuries),
                    "police_reference": inc.police_reference or "",
                },
                agent="fnol_workflow",
            )
            if not out.ok:
                if out.error_code in {"unavailable", "circuit_open", "timeout"}:
                    return self._unavailable(case)
                return WorkflowReply("", handoff_reasons=[f"draft_rejected:{out.error_code}"])
            wf.draft_id = out.data["draft_id"]
            wf.idempotency_key = uuid.uuid4().hex  # generated once per draft, reused on every retry
            for doc_id in wf.document_ids:
                est = wf.estimates.get(doc_id)
                await self.gw.call(
                    "attach_evidence",
                    {
                        "draft_id": wf.draft_id,
                        "document_id": doc_id,
                        "estimate": est.model_dump(mode="json") if est else None,
                    },
                    agent="fnol_workflow",
                )
        wf.state = "confirm_with_customer"
        with self._step("confirm_with_customer"):
            return WorkflowReply(self._readback(case), grounded=grounded)

    def _readback(self, case: CaseState) -> str:
        wf = case.workflow
        inc = wf.incident
        lines = [
            "Here's what I'll submit. Please check it carefully:",
            f"- Policy: {wf.policy_id}",
            f"- Date: {inc.incident_date.strftime('%d %B %Y') if inc.incident_date else 'not given'}",
            f"- What happened: {inc.description}",
        ]
        if inc.location:
            lines.append(f"- Where: {inc.location}")
        lines.append(f"- Anyone else involved: {'yes' if inc.third_party_involved else 'no'}")
        for est in wf.estimates.values():
            if est.total is not None:
                lines.append(f"- Garage estimate: {est.garage_name or 'unnamed garage'}, total {est.total:,}")
        if wf.estimate_warnings:
            lines.append("- Note: an adjuster will review the attached estimate before any decision.")
        lines.append(
            "Submitting reports the incident; it doesn't approve any payment. "
            "Reply YES to submit, or tell me what to change."
        )
        return "\n".join(lines)

    async def _confirm(self, case: CaseState, message: str, stepped_up: bool) -> WorkflowReply:
        wf = case.workflow
        if NO.search(message):
            wf.state = "collect_incident"
            return WorkflowReply("What would you like to change?")
        if not YES.search(message):
            return WorkflowReply("Please reply YES to submit, or tell me what to change.")
        if not stepped_up:
            with self._step("step_up"):
                return WorkflowReply(
                    "To keep your account safe, I've sent a code to your phone. Enter it to confirm the submission.",
                    step_up_required=True,
                )
        with self._step("submit"):
            out = await self.gw.call(
                "submit_claim", {"draft_id": wf.draft_id, "idempotency_key": wf.idempotency_key}, agent="fnol_workflow"
            )
        if not out.ok:
            if out.error_code == "step_up_required":
                return WorkflowReply("Please confirm it's you with the code we've sent.", step_up_required=True)
            if out.error_code in {"unavailable", "circuit_open", "timeout"}:
                return WorkflowReply(
                    "The claims system is unavailable right now, so nothing has been submitted. "
                    "Your draft is saved; reply YES later to try again, or ask for a call back."
                )
            return WorkflowReply("", handoff_reasons=[f"submit_rejected:{out.error_code}"])
        wf.state = "submitted"
        wf.claim_id = out.data["claim_id"]
        case.claims_discussed.append(wf.claim_id)
        return WorkflowReply(
            f"Your claim is registered. Your reference is {wf.claim_id}. {out.data['next_step']}",
            finished=True,
            grounded=[json.dumps(out.data)],
        )

    @staticmethod
    def _unavailable(case: CaseState) -> WorkflowReply:
        return WorkflowReply(
            "Our claims system is unavailable right now, so I can't take the report at the moment. "
            "Nothing has been submitted. I can ask the claims team to call you, or you can "
            "try again later."
        )


class _StepSpan:
    def __init__(self, cm, name: str, attrs: dict[str, str]) -> None:
        self.cm, self.name, self.attrs = cm, name, attrs

    def __enter__(self):
        self.span = self.cm.__enter__()
        for k, v in self.attrs.items():
            self.span.set_attribute(k, v)
        self.span.set_attribute("app.workflow.step", self.name)
        return self

    def set(self, key: str, value) -> None:
        self.span.set_attribute(key, value)

    def __exit__(self, *exc):
        return self.cm.__exit__(*exc)
