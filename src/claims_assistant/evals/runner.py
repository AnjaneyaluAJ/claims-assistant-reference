"""Evaluation as a release gate (Steps 1 and 13).

Each case is a scripted conversation with expectations on three things:

* the trace: which tools ran, with which arguments, which workflow steps happened
  (read from the same OpenTelemetry spans production exports)
* the reply: required and forbidden content, handoff, step-up, degraded mode
* the end state: claims submitted, audit denials, payout account changes

Every case runs several times and the gate uses pass rates. Cases marked ``attack`` must pass
every run. Any change to a prompt, model, tool description or workflow runs this in CI.

With the scripted model this checks the architecture. With ``--provider anthropic`` it also
checks the model: tool choice, arguments, and (with a rubric) answer quality via a judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..config import Route, Settings
from ..llm.base import ModelRequest, ModelResponse
from ..llm.router import build_router
from ..observability import tracing as t
from ..system import System

ROOT = Path(__file__).resolve().parents[3]
CASES_DIR = ROOT / "evals" / "cases"
FIXTURES = ROOT / "evals" / "fixtures"
EVAL_TODAY = date(2026, 11, 6)  # seed data is dated around this day
MEDIA = {".txt": "text/plain", ".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg"}


class RecordingClient:
    """Wraps the model client and keeps every request, so cases can assert what models were shown."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "model")
        self.requests: list[ModelRequest] = []

    async def create(self, request: ModelRequest, *, model: str, effort: str) -> ModelResponse:
        self.requests.append(request)
        return await self.inner.create(request, model=model, effort=effort)

    def all_inputs(self) -> str:
        return "\n".join(r.system + json.dumps(r.messages, default=str) for r in self.requests)


@dataclass
class CaseResult:
    case_id: str
    scenario: str
    attack: bool
    runs: int = 0
    passed: int = 0
    failures: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    judge_scores: list[float] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.runs if self.runs else 0.0


def load_cases(selected: list[str] | None = None) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(CASES_DIR.glob("*.yaml")):
        for case in yaml.safe_load(path.read_text()) or []:
            if not selected or case["id"] in selected or case.get("scenario") in selected:
                cases.append(case)
    return cases


def _apply_setup(s: System, setup: dict[str, Any]) -> None:
    if setup.get("claims_down"):
        s.backends.claims.faults.maintenance = True
    if setup.get("claims_failure_rate"):
        s.backends.claims.faults.failure_rate = float(setup["claims_failure_rate"])
    if setup.get("named_drivers_see_all_claims"):
        for p in s.backends.policies._policies.values():
            p.delegation.named_drivers_see_all_claims = True
    if setup.get("daily_token_budget") is not None:
        s.ledger.daily_token_budget = int(setup["daily_token_budget"])


def _check_turn(
    expect_trace: dict,
    expect_reply: dict,
    spans,
    reply: str,
    result,
    recorder: RecordingClient,
    errors: list[str],
    label: str,
) -> None:
    tool_spans = [sp for sp in spans if (sp.attributes or {}).get(t.OP) == "execute_tool"]
    called = [sp.attributes[t.TOOL_NAME] for sp in tool_spans]
    steps = [sp.attributes.get("app.workflow.step") for sp in spans if sp.attributes.get("app.workflow.step")]
    agents = [sp.attributes.get(t.AGENT_NAME) for sp in spans if sp.attributes.get(t.OP) == "invoke_agent"]
    coordinator_tools = [
        b["name"]
        for r in recorder.requests
        if r.route == "coordinator"
        for m in r.messages
        if m["role"] == "assistant" and isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "tool_use"
    ]

    for tool in expect_trace.get("tools_called", []):
        if tool not in called and tool not in coordinator_tools:
            errors.append(f"{label}: expected tool {tool}, saw {called}")
    for tool in expect_trace.get("tools_not_called", []):
        if tool in called:
            errors.append(f"{label}: tool {tool} must not run")
    for tool, n in (expect_trace.get("max_calls") or {}).items():
        if called.count(tool) > n:
            errors.append(f"{label}: {tool} ran {called.count(tool)} times, max {n}")
    for spec in expect_trace.get("tool_args", []):
        ok = any(
            sp.attributes[t.TOOL_NAME] == spec["tool"]
            and all(
                json.loads(sp.attributes.get("app.tool.arguments", "{}")).get(k) == v for k, v in spec["args"].items()
            )
            for sp in tool_spans
        )
        if not ok:
            errors.append(f"{label}: no {spec['tool']} call with {spec['args']}")
    for step in expect_trace.get("steps", []):
        if step not in steps:
            errors.append(f"{label}: expected workflow step {step}, saw {steps}")
    for agent in expect_trace.get("agents", []):
        if agent not in agents:
            errors.append(f"{label}: expected agent {agent}, saw {agents}")
    for needle in expect_trace.get("result_contains", []):
        if not any(needle in (sp.attributes.get("app.tool.result") or "") for sp in tool_spans):
            errors.append(f"{label}: no tool result contains {needle!r}")

    low = reply.lower()
    for needle in expect_reply.get("contains", []):
        if needle.lower() not in low:
            errors.append(f"{label}: reply missing {needle!r}: {reply[:200]!r}")
    for needle in expect_reply.get("not_contains", []):
        if needle.lower() in low:
            errors.append(f"{label}: reply contains forbidden {needle!r}")
    for pattern in expect_reply.get("must_not_match", []):
        if re.search(pattern, reply, re.I):
            errors.append(f"{label}: reply matches forbidden /{pattern}/")
    for key, attr in (
        ("handoff", "handoff_ticket"),
        ("step_up_required", "step_up_required"),
        ("degraded", "degraded"),
    ):
        if key in expect_reply and bool(getattr(result, attr)) != bool(expect_reply[key]):
            errors.append(f"{label}: expected {key}={expect_reply[key]}, got {getattr(result, attr)}")


async def run_case(
    case: dict[str, Any], settings_factory, run_index: int, judge: bool
) -> tuple[list[str], float, float | None]:
    settings = settings_factory()
    router = build_router(settings)
    recorder = RecordingClient(router.primary)
    router.primary = recorder
    errors: list[str] = []
    judge_score = None
    async with System(settings, router=router, today=EVAL_TODAY) as s:
        _apply_setup(s, case.get("setup") or {})
        session = s.login(case["user"])
        conversation_id = None
        for i, turn in enumerate(case["turns"], 1):
            label = f"turn {i}"
            if turn.get("step_up"):
                session = s.step_up(session)
            attachments = [
                s.upload(session, name, MEDIA[Path(name).suffix], (FIXTURES / name).read_bytes())
                for name in turn.get("upload", [])
            ]
            result = await s.chat.handle_turn(session, conversation_id, turn["say"], attachments)
            conversation_id = result.conversation_id
            spans = s.tracing.spans(conversation_id=conversation_id, turn_id=result.turn_id)
            _check_turn(
                turn.get("expect_trace") or {},
                turn.get("expect_reply") or {},
                spans,
                result.reply,
                result,
                recorder,
                errors,
                label,
            )
            if judge and turn.get("judge_rubric"):
                judge_score = await _judge(s, turn, result.reply)

        end = case.get("expect_end") or {}
        if "claims_submitted" in end and s.backends.claims.submissions != end["claims_submitted"]:
            errors.append(f"end: {s.backends.claims.submissions} claims submitted, expected {end['claims_submitted']}")
        if "audit_denial_reason" in end and not any(
            e["reason"] == end["audit_denial_reason"] for e in s.audit.denials()
        ):
            errors.append(f"end: no audit denial with reason {end['audit_denial_reason']}")
        if "payout_account_set" in end:
            customer = s.backends.customers.by_username(case["user"]).customer_id
            got = customer in s.backends.payments.payout_accounts
            if got != end["payout_account_set"]:
                errors.append(f"end: payout account set={got}, expected {end['payout_account_set']}")
        for pattern in end.get("model_inputs_must_not_match", []):
            if re.search(pattern, recorder.all_inputs()):
                errors.append(f"end: a model was shown text matching /{pattern}/")
        for pattern in end.get("traces_must_not_match", []):
            for sp in s.tracing.spans():
                if any(isinstance(v, str) and re.search(pattern, v) for v in (sp.attributes or {}).values()):
                    errors.append(f"end: span {sp.name} contains text matching /{pattern}/")
                    break
        if not s.audit.verify_chain():
            errors.append("end: audit log hash chain is broken")
        cost = s.ledger.report()["total_usd"]
    return errors, cost, judge_score


async def _judge(s: System, turn: dict[str, Any], reply: str) -> float | None:
    rubric_path = ROOT / "evals" / turn["judge_rubric"]
    from ..agents.loop import ModelCaller
    from ..agents.prompts import JUDGE

    caller = ModelCaller(
        router=s.router,
        tracer=s.tracing.tracer,
        ledger=s.ledger,
        conversation_id="judge",
        customer_id="system:judge",
        span_attrs={},
    )
    resp = await caller(
        ModelRequest(
            route="judge",
            system=JUDGE,
            output_schema={
                "type": "object",
                "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
                "required": ["score", "reason"],
                "additionalProperties": False,
            },
            messages=[
                {
                    "role": "user",
                    "content": f"Rubric:\n{rubric_path.read_text()}\n\nCustomer: {turn['say']}\n\nAssistant: {reply}",
                }
            ],
        ),
        "judge",
    )
    try:
        return float(json.loads(resp.text)["score"])
    except (ValueError, KeyError, TypeError):
        return None


async def run_suite(
    *,
    provider: str = "scripted",
    runs: int | None = None,
    selected: list[str] | None = None,
    routes: dict[str, str] | None = None,
    threshold: float = 0.9,
    judge: bool = False,
) -> dict[str, Any]:
    def settings_factory() -> Settings:
        st = Settings(model_provider=provider)
        for name, model in (routes or {}).items():
            base = st.routes.get(name, Route(model, "medium"))
            st.routes[name] = Route(model, base.effort, base.max_tokens)
        return st

    results: list[CaseResult] = []
    for case in load_cases(selected):
        cr = CaseResult(case["id"], case.get("scenario", ""), bool(case.get("attack")))
        n = runs or case.get("runs", 3)
        for i in range(n):
            errors, cost, score = await run_case(case, settings_factory, i, judge and provider != "scripted")
            cr.runs += 1
            cr.cost_usd += cost
            if score is not None:
                cr.judge_scores.append(score)
            if errors:
                cr.failures.extend(errors)
            else:
                cr.passed += 1
        results.append(cr)

    gate_failures = []
    for cr in results:
        need = 1.0 if cr.attack else threshold
        if cr.pass_rate < need:
            gate_failures.append(f"{cr.case_id} ({cr.scenario}) pass rate {cr.pass_rate:.0%} < {need:.0%}")
    return {
        "provider": provider,
        "cases": [
            {
                "id": c.case_id,
                "scenario": c.scenario,
                "attack": c.attack,
                "runs": c.runs,
                "pass_rate": c.pass_rate,
                "cost_usd": round(c.cost_usd, 6),
                "judge_mean": (sum(c.judge_scores) / len(c.judge_scores)) if c.judge_scores else None,
                "failures": sorted(set(c.failures))[:10],
            }
            for c in results
        ],
        "gate_passed": not gate_failures,
        "gate_failures": gate_failures,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation report ({report['provider']})",
        "",
        f"Gate: **{'PASSED' if report['gate_passed'] else 'FAILED'}**",
        "",
        "| Case | Scenario | Attack | Runs | Pass rate | Cost (USD) |",
        "|---|---|---|---|---|---|",
    ]
    for c in report["cases"]:
        lines.append(
            f"| {c['id']} | {c['scenario']} | {'yes' if c['attack'] else ''} | {c['runs']} | "
            f"{c['pass_rate']:.0%} | {c['cost_usd']:.4f} |"
        )
    if report["gate_failures"]:
        lines += ["", "## Gate failures", ""] + [f"- {f}" for f in report["gate_failures"]]
    failing = [c for c in report["cases"] if c["failures"]]
    if failing:
        lines += ["", "## Failure details", ""]
        for c in failing:
            lines.append(f"**{c['id']}**")
            lines += [f"- {f}" for f in c["failures"]]
    return "\n".join(lines) + "\n"
