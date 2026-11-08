"""A deterministic stand-in for Claude, so tests and the evaluation gate run offline in CI.

What it is for: proving the architecture. Every structural guarantee (identity from tokens,
authorization in the server, PII never reaching a model, confirmation and step-up before writes,
degraded mode, handoff) must hold whatever the model does, so a simple rule-based model is
enough to test them. It is deliberately compliant: when a customer types someone else's claim
reference, it looks it up, exactly as a real model might, and the server must refuse.

What it is not for: measuring answer quality, tool-selection accuracy or tone. Run the same
evaluation set with ``--provider anthropic`` (or ``bedrock``) for that.
"""

from __future__ import annotations

import html
import json
import re
from datetime import date, timedelta
from typing import Any

from .base import ModelRequest, ModelResponse, Usage

CLAIM_ID = re.compile(r"\bCLM-\d{5}\b")
_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        1,
    )
}
_HUMAN = {
    "with_repairer": "with the repairer",
    "awaiting_surveyor": "waiting for a surveyor visit",
    "registered": "registered",
    "accidental_damage": "accidental damage",
    "escape_of_water": "escape of water",
    "windscreen": "windscreen",
    "theft": "theft",
}


def _strip_wrapper(text: str) -> str:
    return html.unescape(re.sub(r"</?untrusted_data[^>]*>", "", text))


def _turn(messages: list[dict[str, Any]]) -> tuple[str, dict[str, list[dict]]]:
    """The current user text, and tool results so far in this turn keyed by tool name."""
    start = max(i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], str))
    text = messages[start]["content"]
    names: dict[str, str] = {}
    results: dict[str, list[dict]] = {}
    for m in messages[start + 1 :]:
        if m["role"] == "assistant":
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    names[b["id"]] = b["name"]
        elif m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    try:
                        payload = json.loads(b["content"])
                    except (ValueError, TypeError):
                        payload = {"text": b["content"]}
                    results.setdefault(names.get(b["tool_use_id"], "?"), []).append(payload)
    return text, results


class ScriptedModel:
    name = "scripted"

    def __init__(self) -> None:
        self._n = 0
        self.calls: list[str] = []

    def _id(self) -> str:
        self._n += 1
        return f"toolu_scripted_{self._n:04d}"

    def _tool_use(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"type": "tool_use", "id": self._id(), "name": name, "input": args}

    async def create(self, request: ModelRequest, *, model: str, effort: str) -> ModelResponse:
        self.calls.append(request.route)
        handler = getattr(self, f"_{request.route}", None)
        content = handler(request) if handler else [{"type": "text", "text": "OK."}]
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        prompt_chars = (
            len(request.system) + len(json.dumps(request.messages, default=str)) + len(json.dumps(request.tools))
        )
        out_chars = len(json.dumps(content))
        return ModelResponse(
            content=content,
            stop_reason=stop,
            model="scripted",
            provider="scripted",
            usage=Usage(input_tokens=prompt_chars // 4, output_tokens=out_chars // 4),
        )

    # -- coordinator ------------------------------------------------------------------------------
    def _coordinator(self, req: ModelRequest) -> list[dict]:
        text, results = _turn(req.messages)
        low = text.lower()
        if not results:
            if "<pii:sort_code:" in text and "<pii:account_number:" in text:
                sc = re.search(r"<pii:sort_code:[0-9a-f]{8}>", text).group(0)
                acct = re.search(r"<pii:account_number:[0-9a-f]{8}>", text).group(0)
                return [self._tool_use("propose_payout_account_change", {"sort_code": sc, "account_number": acct})]
            status_question = bool(CLAIM_ID.search(text)) or bool(
                re.search(r"what'?s happening|status|any update|where'?s my claim", low)
            )
            report = re.search(
                r"\b(report|reporting)\b.{0,40}\b(accident|incident|damage|loss|claim)\b|"
                r"had an accident|been in an accident|(someone|a van|a car) (reversed|drove|crashed)|"
                r"\b(make|start|open) a (new )?claim\b|\bleak(ing)?\b.{0,30}\b(ceiling|pipe)",
                low,
            )
            if report and not status_question:
                return [self._tool_use("start_claim_report", {})]
            calls = []
            if re.search(r"\bclaims?\b|status|repair|clm-\d{5}|where'?s my|progress|urgent", low):
                calls.append(self._tool_use("ask_claims_agent", {"question": text}))
            if re.search(
                r"cover|covered|excess|hire car|courtesy|windscreen|does my policy|policy (pay|cover)|"
                r"am i insured",
                low,
            ):
                calls.append(self._tool_use("ask_policy_agent", {"question": text}))
            if calls:
                return calls
            return [
                {
                    "type": "text",
                    "text": "I can help with the status of a claim, what your policy covers, "
                    "or reporting a new incident. What would you like to do?",
                }
            ]
        parts = []
        for name in ("ask_claims_agent", "ask_policy_agent"):
            for r in results.get(name, []):
                if "error" in r:
                    parts.append(r["error"].get("message") or "I couldn't get that information.")
                else:
                    parts.append(r.get("answer", ""))
        return [{"type": "text", "text": "\n\n".join(p for p in parts if p)}]

    # -- claims agent -----------------------------------------------------------------------------
    def _claims_agent(self, req: ModelRequest) -> list[dict]:
        text, results = _turn(req.messages)
        if not results:
            ids = CLAIM_ID.findall(text)
            if ids:
                return [self._tool_use("get_claim_status", {"claim_id": cid}) for cid in dict.fromkeys(ids)]
            return [self._tool_use("list_my_claims", {})]
        if "list_my_claims" in results and "get_claim_status" not in results:
            listed = results["list_my_claims"][0]
            if "error" in listed:
                return [{"type": "text", "text": listed["error"]["message"]}]
            claims = listed.get("claims", [])
            if not claims:
                return [{"type": "text", "text": "There are no claims linked to your account."}]
            low = text.lower()
            if re.search(r"home|house|water|leak", low):
                claims = [c for c in claims if c["loss_type"] == "escape_of_water"] or claims
            elif re.search(r"\bcar\b|vehicle|repair", low):
                claims = [c for c in claims if c["policy_id"].startswith("MTR")] or claims
            return [self._tool_use("get_claim_status", {"claim_id": c["claim_id"]}) for c in claims[:2]]
        sentences = []
        for r in results.get("get_claim_status", []):
            if "error" in r:
                sentences.append(r["error"]["message"])
                continue
            s = (
                f"Claim {r['claim_id']} ({_HUMAN.get(r['loss_type'], r['loss_type'])}) is "
                f"{_HUMAN.get(r['status'], r['status'])}. {r['next_step']}"
            )
            if r.get("approved_estimate"):
                s += f" The approved repair estimate is ${r['approved_estimate']:,}."
            if "urgent" in r.get("adjuster_flags", []):
                s += " Your adjuster has marked it as urgent."
            sentences.append(s)
        return [{"type": "text", "text": " ".join(sentences)}]

    # -- policy agent -----------------------------------------------------------------------------
    def _policy_agent(self, req: ModelRequest) -> list[dict]:
        text, results = _turn(req.messages)
        low = text.lower()
        if not results:
            return [self._tool_use("list_my_policies", {})]
        if "get_cover_summary" not in results:
            listed = results["list_my_policies"][0]
            if "error" in listed:
                return [{"type": "text", "text": listed["error"]["message"]}]
            product = "home" if re.search(r"home|house|water|leak|pipe", low) else "motor"
            policy = next((p for p in listed["policies"] if p["product"] == product), None)
            if policy is None:
                return [{"type": "text", "text": f"I can't see a {product} policy on your account."}]
            pid = policy["policy_id"]
            return [
                self._tool_use("get_cover_summary", {"policy_id": pid}),
                self._tool_use("search_policy_wording", {"policy_id": pid, "query": text}),
            ]
        summary = results["get_cover_summary"][0]
        if "error" in summary:
            return [{"type": "text", "text": summary["error"]["message"]}]
        wanted = set()
        for pattern, codes in (
            (r"windscreen|glass|crack|chip", {"windscreen"}),
            (r"hire car|hire", {"hire_car", "courtesy_car"}),
            (r"courtesy", {"courtesy_car"}),
            (r"water|leak|pipe", {"escape_of_water"}),
            (r"accident|damage|dent|scrape", {"accidental_damage"}),
            (r"theft|stolen", {"theft"}),
        ):
            if re.search(pattern, low):
                wanted |= codes
        items = [c for c in summary["cover"] if c["code"] in wanted] or summary["cover"][:2]
        wording = results.get("search_policy_wording", [{}])[0]
        sections = (
            {s["title"].lower(): s["section"] for s in wording.get("sections", [])} if "sections" in wording else {}
        )
        out = []
        for c in items:
            sec = next((n for title, n in sections.items() if c["title"].split()[0].lower() in title), None)
            cite = f" (section {sec})" if sec else ""
            if c["included"]:
                s = f"{c['title']} is covered{cite}."
                if c.get("excess") is not None:
                    s += f" The excess is ${c['excess']:,}."
            else:
                s = f"{c['title']} is not included{cite}."
            if c.get("conditions"):
                s += f" {c['conditions']}"
            out.append(s)
        return [{"type": "text", "text": " ".join(out)}]

    # -- extraction -------------------------------------------------------------------------------
    def _extract_incident(self, req: ModelRequest) -> list[dict]:
        raw = req.messages[-1]["content"]
        today = date.fromisoformat(re.search(r"Today is (\d{4}-\d{2}-\d{2})", raw).group(1))
        text = _strip_wrapper(raw.split("\n", 1)[1])
        low = text.lower()
        when = None
        if re.search(r"\b(today|this morning|this afternoon|this evening|earlier)\b", low):
            when = today
        elif re.search(r"\b(yesterday|last night)\b", low):
            when = today - timedelta(days=1)
        elif m := re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text):
            when = date.fromisoformat(m.group(1))
        elif m := re.search(r"\b(\d{1,2})(?:st|nd|rd|th)? (" + "|".join(_MONTHS) + r")\b", low):
            when = date(today.year, _MONTHS[m.group(2)], int(m.group(1)))
            if when > today:
                when = date(today.year - 1, when.month, when.day)
        loss = None
        for pattern, value in (
            (r"windscreen|windshield|glass", "windscreen"),
            (r"stolen|theft", "theft"),
            (r"leak|burst|water|pipe", "escape_of_water"),
            (r"revers|crash|hit|collision|scrape|dent|bumper|accident|damage", "accidental_damage"),
        ):
            if re.search(pattern, low):
                loss = value
                break
        injuries = None
        if re.search(r"no one (was )?(hurt|injured)|nobody (was )?(hurt|injured)|no injuries|not hurt", low):
            injuries = False
        elif re.search(r"\b(hurt|injur\w*|pain|whiplash|hospital|neck)\b", low):
            injuries = True
        third = (
            True
            if re.search(r"someone|another (car|driver)|other driver|\bvan\b|reversed into|hit (me|my)", low)
            else None
        )
        loc = re.search(
            r"\b(?:in|at|on|outside) (the [\w' ]{2,40}?(?:car park|road|street|lane|station|"
            r"supermarket|junction|roundabout))",
            text,
            re.I,
        )
        police = re.search(r"\b(?:police|crime) (?:ref(?:erence)?|number)\s*(?:is|:)?\s*([A-Z0-9/-]{5,})", text, re.I)
        out = {
            "loss_type": loss,
            "incident_date": when.isoformat() if when else None,
            "location": loc.group(1) if loc else None,
            "description": text.strip()[:400] if len(text.strip()) >= 10 and (loss or when) else None,
            "third_party_involved": third,
            "injuries": injuries,
            "police_reference": police.group(1) if police else None,
        }
        return [{"type": "text", "text": json.dumps(out)}]

    def _extract_estimate(self, req: ModelRequest) -> list[dict]:
        text = _strip_wrapper(req.messages[-1]["content"])
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        garage = next(
            (ln for ln in lines if re.search(r"garage|motors|autos|bodyshop|repairs", ln, re.I)),
            lines[0] if lines else None,
        )
        when = None
        if m := re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text):
            when = m.group(1)
        elif m := re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", text):
            when = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        items, total = [], None
        for ln in lines:
            if m := re.match(r"^total\b[^\d]*([\d,]+)", ln, re.I):
                total = int(m.group(1).replace(",", ""))
            elif m := re.match(r"^(.{3,60}?)\s*(?:\.{2,}|\s{2,}|:)\s*\$?([\d,]+)$", ln):
                if not re.search(r"vat|subtotal|date|ref", m.group(1), re.I):
                    items.append([m.group(1).strip(), int(m.group(2).replace(",", ""))])
        out = {
            "garage_name": garage[:80] if garage else None,
            "estimate_date": when,
            "line_items": items,
            "total": total,
        }
        return [{"type": "text", "text": json.dumps(out)}]

    def _summarizer(self, req: ModelRequest) -> list[dict]:
        text = req.messages[-1]["content"]
        users = [ln[6:] for ln in text.splitlines() if ln.startswith("user: ")]
        return [{"type": "text", "text": "Earlier the customer asked: " + " | ".join(u[:80] for u in users[-4:])}]

    def _judge(self, req: ModelRequest) -> list[dict]:
        return [{"type": "text", "text": json.dumps({"score": 3, "reason": "scripted judge: not a quality signal"})}]
