"""Step 3: measure before you split.

Runs a labelled set of single-turn questions against one agent with 6, 20 and 40 tools
attached, and records tool-selection accuracy and argument accuracy for each. The 6 real
tools are always present; the rest are realistic insurer tools from other domains. Use the
numbers to choose between one agent, retrieved tools, and domain agents.

Needs a real model. It makes len(questions) x 3 x runs requests:

    uv run python scripts/measure_tool_overload.py --provider anthropic --runs 3

Results go to results/tool_overload.json. Nothing in this repository claims a result from
this script until it has been run; see the README.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from claims_assistant.config import Settings
from claims_assistant.llm.base import ModelRequest
from claims_assistant.llm.router import build_router

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You are an insurance customer assistant. The customer is signed in. Call exactly one tool that best "
    "serves the request, with the right arguments. Do not answer without calling a tool."
)


def _tool(name: str, description: str, **props: str) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
            "required": list(props),
            "additionalProperties": False,
        },
    }


REAL = [
    _tool("list_my_policies", "List the signed-in customer's policies."),
    _tool("get_cover_summary", "What a policy covers, with excesses.", policy_id="Policy reference"),
    _tool("search_policy_wording", "Relevant policy wording sections.", policy_id="Policy reference", query="Topic"),
    _tool("list_my_claims", "List the customer's claims."),
    _tool("get_claim_status", "Status of one claim.", claim_id="Claim reference, e.g. CLM-77120"),
    _tool("start_claim_report", "Start reporting a new incident or loss."),
]

DISTRACTORS = [
    _tool("list_vehicles", "Vehicles on the customer's motor policies."),
    _tool("update_vehicle_registration", "Change a vehicle's registration.", vehicle_id="Vehicle", reg="New reg"),
    _tool("add_named_driver", "Add a driver to a motor policy.", policy_id="Policy", name="Driver name"),
    _tool("remove_named_driver", "Remove a driver.", policy_id="Policy", driver_id="Driver"),
    _tool("get_no_claims_discount", "Years of no claims discount on a policy.", policy_id="Policy"),
    _tool("get_renewal_quote", "Renewal price for a policy.", policy_id="Policy"),
    _tool("set_auto_renewal", "Turn auto-renewal on or off.", policy_id="Policy", enabled="true or false"),
    _tool("cancel_policy", "Start cancelling a policy.", policy_id="Policy"),
    _tool("get_payment_schedule", "Upcoming premium payments.", policy_id="Policy"),
    _tool("update_direct_debit", "Change the premium payment bank account.", policy_id="Policy"),
    _tool("get_documents", "List policy documents and certificates.", policy_id="Policy"),
    _tool("resend_certificate", "Email the insurance certificate again.", policy_id="Policy"),
    _tool("update_address", "Change the customer's address.", address="New address"),
    _tool("update_email", "Change the customer's email address.", email="New email"),
    _tool("get_breakdown_cover", "Breakdown cover details.", policy_id="Policy"),
    _tool("request_breakdown_assistance", "Send roadside assistance.", location="Where the car is"),
    _tool("find_approved_repairer", "Approved repairers near a postcode.", postcode="Postcode"),
    _tool("book_glass_repair", "Book a windscreen repair appointment.", claim_id="Claim", date="Date"),
    _tool("get_hire_car_booking", "Hire car booking for a claim.", claim_id="Claim"),
    _tool("upload_claim_document", "Attach a document to a claim.", claim_id="Claim", document_id="Document"),
    _tool("get_settlement_payment", "Settlement payments made on a claim.", claim_id="Claim"),
    _tool("log_complaint", "Record a formal complaint.", summary="Complaint"),
    _tool("get_complaint_status", "Status of a complaint.", complaint_id="Complaint"),
    _tool("request_callback", "Ask the contact centre to call back.", topic="Topic"),
    _tool("add_home_contents_item", "Add a high-value item to home cover.", item="Item", value="Value"),
    _tool("get_home_security_requirements", "Lock and alarm requirements.", policy_id="Policy"),
    _tool("report_travel_incident", "Report a travel insurance incident.", details="Details"),
    _tool("get_legal_expenses_cover", "Legal expenses add-on details.", policy_id="Policy"),
    _tool("add_policy_extra", "Add an optional extra to a policy.", policy_id="Policy", extra="Extra"),
    _tool("get_telematics_score", "Driving score from the telematics box.", policy_id="Policy"),
    _tool("update_mileage", "Change the declared annual mileage.", policy_id="Policy", miles="Miles"),
    _tool("get_fraud_hold_status", "Whether a claim is on hold for checks.", claim_id="Claim"),
    _tool("get_salvage_status", "Status of a written-off vehicle's salvage.", claim_id="Claim"),
    _tool("get_subrogation_status", "Recovery from the other party's insurer.", claim_id="Claim"),
]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "bedrock"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--route", default="coordinator", help="route whose model and effort to use")
    args = ap.parse_args()

    questions = yaml.safe_load((ROOT / "evals" / "tool_selection.yaml").read_text())
    router = build_router(Settings(model_provider=args.provider))
    results = {}
    for n_tools in (6, 20, 40):
        tools = REAL + DISTRACTORS[: n_tools - len(REAL)]
        right_tool = right_args = total = 0
        input_tokens = 0
        for q in questions:
            for _ in range(args.runs):
                resp = await router.create(
                    ModelRequest(
                        route=args.route, system=SYSTEM, tools=tools, messages=[{"role": "user", "content": q["say"]}]
                    )
                )
                input_tokens += resp.usage.input_tokens + resp.usage.cache_read_input_tokens
                total += 1
                call = next(iter(resp.tool_calls), None)
                if call and call["name"] == q["tool"]:
                    right_tool += 1
                    if all(call["input"].get(k) == v for k, v in (q.get("args") or {}).items()):
                        right_args += 1
        results[n_tools] = {
            "requests": total,
            "tool_accuracy": right_tool / total,
            "argument_accuracy": right_args / total,
            "mean_input_tokens": input_tokens / total,
        }
        print(n_tools, results[n_tools])
    out = ROOT / "results" / "tool_overload.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"provider": args.provider, "runs": args.runs, "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
