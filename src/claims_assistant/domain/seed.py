"""Fictional seed data. Every person, policy and amount here is invented for teaching.

Households are chosen so that each scenario in the blog has real data behind it:

* Maya Fernandes: S1, S2, S3, S4, S8, S9, S10 (policyholder, motor + home)
* Tom Becker:     the "neighbour" in S6; Maya must never see his claim
* Arjun and Priya Shah: S5, a joint policy where Priya is a named driver
"""

from __future__ import annotations

from datetime import date, datetime

from .models import (
    Claim,
    ClaimEvent,
    CoverItem,
    Customer,
    Delegation,
    Party,
    Policy,
    Role,
)

CUSTOMERS = [
    Customer(customer_id="cust-1001", display_name="Maya Fernandes", username="maya", phone_last4="4410"),
    Customer(customer_id="cust-2001", display_name="Tom Becker", username="tom", phone_last4="9032"),
    Customer(customer_id="cust-3001", display_name="Arjun Shah", username="arjun", phone_last4="1188"),
    Customer(customer_id="cust-3002", display_name="Priya Shah", username="priya", phone_last4="5521"),
]

MOTOR_COVER = [
    CoverItem(code="accidental_damage", title="Accidental damage", included=True, excess=350),
    CoverItem(
        code="windscreen",
        title="Windscreen and window glass",
        included=True,
        excess=100,
        conditions="Repairs have no excess. Replacements carry a 100 excess. Use the approved glass network.",
    ),
    CoverItem(
        code="courtesy_car",
        title="Courtesy car",
        included=True,
        conditions="Provided only while the car is at an approved repairer, subject to availability.",
    ),
    CoverItem(
        code="hire_car",
        title="Hire car after a claim",
        included=False,
        conditions="Not included. Available as the Hire Car Plus add-on.",
    ),
    CoverItem(code="theft", title="Fire and theft", included=True, excess=350),
]

HOME_COVER = [
    CoverItem(code="escape_of_water", title="Escape of water", included=True, excess=450, limit=50000),
    CoverItem(
        code="accidental_damage",
        title="Accidental damage to buildings",
        included=False,
        conditions="Not included on this policy.",
    ),
]

POLICIES = [
    Policy(
        policy_id="MTR-5530217",
        product="motor",
        status="active",
        start=date(2026, 3, 1),
        end=date(2027, 2, 28),
        vehicle="2021 Skoda Octavia, AB21 XYZ",
        parties=[Party(customer_id="cust-1001", role=Role.POLICYHOLDER)],
        cover=MOTOR_COVER,
        wording_version="MTR-2026.1",
    ),
    Policy(
        policy_id="HOM-2290031",
        product="home",
        status="active",
        start=date(2026, 1, 15),
        end=date(2027, 1, 14),
        address_district="Riverside",
        parties=[Party(customer_id="cust-1001", role=Role.POLICYHOLDER)],
        cover=HOME_COVER,
        wording_version="HOM-2026.1",
    ),
    Policy(
        policy_id="MTR-4471902",
        product="motor",
        status="active",
        start=date(2026, 5, 10),
        end=date(2027, 5, 9),
        vehicle="2019 Ford Focus, CD19 KLM",
        parties=[Party(customer_id="cust-2001", role=Role.POLICYHOLDER)],
        cover=MOTOR_COVER,
        wording_version="MTR-2026.1",
    ),
    Policy(
        policy_id="MTR-6120458",
        product="motor",
        status="active",
        start=date(2026, 7, 1),
        end=date(2027, 6, 30),
        vehicle="2022 Toyota Corolla, EF22 PQR",
        parties=[
            Party(customer_id="cust-3001", role=Role.POLICYHOLDER),
            Party(customer_id="cust-3002", role=Role.NAMED_DRIVER),
        ],
        delegation=Delegation(named_drivers_see_all_claims=False),
        cover=MOTOR_COVER,
        wording_version="MTR-2026.1",
    ),
]

CLAIMS = [
    Claim(
        claim_id="CLM-77120",
        policy_id="MTR-5530217",
        incident_party="cust-1001",
        loss_type="accidental_damage",
        incident_date=date(2026, 10, 14),
        status="with_repairer",
        next_step="Repairer expects to finish by 12 November.",
        approved_estimate=2150,
        adjuster_flags=["urgent"],
        events=[
            ClaimEvent(at=datetime(2026, 10, 14, 9, 5), text="Claim reported by phone.", source="phone"),
            ClaimEvent(at=datetime(2026, 10, 30, 15, 42), text="Marked urgent after customer call.", source="adjuster"),
            ClaimEvent(at=datetime(2026, 11, 2, 11, 0), text="Repair estimate approved.", source="adjuster"),
        ],
    ),
    Claim(
        claim_id="CLM-90107",
        policy_id="HOM-2290031",
        incident_party="cust-1001",
        loss_type="escape_of_water",
        incident_date=date(2026, 10, 2),
        status="awaiting_surveyor",
        next_step="Surveyor visit booked for 10 November.",
    ),
    Claim(
        claim_id="CLM-88213",
        policy_id="MTR-4471902",
        incident_party="cust-2001",
        loss_type="accidental_damage",
        incident_date=date(2026, 10, 20),
        status="with_repairer",
        next_step="Parts on order.",
        approved_estimate=3840,
    ),
    Claim(
        claim_id="CLM-90311",
        policy_id="MTR-6120458",
        incident_party="cust-3001",
        loss_type="accidental_damage",
        incident_date=date(2026, 11, 5),
        status="registered",
        next_step="An adjuster will call within two working days.",
    ),
]

# Policy wording, split into sections so the policy MCP server can retrieve the
# relevant ones instead of sending the whole document to the model.
POLICY_WORDING: dict[str, list[tuple[str, str, str]]] = {
    "MTR-2026.1": [
        (
            "4.1",
            "Windscreen and window glass",
            "We cover repair or replacement of broken or chipped windscreen and window glass. "
            "A repair has no excess. A replacement carries the windscreen excess shown in your schedule. "
            "You must use our approved glass network, or we will pay up to 150 for glass work done elsewhere. "
            "A windscreen claim does not affect your no claims discount.",
        ),
        (
            "4.2",
            "Courtesy car",
            "While your car is being repaired by one of our approved repairers after a claim we cover, "
            "we will provide a small courtesy car, subject to availability. "
            "A courtesy car is not provided if your car is written off or stolen and not recovered.",
        ),
        (
            "4.3",
            "Hire car after a claim",
            "A like-for-like hire car is only provided if Hire Car Plus is shown in your schedule. "
            "It covers up to 21 days while your car is repaired, written off or stolen.",
        ),
        (
            "5.1",
            "Accidental damage",
            "We cover accidental damage to your car, including damage caused by fire, vandalism or collision. "
            "The accidental damage excess in your schedule applies to each claim.",
        ),
        (
            "5.4",
            "What you must do after an accident",
            "Report the accident to us as soon as possible. Do not admit fault or agree to pay anyone. "
            "Get the names, addresses and insurance details of everyone involved and any witnesses.",
        ),
    ],
    "HOM-2026.1": [
        (
            "3.2",
            "Escape of water",
            "We cover damage caused by water escaping from a fixed water or heating installation. "
            "We do not cover the cost of repairing the part that failed, unless it is accidentally damaged.",
        ),
    ],
}
