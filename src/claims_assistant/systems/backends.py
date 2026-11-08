"""Simulated systems of record: policy admin, claims, payments and document storage.

In production these are the insurer's existing APIs. Here they are in-memory, with
two switches the evaluation suite uses: a maintenance mode (S10) and an injected
failure rate (to exercise retries and circuit breakers).
"""

from __future__ import annotations

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from ..domain import seed
from ..domain.models import Claim, ClaimDraft, ClaimEvent, Customer, EstimateFields, IncidentDetails, Policy


class BackendUnavailable(Exception):
    """The system of record is down or in maintenance."""


class NotFound(Exception):
    pass


@dataclass
class FaultInjection:
    maintenance: bool = False
    failure_rate: float = 0.0
    latency_s: float = 0.0
    rng: random.Random = field(default_factory=lambda: random.Random(7))  # noqa: S311 (fault injection)

    async def check(self, system: str) -> None:
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        if self.maintenance:
            raise BackendUnavailable(f"{system} is in a maintenance window")
        if self.failure_rate and self.rng.random() < self.failure_rate:
            raise BackendUnavailable(f"{system} returned 503")


class PolicySystem:
    def __init__(self) -> None:
        self.faults = FaultInjection()
        self._policies = {p.policy_id: p.model_copy(deep=True) for p in seed.POLICIES}
        self._wording = seed.POLICY_WORDING

    async def get(self, policy_id: str) -> Policy:
        await self.faults.check("policy-admin")
        try:
            return self._policies[policy_id]
        except KeyError:
            raise NotFound(policy_id) from None

    async def policies_for(self, customer_id: str) -> list[Policy]:
        await self.faults.check("policy-admin")
        return [p for p in self._policies.values() if any(x.customer_id == customer_id for x in p.parties)]

    async def wording(self, version: str) -> list[tuple[str, str, str]]:
        await self.faults.check("policy-admin")
        return self._wording.get(version, [])


class ClaimsSystem:
    def __init__(self) -> None:
        self.faults = FaultInjection()
        self._claims = {c.claim_id: c.model_copy(deep=True) for c in seed.CLAIMS}
        self._drafts: dict[str, ClaimDraft] = {}
        self._idempotency: dict[str, str] = {}  # idempotency key -> claim id
        self._seq = 93000
        self.submissions = 0  # number of claims actually created; tests assert on this

    async def get(self, claim_id: str) -> Claim:
        await self.faults.check("claims")
        try:
            return self._claims[claim_id]
        except KeyError:
            raise NotFound(claim_id) from None

    async def claims_for_policies(self, policy_ids: list[str]) -> list[Claim]:
        await self.faults.check("claims")
        return [c for c in self._claims.values() if c.policy_id in policy_ids]

    async def create_draft(self, policy_id: str, owner: str, incident: IncidentDetails) -> ClaimDraft:
        await self.faults.check("claims")
        draft = ClaimDraft(
            draft_id=f"DRF-{uuid.uuid4().hex[:8]}",
            policy_id=policy_id,
            owner=owner,
            incident=incident,
            created_at=datetime.now(UTC),
        )
        self._drafts[draft.draft_id] = draft
        return draft

    async def get_draft(self, draft_id: str) -> ClaimDraft:
        await self.faults.check("claims")
        try:
            return self._drafts[draft_id]
        except KeyError:
            raise NotFound(draft_id) from None

    async def attach_evidence(self, draft_id: str, document_id: str, estimate: EstimateFields | None) -> ClaimDraft:
        draft = await self.get_draft(draft_id)
        draft.evidence_document_ids.append(document_id)
        if estimate is not None:
            draft.estimate = estimate
        return draft

    async def submit(self, draft_id: str, idempotency_key: str) -> tuple[str, bool]:
        """Create a claim from a draft. Returns (claim_id, created_now). Repeats return the original."""
        await self.faults.check("claims")
        if idempotency_key in self._idempotency:
            return self._idempotency[idempotency_key], False
        draft = await self.get_draft(draft_id)
        self._seq += 1
        claim_id = f"CLM-{self._seq}"
        self._claims[claim_id] = Claim(
            claim_id=claim_id,
            policy_id=draft.policy_id,
            incident_party=draft.owner,
            loss_type=draft.incident.loss_type or "accidental_damage",
            incident_date=draft.incident.incident_date or date.today(),
            status="registered",
            next_step="An adjuster will contact you within two working days.",
            events=[ClaimEvent(at=datetime.now(UTC), text="First notice of loss via assistant.", source="assistant")],
        )
        self._idempotency[idempotency_key] = claim_id
        self.submissions += 1
        return claim_id, True


class PaymentsSystem:
    def __init__(self) -> None:
        self.faults = FaultInjection()
        self.payout_accounts: dict[str, tuple[str, str]] = {}  # customer -> (sort code, account number)

    async def set_payout_account(self, customer_id: str, sort_code: str, account_number: str) -> None:
        await self.faults.check("payments")
        self.payout_accounts[customer_id] = (sort_code, account_number)


@dataclass
class StoredDocument:
    document_id: str
    owner: str
    filename: str
    media_type: str
    content: bytes
    scanned_clean: bool


class DocumentStore:
    def __init__(self) -> None:
        self._docs: dict[str, StoredDocument] = {}

    def put(self, owner: str, filename: str, media_type: str, content: bytes, scanned_clean: bool) -> str:
        document_id = f"DOC-{uuid.uuid4().hex[:8]}"
        self._docs[document_id] = StoredDocument(document_id, owner, filename, media_type, content, scanned_clean)
        return document_id

    def get(self, document_id: str) -> StoredDocument:
        try:
            return self._docs[document_id]
        except KeyError:
            raise NotFound(document_id) from None


class CustomerDirectory:
    def __init__(self) -> None:
        self._by_id = {c.customer_id: c for c in seed.CUSTOMERS}

    def get(self, customer_id: str) -> Customer:
        return self._by_id[customer_id]

    def by_username(self, username: str) -> Customer | None:
        return next((c for c in self._by_id.values() if c.username == username), None)


@dataclass
class Backends:
    policies: PolicySystem = field(default_factory=PolicySystem)
    claims: ClaimsSystem = field(default_factory=ClaimsSystem)
    payments: PaymentsSystem = field(default_factory=PaymentsSystem)
    documents: DocumentStore = field(default_factory=DocumentStore)
    customers: CustomerDirectory = field(default_factory=CustomerDirectory)
