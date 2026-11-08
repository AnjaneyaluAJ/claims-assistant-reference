"""Tracing (Step 14), following the OpenTelemetry semantic conventions for generative AI.

Every turn is a tree of spans: input guardrails, each agent, each model call, each tool call,
each workflow step. Any OTLP backend can read them. Two production rules are built in:

* Personal data is redacted at a single choke point, the exporter wrapper, so no code path
  can forget to do it.
* The evaluation suite asserts on these same spans ("which tools ran, with which arguments"),
  so what we test is what we can see in production.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from ..pii.vault import redact

# GenAI semantic convention attribute names.
OP = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
REQ_MODEL = "gen_ai.request.model"
RESP_MODEL = "gen_ai.response.model"
IN_TOK = "gen_ai.usage.input_tokens"
OUT_TOK = "gen_ai.usage.output_tokens"
FINISH = "gen_ai.response.finish_reasons"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
AGENT_NAME = "gen_ai.agent.name"
CONVERSATION_ID = "gen_ai.conversation.id"


class RedactingExporter(SpanExporter):
    """Redacts personal data from every string attribute and event before export."""

    def __init__(self, inner: SpanExporter) -> None:
        self.inner = inner

    @staticmethod
    def _clean(value: Any) -> Any:
        if isinstance(value, str):
            return redact(value)
        if isinstance(value, (list, tuple)):
            return type(value)(redact(v) if isinstance(v, str) else v for v in value)
        return value

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        cleaned = [
            ReadableSpan(
                name=s.name,
                context=s.context,
                parent=s.parent,
                resource=s.resource,
                attributes={k: self._clean(v) for k, v in (s.attributes or {}).items()},
                events=s.events,
                links=s.links,
                kind=s.kind,
                status=s.status,
                start_time=s.start_time,
                end_time=s.end_time,
                instrumentation_scope=s.instrumentation_scope,
            )
            for s in spans
        ]
        return self.inner.export(cleaned)

    def shutdown(self) -> None:
        self.inner.shutdown()


class Tracing:
    def __init__(self, service_name: str = "claims-assistant", console: bool | None = None) -> None:
        self.memory = InMemorySpanExporter()
        self.provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        self.provider.add_span_processor(SimpleSpanProcessor(RedactingExporter(self.memory)))
        if console if console is not None else os.getenv("CLAIMS_TRACE_CONSOLE") == "1":
            self.provider.add_span_processor(SimpleSpanProcessor(RedactingExporter(ConsoleSpanExporter())))
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # optional dep

            self.provider.add_span_processor(SimpleSpanProcessor(RedactingExporter(OTLPSpanExporter())))
        self.tracer: Tracer = self.provider.get_tracer("claims_assistant")

    def spans(self, conversation_id: str | None = None, turn_id: str | None = None) -> list[ReadableSpan]:
        out = []
        for s in self.memory.get_finished_spans():
            attrs = s.attributes or {}
            if conversation_id and attrs.get(CONVERSATION_ID) != conversation_id:
                continue
            if turn_id and attrs.get("app.turn.id") != turn_id:
                continue
            out.append(s)
        return out


def as_json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)[:4000]
