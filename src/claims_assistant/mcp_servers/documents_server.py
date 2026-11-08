"""Documents MCP server: turns an uploaded document into typed fields (Step 9).

The document's free text never leaves this server. The extractor model runs with no
tools, its output is validated against a schema, and only the typed fields come back.
Whatever an uploaded estimate says to "the assistant", the most it can do is put a wrong
value in a typed field, which the workflow shows to the customer before anything is submitted.
"""

from __future__ import annotations

import io
import re

from mcp.server.mcpserver import MCPServer

from ..systems.backends import NotFound
from .common import (
    NOT_LINKED,
    JwtTokenVerifier,
    ServerContext,
    auth_settings,
    current_caller,
    require_scope,
    tool_error,
)

SERVICE = "documents-mcp"

# Signals worth recording on the trace. They do not block anything on their own, because the
# structural defences do not depend on detecting the attack.
_INJECTION_SIGNALS = re.compile(
    r"(ignore (all|previous|prior) instructions|note to (the )?assistant|system prompt|you are now|"
    r"pre-?approved|mark (the )?claim as|tell the customer)",
    re.IGNORECASE,
)


def document_text(media_type: str, content: bytes) -> str:
    if media_type == "application/pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return content.decode("utf-8", errors="replace")


def build(ctx: ServerContext) -> MCPServer:
    url = ctx.urls[SERVICE]
    srv = MCPServer(
        SERVICE,
        instructions="Typed field extraction from documents the signed-in customer uploaded.",
        token_verifier=JwtTokenVerifier(ctx.idp, url),
        auth=auth_settings(url),
    )

    @srv.tool()
    async def extract_estimate(document_id: str) -> dict:
        """Extract garage name, date, line items and total from an uploaded repair estimate.
        Returns typed fields only; the document text is never returned."""
        caller = current_caller(ctx)
        if err := require_scope(caller, "document:read"):
            return err
        try:
            doc = ctx.backends.documents.get(document_id)
        except NotFound:
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        if doc.owner != caller.customer_id:
            ctx.audit.write(
                subject=caller.customer_id,
                action="document:read",
                resource=document_id,
                allowed=False,
                reason="not_owner",
                service=SERVICE,
                session_id=caller.session_id,
            )
            return tool_error("not_found_or_not_permitted", NOT_LINKED)
        if not doc.scanned_clean:
            return tool_error("rejected", "This file could not be accepted.")
        text = document_text(doc.media_type, doc.content)
        fields = await ctx.extractor(text, caller.session_id)
        warnings = []
        if _INJECTION_SIGNALS.search(text):
            warnings.append("document_contains_instruction_like_text")
        if fields.line_items and fields.total is not None and sum(v for _, v in fields.line_items) != fields.total:
            warnings.append("line_items_do_not_sum_to_total")
        return {"document_id": document_id, "fields": fields.model_dump(mode="json"), "warnings": warnings}

    return srv
