import inspect
from datetime import date

import pytest

from claims_assistant.agents.gateway import McpGateway
from claims_assistant.config import Settings
from claims_assistant.system import System

TODAY = date(2026, 11, 6)


def pytest_collection_modifyitems(items):
    # Run every async test under anyio, so fixtures and tests share one task (the MCP session
    # managers use task groups, which must be exited in the task that entered them).
    for item in items:
        if inspect.iscoroutinefunction(getattr(item, "function", None)):
            item.add_marker(pytest.mark.anyio)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def system():
    async with System(Settings(model_provider="scripted"), today=TODAY) as s:
        yield s


def gateway_for(s: System, session_token: str) -> McpGateway:
    return McpGateway(
        idp=s.idp,
        session_token=session_token,
        catalog=s.catalog,
        urls=s.settings.service_urls,
        connect=s.connect,
        breakers=s.breakers,
        tracer=s.tracing.tracer,
        span_attrs={},
        tool_timeout_s=5,
    )
