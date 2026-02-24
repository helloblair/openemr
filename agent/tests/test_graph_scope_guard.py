"""Integration tests: scope guard wired into the LangGraph agent graph.

These tests verify the graph-level routing (blocked inputs short-circuit,
allowed inputs reach the agent node) WITHOUT calling the real LLM.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.agent.graph import graph, run_agent
from src.verification.scope_guard import (
    BLOCK_MESSAGES,
    CLINICAL_DISCLAIMER,
    DIAGNOSIS_REQUEST,
    OUT_OF_SCOPE,
    TREATMENT_REQUEST,
)


# ── Blocked requests short-circuit (no LLM call) ────────────────────────────


@pytest.mark.asyncio
async def test_diagnosis_blocked_no_llm_call():
    """A diagnosis request should return the block message directly."""
    resp = await run_agent("Diagnose what's wrong with me")
    assert resp == BLOCK_MESSAGES[DIAGNOSIS_REQUEST]


@pytest.mark.asyncio
async def test_treatment_blocked_no_llm_call():
    """A treatment/prescribe request should return the block message."""
    resp = await run_agent("What medication should I prescribe?")
    assert resp == BLOCK_MESSAGES[TREATMENT_REQUEST]


@pytest.mark.asyncio
async def test_out_of_scope_blocked():
    """An unrelated query should return the out-of-scope message."""
    resp = await run_agent("Write me a poem about cats")
    assert resp == BLOCK_MESSAGES[OUT_OF_SCOPE]


# ── Allowed requests reach the agent ────────────────────────────────────────

# We patch the inner react agent so we don't need real API keys.


@pytest.mark.asyncio
async def test_data_retrieval_reaches_agent():
    """A data-retrieval query should pass through the scope guard."""
    with patch(
        "src.agent.graph._react_agent.ainvoke",
        new_callable=AsyncMock,
    ) as mock_agent:
        from langchain_core.messages import AIMessage, HumanMessage

        mock_agent.return_value = {
            "messages": [
                HumanMessage(content="Look up patient John Smith"),
                AIMessage(content="Found patient John Smith."),
            ],
        }

        resp = await run_agent("Look up patient John Smith")
        # The agent was invoked (not short-circuited).
        mock_agent.assert_called_once()
        assert "John Smith" in resp


@pytest.mark.asyncio
async def test_clinical_support_reaches_agent_with_disclaimer():
    """A clinical-support query should pass through and get a disclaimer."""
    with patch(
        "src.agent.graph._react_agent.ainvoke",
        new_callable=AsyncMock,
    ) as mock_agent:
        from langchain_core.messages import AIMessage, HumanMessage

        mock_agent.return_value = {
            "messages": [
                HumanMessage(content="Check drug interaction between aspirin and warfarin"),
                AIMessage(content="There is a known interaction."),
            ],
        }

        resp = await run_agent(
            "Check drug interaction between aspirin and warfarin"
        )
        mock_agent.assert_called_once()
        assert "known interaction" in resp
        assert "Disclaimer" in resp or "clinical support" in resp


# ── Multiple blocked requests don't leak state ──────────────────────────────


@pytest.mark.asyncio
async def test_sequential_blocks_independent():
    """Two blocked requests in the same thread should each return properly."""
    tid = "test-thread-blocks"

    resp1 = await run_agent("Diagnose this rash", thread_id=tid)
    assert resp1 == BLOCK_MESSAGES[DIAGNOSIS_REQUEST]

    resp2 = await run_agent("Prescribe antibiotics", thread_id=tid)
    assert resp2 == BLOCK_MESSAGES[TREATMENT_REQUEST]
