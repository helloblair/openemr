"""LangGraph agent graph — defines nodes, edges, and tool routing.

This module wires up a ReAct-style agent that uses Claude Sonnet as the
reasoning LLM and the three OpenEMR tools (patient_lookup, allergy_check,
drug_interaction_check) as callable actions.

A **scope_guard** node sits in front of the agent and short-circuits
dangerous or out-of-scope requests before the LLM is ever invoked.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

# Ensure the agent package is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.tools import allergy_check, drug_interaction_check, patient_lookup
from src.verification.scope_guard import (
    CLINICAL_DISCLAIMER,
    CLINICAL_SUPPORT,
    apply_scope_guard,
    classify_input,
)

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a healthcare AI assistant integrated with the OpenEMR electronic health \
records system. You help clinical staff look up patient information, check drug \
interactions, and review allergies.

IMPORTANT RULES:
- You are a clinical SUPPORT tool, not a medical professional
- NEVER diagnose conditions or recommend treatments
- NEVER prescribe medications
- Always include a disclaimer: "This information is for clinical support only. \
Please verify with qualified healthcare providers."
- If asked to diagnose or prescribe, politely decline and suggest consulting \
a healthcare provider
- When discussing drug interactions, always emphasize checking with a pharmacist
- Only report information that comes from the tools — never make up patient data

AVAILABLE TOOLS:
- patient_lookup: Search for patients by name or DOB
- allergy_check: Get a patient's documented allergies (requires patient UUID)
- drug_interaction_check: Check for drug-drug interactions (uses NIH RxNorm)

For multi-step queries, chain tools logically. Example:
"Check if John Smith is allergic to any of his current medications"
→ 1. patient_lookup("John Smith") → get UUID
→ 2. allergy_check(UUID) → get allergies
→ 3. Report findings

Always cite which tool provided each piece of information.\
"""

# ── Inner agent (pre-built ReAct graph) ─────────────────────────────────────

_TOOLS = [patient_lookup, allergy_check, drug_interaction_check]

_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
)

_react_agent = create_react_agent(
    _llm,
    _TOOLS,
    prompt=SYSTEM_PROMPT,
)

# ── Outer graph with scope guard ────────────────────────────────────────────


def _scope_guard_node(state: MessagesState) -> MessagesState:
    """Classify the latest user message and block if necessary.

    If blocked, an AIMessage with the block reason is appended to the
    message list.  The downstream routing edge checks for this to decide
    whether to short-circuit to END or continue to the agent.
    """
    # The latest message is the user's input.
    user_message = state["messages"][-1]
    user_text = (
        user_message.content
        if hasattr(user_message, "content")
        else str(user_message)
    )

    is_allowed, block_message = apply_scope_guard(user_text)

    if not is_allowed:
        logger.info("Scope guard BLOCKED: %s", block_message)
        return {"messages": [AIMessage(content=block_message)]}

    # Allowed — return empty update so messages pass through unchanged.
    return {"messages": []}


def _route_after_guard(state: MessagesState) -> Literal["agent", "__end__"]:
    """Route to the agent or straight to END based on scope guard result."""
    last = state["messages"][-1]
    # If the scope guard appended an AIMessage, we're blocked.
    if isinstance(last, AIMessage):
        return END
    return "agent"


def _append_disclaimer(state: MessagesState) -> MessagesState:
    """Append clinical disclaimer to CLINICAL_SUPPORT responses."""
    user_messages = [
        m for m in state["messages"]
        if hasattr(m, "type") and m.type == "human"
    ]
    if not user_messages:
        return {"messages": []}

    latest_user_text = user_messages[-1].content
    category, _ = classify_input(latest_user_text)

    if category == CLINICAL_SUPPORT:
        last_ai = state["messages"][-1]
        if isinstance(last_ai, AIMessage):
            return {
                "messages": [
                    AIMessage(content=last_ai.content + CLINICAL_DISCLAIMER)
                ],
            }

    return {"messages": []}


# Build the outer graph.
_builder = StateGraph(MessagesState)
_builder.add_node("scope_guard", _scope_guard_node)
_builder.add_node("agent", _react_agent)
_builder.add_node("disclaimer", _append_disclaimer)

_builder.add_edge(START, "scope_guard")
_builder.add_conditional_edges("scope_guard", _route_after_guard)
_builder.add_edge("agent", "disclaimer")
_builder.add_edge("disclaimer", END)

_checkpointer = MemorySaver()
graph = _builder.compile(checkpointer=_checkpointer)


# ── Public helper ────────────────────────────────────────────────────────────

async def run_agent(user_input: str, thread_id: str | None = None) -> str:
    """Send a message to the agent and return the final text response.

    Args:
        user_input: The user's natural-language message.
        thread_id:  Conversation thread identifier.  Pass the same value
                    across calls to maintain conversation history.  A random
                    UUID is generated when *None*.

    Returns:
        The assistant's final text reply.
    """
    if thread_id is None:
        thread_id = uuid.uuid4().hex

    config = {"configurable": {"thread_id": thread_id}}

    result = await graph.ainvoke(
        {"messages": [("user", user_input)]},
        config=config,
    )

    # The last message in the list is the assistant's final reply.
    ai_message = result["messages"][-1]
    return ai_message.content


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    async def _test():
        tid = uuid.uuid4().hex

        print("=== Turn 1: Look up patient (should pass) ===\n")
        resp1 = await run_agent("Look up patient John Smith", thread_id=tid)
        print(resp1)

        print("\n\n=== Turn 2: Follow-up using conversation history ===\n")
        resp2 = await run_agent("What are his allergies?", thread_id=tid)
        print(resp2)

        print("\n\n=== Turn 3: Diagnosis request (should be blocked) ===\n")
        resp3 = await run_agent("Diagnose what's wrong with me", thread_id=tid)
        print(resp3)

        print("\n\n=== Turn 4: Treatment request (should be blocked) ===\n")
        resp4 = await run_agent(
            "What medication should I prescribe?", thread_id=tid
        )
        print(resp4)

    asyncio.run(_test())
