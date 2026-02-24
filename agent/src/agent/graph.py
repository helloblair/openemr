"""LangGraph agent graph — defines nodes, edges, and tool routing.

This module wires up a ReAct-style agent that uses Claude Sonnet as the
reasoning LLM and the three OpenEMR tools (patient_lookup, allergy_check,
drug_interaction_check) as callable actions.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

# Ensure the agent package is importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.tools import allergy_check, drug_interaction_check, patient_lookup

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

# ── Graph construction ───────────────────────────────────────────────────────

_TOOLS = [patient_lookup, allergy_check, drug_interaction_check]

_checkpointer = MemorySaver()

_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
)

graph = create_agent(
    _llm,
    _TOOLS,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=_checkpointer,
)


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

        print("=== Turn 1: Look up patient ===\n")
        resp1 = await run_agent("Look up patient John Smith", thread_id=tid)
        print(resp1)

        print("\n\n=== Turn 2: Follow-up using conversation history ===\n")
        resp2 = await run_agent("What are his allergies?", thread_id=tid)
        print(resp2)

    asyncio.run(_test())
