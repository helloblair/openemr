"""Streamlit chat UI for interacting with the OpenEMR AI agent."""

import asyncio
import sys
import uuid
from pathlib import Path

# Ensure the agent package root is on sys.path so `src.*` imports resolve
# when running via: cd agent && streamlit run frontend/streamlit_app.py
_AGENT_ROOT = str(Path(__file__).resolve().parent.parent)
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

import streamlit as st

from src.agent.graph import run_agent

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OpenEMR Healthcare Agent",
    page_icon="🏥",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏥 OpenEMR Healthcare Agent")
    st.markdown(
        "An AI-powered clinical support assistant connected to OpenEMR. "
        "It can look up patients, check drug interactions, and review allergies."
    )
    st.divider()
    st.markdown("**Example queries to try:**")
    st.markdown(
        "- *Look up patient John Smith*\n"
        "- *Check interactions between aspirin and warfarin*\n"
        "- *What allergies does Jane Doe have?*"
    )

# ── Session state ────────────────────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Handle user input ───────────────────────────────────────────────────────

if prompt := st.chat_input("Ask the healthcare agent…"):
    # Display & store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = asyncio.run(
                run_agent(prompt, thread_id=st.session_state.thread_id)
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
