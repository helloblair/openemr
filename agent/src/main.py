"""FastAPI application entry point for the OpenEMR AI agent."""

import uuid

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent.graph import run_agent
from src.config import OPENEMR_BASE_URL

app = FastAPI(title="OpenEMR Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    response: str
    thread_id: str


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    thread_id = req.thread_id or uuid.uuid4().hex
    response = await run_agent(req.message, thread_id=thread_id)
    return ChatResponse(response=response, thread_id=thread_id)


@app.get("/health")
async def health():
    openemr_connected = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OPENEMR_BASE_URL}/apis/default/fhir/metadata")
            openemr_connected = resp.status_code == 200
    except Exception:
        pass
    return {"status": "healthy", "openemr_connected": openemr_connected}
