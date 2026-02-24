"""FastAPI application entry point for the OpenEMR AI agent."""

from fastapi import FastAPI

app = FastAPI(title="OpenEMR Agent", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}
