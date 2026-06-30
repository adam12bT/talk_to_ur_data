"""
api.py — FastAPI REST API for the Knowledge Agent.
Run with: uvicorn api:app --reload

Endpoints:
  POST /chat         — send a question, get an answer
  POST /ingest/url   — add a URL to the knowledge base
  POST /ingest/file  — upload and ingest a document
  GET  /health       — liveness check
  GET  /stats        — knowledge base statistics
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

import sys
sys.path.insert(0, ".")

from src.agent.agent import get_agent
from src.knowledge.ingestion import ingest_file, ingest_url
from src.knowledge.vectorstore import count_documents

app = FastAPI(
    title="AI Knowledge Agent API",
    description="Autonomous research agent with RAG + web search",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    reset_memory: bool = False


class ChatStep(BaseModel):
    action: str
    input: str
    observation: str


class ChatResponse(BaseModel):
    answer: str
    steps: list[ChatStep]
    tools_used: list[str]


class IngestURLRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    chunks_added: int
    message: str


class HealthResponse(BaseModel):
    status: str
    model: str


class StatsResponse(BaseModel):
    chunks_in_knowledge_base: int


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    from src.config import LLM_MODEL
    return HealthResponse(status="ok", model=LLM_MODEL)


@app.get("/stats", response_model=StatsResponse)
async def stats():
    return StatsResponse(chunks_in_knowledge_base=count_documents())


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    try:
        agent = get_agent()
        if body.reset_memory:
            agent.reset_memory()
        response = agent.chat(body.question)
        return ChatResponse(
            answer=response.answer,
            steps=[
                ChatStep(
                    action=s["action"],
                    input=str(s["input"]),
                    observation=str(s["observation"])[:1000],
                )
                for s in response.steps
            ],
            tools_used=response.sources,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url_endpoint(body: IngestURLRequest):
    try:
        n = ingest_url(body.url)
        return IngestResponse(chunks_added=n, message=f"Successfully indexed {body.url}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file_endpoint(file: UploadFile = File(...)):
    suffix = Path(file.filename or "doc.txt").suffix
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        n = ingest_file(tmp_path)
        return IngestResponse(chunks_added=n, message=f"Successfully indexed {file.filename}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
