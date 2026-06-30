"""
src/config.py — Central configuration for all tuneable parameters.
Change values here; everything else reads from this file.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
# Get yours FREE at https://aistudio.google.com/app/apikey (no credit card)
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL: str = "gemini-2.5-flash"   # free tier: 15 req/min, 1M tokens/day
LLM_TEMPERATURE: float = 0.0
MAX_TOKENS: int = 4096

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # local, free, no API key needed

# ── Vector store ──────────────────────────────────────────────────────────────
# VECTOR_BACKEND: "chroma" (local disk, default for dev) or "qdrant" (cloud, persistent across restarts)
VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "chroma")

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")

# Qdrant Cloud — free tier at https://cloud.qdrant.io (no credit card needed)
QDRANT_URL: str = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base")

# ── Document chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE: int = 512          # tokens per chunk
CHUNK_OVERLAP: int = 50        # overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = 5                 # number of chunks to retrieve per query

# ── Agent ─────────────────────────────────────────────────────────────────────
MAX_ITERATIONS: int = 10       # max ReAct reasoning steps before forced stop
AGENT_VERBOSE: bool = True     # print reasoning steps to console

# ── Data paths ────────────────────────────────────────────────────────────────
DOCS_DIR: str = "./data/docs"