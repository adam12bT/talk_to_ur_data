"""
src/knowledge/vectorstore.py
Manages the vector store: initialisation, adding documents, and semantic
retrieval. Supports two backends, selected via VECTOR_BACKEND in .env:

  - "chroma" (default): local disk persistence at CHROMA_PERSIST_DIR.
    Good for local dev. Does NOT survive a Hugging Face Space restart —
    the filesystem is wiped on every redeploy/sleep cycle.

  - "qdrant": Qdrant Cloud, a free hosted vector database that persists
    independently of where your app runs. Use this for HF Spaces deployment
    so your knowledge base survives restarts. Free tier signup:
    https://cloud.qdrant.io (no credit card needed).

All other modules (ingestion.py, retrieval_tool.py, drift_tracker.py, the
Streamlit pages) call the functions below and never touch Chroma/Qdrant
directly — so switching backends is just an env var change, no other
code needs to change.
"""
from __future__ import annotations

from langchain_core.documents import Document

from src.config import (
    VECTOR_BACKEND,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    TOP_K,
)
from src.knowledge.embeddings import get_embeddings


# ── Backend-specific store builders ─────────────────────────────────────────

def _get_chroma_store():
    from langchain_community.vectorstores import Chroma
    from chromadb.config import Settings
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
        client_settings=Settings(allow_reset=True),
    )


def _get_qdrant_store():
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse

    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError(
            "VECTOR_BACKEND=qdrant but QDRANT_URL / QDRANT_API_KEY are not set. "
            "Get a free cluster at https://cloud.qdrant.io and set both in .env."
        )

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Create the collection on first use if it doesn't exist yet.
    try:
        client.get_collection(QDRANT_COLLECTION_NAME)
    except (UnexpectedResponse, ValueError, Exception):
        from qdrant_client.http.models import Distance, VectorParams
        # all-MiniLM-L6-v2 produces 384-dim vectors; adjust if you change EMBEDDING_MODEL
        sample_vec = get_embeddings().embed_query("dimension probe")
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=len(sample_vec), distance=Distance.COSINE),
        )

    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=get_embeddings(),
    )


def get_vectorstore():
    """Return (or create) the active vector store, based on VECTOR_BACKEND."""
    if VECTOR_BACKEND == "qdrant":
        return _get_qdrant_store()
    return _get_chroma_store()


# ── Shared operations (backend-agnostic) ────────────────────────────────────

def add_documents(documents: list[Document]) -> int:
    """Add a list of LangChain Documents to the vector store.

    Returns the number of chunks added. Safely no-ops on empty input
    instead of crashing on an empty embeddings upsert.
    """
    if not documents:
        return 0
    store = get_vectorstore()
    store.add_documents(documents)
    return len(documents)


def similarity_search(query: str, k: int = TOP_K) -> list[Document]:
    """Return the top-k most relevant document chunks for a query."""
    store = get_vectorstore()
    return store.similarity_search(query, k=k)


def get_retriever(k: int = TOP_K):
    """Return a LangChain retriever interface (used by tools and chains)."""
    return get_vectorstore().as_retriever(search_kwargs={"k": k})


def _chroma_client():
    """Return a ChromaDB PersistentClient compatible with ChromaDB 1.x.
    ChromaDB 1.x requires the default tenant/database to be created
    before any collection operations — EphemeralClient handles this
    automatically but PersistentClient does not on a fresh directory.
    """
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(allow_reset=True),
    )
    return client


def count_documents() -> int:
    """Return the total number of chunks stored."""
    if VECTOR_BACKEND == "qdrant":
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            info = client.get_collection(QDRANT_COLLECTION_NAME)
            return info.points_count or 0
        except Exception:
            return 0
    else:
        try:
            col = _chroma_client().get_collection(CHROMA_COLLECTION_NAME)
            return col.count()
        except Exception:
            return 0


def get_all_chunks() -> list[dict]:
    """Return every chunk in the store as [{id, content, metadata}, ...].

    Backend-agnostic — used by the Knowledge Base browser page so it
    doesn't depend on Chroma-specific internals like store._collection.
    """
    if VECTOR_BACKEND == "qdrant":
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        try:
            results = []
            next_offset = None
            while True:
                points, next_offset = client.scroll(
                    collection_name=QDRANT_COLLECTION_NAME,
                    limit=200,
                    offset=next_offset,
                    with_payload=True,
                )
                for p in points:
                    payload = p.payload or {}
                    results.append({
                        "id": str(p.id),
                        "content": payload.get("page_content", ""),
                        "metadata": payload.get("metadata", {}),
                    })
                if next_offset is None:
                    break
            return results
        except Exception:
            return []
    else:
        try:
            col = _chroma_client().get_collection(CHROMA_COLLECTION_NAME)
            raw = col.get(include=["documents", "metadatas"])
            return [
                {"id": doc_id, "content": doc, "metadata": meta or {}}
                for doc_id, doc, meta in zip(
                    raw.get("ids", []),
                    raw.get("documents", []),
                    raw.get("metadatas", []),
                )
            ]
        except Exception:
            return []


def reset_vectorstore() -> None:
    """Delete all documents from the collection (useful for testing)."""
    if VECTOR_BACKEND == "qdrant":
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            client.delete_collection(QDRANT_COLLECTION_NAME)
        except Exception:
            pass
    else:
        try:
            _chroma_client().delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass