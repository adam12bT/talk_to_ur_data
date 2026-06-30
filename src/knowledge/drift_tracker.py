"""
src/knowledge/drift_tracker.py
Tracks RAG retrieval quality over time as the knowledge base grows.

Every time you call record_snapshot(), it runs a fixed set of benchmark
queries against the current KB, scores them, and appends the result to
drift_log.json. The drift page reads this log and plots the trend.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from src.knowledge.vectorstore import similarity_search, count_documents

DRIFT_LOG_PATH = Path("./drift_log.json")

# Default benchmark queries — edit these to match your domain
DEFAULT_BENCHMARKS = [
    "What is retrieval-augmented generation?",
    "How does a vector database work?",
    "What is semantic search?",
    "How are embeddings created?",
    "What is a language model?",
]


def _score_query(query: str) -> float:
    """Run a query and return the top-1 relevance score (0-1).
    ChromaDB returns L2 distance; we convert to similarity."""
    from src.knowledge.vectorstore import get_vectorstore
    store = get_vectorstore()
    try:
        results = store.similarity_search_with_score(query, k=1)
        if not results:
            return 0.0
        _, distance = results[0]
        # Convert L2 distance to a 0-1 similarity score
        similarity = max(0.0, 1.0 - float(distance))
        return round(similarity, 4)
    except Exception:
        return 0.0


def record_snapshot(
    benchmark_queries: list[str] | None = None,
    label: str = "",
) -> dict:
    """
    Run all benchmark queries against the current KB and save a snapshot.

    Args:
        benchmark_queries: list of queries to score. Uses DEFAULT_BENCHMARKS if None.
        label: optional human-readable label (e.g. the doc name just ingested)

    Returns:
        The snapshot dict that was saved.
    """
    queries = benchmark_queries or DEFAULT_BENCHMARKS
    chunk_count = count_documents()

    scores = {}
    for q in queries:
        scores[q] = _score_query(q)

    avg_score = round(sum(scores.values()) / len(scores), 4) if scores else 0.0

    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "unix_ts": time.time(),
        "chunk_count": chunk_count,
        "label": label,
        "avg_score": avg_score,
        "query_scores": scores,
    }

    # Load existing log
    log = _load_log()
    log.append(snapshot)

    DRIFT_LOG_PATH.write_text(json.dumps(log, indent=2))
    return snapshot


def _load_log() -> list[dict]:
    if DRIFT_LOG_PATH.exists():
        try:
            return json.loads(DRIFT_LOG_PATH.read_text())
        except Exception:
            return []
    return []


def get_drift_log() -> list[dict]:
    """Return all recorded snapshots, oldest first."""
    return _load_log()


def clear_drift_log() -> None:
    """Delete the drift log."""
    if DRIFT_LOG_PATH.exists():
        DRIFT_LOG_PATH.unlink()


def get_benchmark_queries() -> list[str]:
    return DEFAULT_BENCHMARKS
