"""
src/knowledge/embeddings.py
Wraps sentence-transformers so the rest of the codebase
never needs to import the library directly.
"""
from __future__ import annotations
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached embedding model instance.

    Uses all-MiniLM-L6-v2 by default — 80 MB, runs on CPU, no API key.
    Override via EMBEDDING_MODEL in config.py.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
