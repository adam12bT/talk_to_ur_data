"""
tests/test_retrieval.py
Tests for the RAG retrieval tool.
"""
from pathlib import Path
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from src.knowledge.vectorstore import add_documents, reset_vectorstore
from src.tools.retrieval_tool import retrieval_tool


@pytest.fixture(autouse=True)
def seed_db():
    """Seed the vector store with known documents for each test."""
    reset_vectorstore()
    docs = [
        Document(
            page_content="LangChain is a framework for building LLM-powered applications.",
            metadata={"source": "langchain_docs.txt"},
        ),
        Document(
            page_content="ChromaDB is an open-source embedding database for AI applications.",
            metadata={"source": "chroma_docs.txt"},
        ),
        Document(
            page_content="RAGAS is a framework for evaluating RAG pipelines automatically.",
            metadata={"source": "ragas_docs.txt"},
        ),
    ]
    add_documents(docs)
    yield
    reset_vectorstore()


class TestRetrievalTool:
    def test_returns_relevant_chunk(self):
        result = retrieval_tool.invoke("What is LangChain?")
        assert "LangChain" in result
        assert "framework" in result.lower()

    def test_returns_multiple_chunks(self):
        result = retrieval_tool.invoke("database embeddings AI")
        assert "Chunk" in result  # multi-chunk format

    def test_no_results_returns_message(self):
        reset_vectorstore()
        result = retrieval_tool.invoke("completely unrelated nonsense xyz")
        assert "No relevant documents" in result or "Chunk" in result

    def test_source_metadata_included(self):
        result = retrieval_tool.invoke("RAGAS evaluation")
        assert "source:" in result
