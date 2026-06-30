"""
tests/test_ingestion.py
Unit tests for the document ingestion pipeline.
"""
import os
import tempfile
from pathlib import Path

import pytest

# Adjust path for running from project root
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge.ingestion import load_file, load_url, _get_splitter
from src.knowledge.vectorstore import reset_vectorstore


@pytest.fixture(autouse=True)
def clean_db():
    """Reset the vector store before each test."""
    reset_vectorstore()
    yield
    reset_vectorstore()


def make_temp_txt(content: str) -> str:
    """Write content to a temporary .txt file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestLoadFile:
    def test_load_txt(self):
        path = make_temp_txt("Hello world. This is a test document.")
        docs = load_file(path)
        assert len(docs) >= 1
        assert "Hello world" in docs[0].page_content
        os.unlink(path)

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_file("document.docx")


class TestTextSplitter:
    def test_splits_long_text(self):
        splitter = _get_splitter()
        long_text = "This is a sentence. " * 200
        from langchain_core.documents import Document
        docs = splitter.split_documents([Document(page_content=long_text)])
        assert len(docs) > 1

    def test_chunk_size_respected(self):
        from src.config import CHUNK_SIZE
        splitter = _get_splitter()
        long_text = "word " * 1000
        from langchain_core.documents import Document
        chunks = splitter.split_documents([Document(page_content=long_text)])
        for chunk in chunks:
            assert len(chunk.page_content) <= CHUNK_SIZE * 5  # generous bound


class TestVectorStoreIntegration:
    def test_add_and_retrieve(self):
        from src.knowledge.vectorstore import add_documents, similarity_search, count_documents
        from langchain_core.documents import Document

        docs = [
            Document(page_content="Python is a programming language.", metadata={"source": "test"}),
            Document(page_content="ChromaDB is a vector database.", metadata={"source": "test"}),
        ]
        added = add_documents(docs)
        assert added == 2
        assert count_documents() == 2

        results = similarity_search("programming language")
        assert len(results) > 0
        assert "Python" in results[0].page_content
