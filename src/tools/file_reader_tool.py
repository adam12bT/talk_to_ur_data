"""
src/tools/file_reader_tool.py
LangChain tool that lets the agent read a file from disk on demand,
without pre-ingesting it into the vector store.
"""
from __future__ import annotations

from pathlib import Path

from langchain.tools import tool

from src.knowledge.ingestion import load_file, load_url


@tool
def file_reader_tool(path_or_url: str) -> str:
    """Read and return the full text content of a local file or a web URL.

    Use this when you need to read an entire document rather than searching
    for specific chunks. Supports .pdf, .txt, .md files and http/https URLs.
    Input: an absolute or relative file path, or a full URL starting with http.
    """
    path_or_url = path_or_url.strip()
    try:
        if path_or_url.startswith("http"):
            docs = load_url(path_or_url)
        else:
            path = Path(path_or_url)
            if not path.exists():
                return f"File not found: {path_or_url}"
            docs = load_file(path)

        if not docs:
            return "Could not extract any text from the provided source."

        full_text = "\n\n".join(doc.page_content for doc in docs)
        # Truncate very long documents to avoid overflowing context
        if len(full_text) > 12_000:
            full_text = full_text[:12_000] + "\n\n[... document truncated ...]"
        return full_text

    except Exception as e:
        return f"Error reading {path_or_url}: {e}"
