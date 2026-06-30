"""
src/tools/retrieval_tool.py
LangChain tool that lets the agent search the local knowledge base.
"""
from __future__ import annotations

from langchain.tools import tool

from src.knowledge.vectorstore import similarity_search


@tool
def retrieval_tool(query: str) -> str:
    """Search the local knowledge base for information relevant to the query.

    Use this tool FIRST whenever answering questions that might be covered
    by ingested documents. Returns the most relevant text passages found.
    Input should be a clear search query in natural language.
    """
    docs = similarity_search(query)

    if not docs:
        return "No relevant documents found in the knowledge base for this query."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_str = f" (page {page})" if page != "" else ""
        results.append(
            f"[Chunk {i} — source: {source}{page_str}]\n{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(results)
