"""
src/tools/web_search_tool.py
LangChain tool that searches the web via DuckDuckGo.
Completely free — no API key, no account needed.
"""
from __future__ import annotations

from langchain.tools import tool
from ddgs import DDGS

@tool
def web_search_tool(query: str) -> str:
    """Search the live web for up-to-date information not in the knowledge base.

    Use this tool when:
    - The local knowledge base returns nothing relevant
    - The question involves recent events or real-time data
    - You need to verify or supplement retrieved knowledge
    Input should be a concise search query (under 10 words works best).
    """
    try:
        results = []
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))

        if not hits:
            return "No results found for this query."

        for i, hit in enumerate(hits, 1):
            results.append(
                f"[Result {i}] {hit.get('title', 'No title')}\n"
                f"URL: {hit.get('href', '')}\n"
                f"{hit.get('body', '').strip()}"
            )

        return "\n\n---\n\n".join(results)

    except Exception as e:
        return f"Web search failed: {e}"
