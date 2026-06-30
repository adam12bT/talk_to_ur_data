"""
src/agent/agent.py
The main ReAct agent using Google Gemini (free tier).
Compatible with LangChain >= 0.3 + LangGraph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.agent.memory import AgentMemory
from src.config import (
    AGENT_VERBOSE,
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_ITERATIONS,
    MAX_TOKENS,
)
from src.tools.file_reader_tool import file_reader_tool
from src.tools.retrieval_tool import retrieval_tool
from src.tools.web_search_tool import web_search_tool

SYSTEM_PROMPT = """You are an expert AI research assistant with access to a local knowledge base and the live web.

Your goal is to give accurate, well-sourced answers by reasoning step by step and using your tools effectively.

Guidelines:
- Always check the local knowledge base FIRST using the retrieval_tool
- Use web_search_tool when the knowledge base doesn't have enough information
- Use file_reader_tool when you need to read a specific document in full
- Cite your sources in the final answer (mention document names or URLs)
- Be honest when you are uncertain; never fabricate facts
- If tools return conflicting information, mention it and explain your reasoning"""


@dataclass
class AgentResponse:
    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


class KnowledgeAgent:
    """Autonomous research agent with RAG + web search, powered by Gemini (free)."""

    def __init__(self):
        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set.\n"
                "Get your FREE key at: https://aistudio.google.com/app/apikey\n"
                "Then add it to your .env file: GOOGLE_API_KEY=AIza..."
            )

        self._llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            google_api_key=GOOGLE_API_KEY,
        )

        self._tools: list[BaseTool] = [
            retrieval_tool,
            web_search_tool,
            file_reader_tool,
        ]

        self._memory = AgentMemory()
        self._graph = create_react_agent(
            self._llm,
            self._tools,
            prompt=SystemMessage(content=SYSTEM_PROMPT),
        )

    def chat(self, question: str) -> AgentResponse:
        # Get prior history from memory, then append new question
        history = self._memory.get_history()
        messages = history + [HumanMessage(content=question)]

        config = {"recursion_limit": MAX_ITERATIONS * 2}
        result = self._graph.invoke({"messages": messages}, config=config)

        all_messages = result.get("messages", [])

        # Final answer = last AIMessage with non-empty content
        answer = ""
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content:
                raw = msg.content
                # Newer LangChain/Gemini returns content as a list of blocks
                # e.g. [{'type': 'text', 'text': '...', 'extras': {...}}]
                if isinstance(raw, list):
                    answer = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in raw
                        if block
                    ).strip()
                else:
                    answer = str(raw)
                break

        # Collect tool steps from ToolMessage objects
        steps = []
        sources = []
        for msg in all_messages:
            # ToolMessages have a .name attribute set to the tool name
            if hasattr(msg, "name") and msg.name and not isinstance(msg, (HumanMessage, AIMessage)):
                steps.append({
                    "action": msg.name,
                    "input": str(getattr(msg, "tool_call_id", "")),
                    "observation": str(msg.content)[:800],
                })
                if msg.name not in sources:
                    sources.append(msg.name)

        self._memory.add_interaction(question, answer)

        return AgentResponse(answer=answer, steps=steps, sources=sources)

    def reset_memory(self) -> None:
        self._memory.clear()

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools


_agent_instance: KnowledgeAgent | None = None


def get_agent() -> KnowledgeAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = KnowledgeAgent()
    return _agent_instance