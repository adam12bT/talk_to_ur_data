"""
tests/test_agent.py
Integration tests for the KnowledgeAgent.
Requires ANTHROPIC_API_KEY to be set.
"""
from pathlib import Path
import os
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# Skip all tests in this module if no API key is configured
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping live agent tests",
)

from langchain_core.documents import Document
from src.agent.agent import KnowledgeAgent
from src.knowledge.vectorstore import add_documents, reset_vectorstore


@pytest.fixture
def agent_with_docs():
    """Create a fresh agent with seeded documents."""
    reset_vectorstore()
    add_documents([
        Document(
            page_content="The capital of France is Paris. Paris is known for the Eiffel Tower.",
            metadata={"source": "geography.txt"},
        ),
        Document(
            page_content="Python was created by Guido van Rossum and released in 1991.",
            metadata={"source": "programming.txt"},
        ),
    ])
    agent = KnowledgeAgent()
    yield agent
    reset_vectorstore()


class TestKnowledgeAgent:
    def test_agent_answers_from_knowledge_base(self, agent_with_docs):
        response = agent_with_docs.chat("What is the capital of France?")
        assert response.answer
        assert "Paris" in response.answer

    def test_agent_returns_steps(self, agent_with_docs):
        response = agent_with_docs.chat("Who created Python?")
        # Agent should have used at least the retrieval tool
        assert isinstance(response.steps, list)
        assert len(response.steps) >= 1

    def test_agent_memory_persists(self, agent_with_docs):
        agent_with_docs.chat("My name is Alice.")
        response = agent_with_docs.chat("What is my name?")
        assert "Alice" in response.answer

    def test_agent_memory_clears(self, agent_with_docs):
        agent_with_docs.chat("My favourite colour is blue.")
        agent_with_docs.reset_memory()
        response = agent_with_docs.chat("What is my favourite colour?")
        # After reset, agent should not know
        assert "blue" not in response.answer.lower() or "don't" in response.answer.lower()
