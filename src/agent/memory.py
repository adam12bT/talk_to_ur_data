"""
src/agent/memory.py
Manages conversation history for the agent.
Updated for LangChain >= 0.3 — ConversationBufferWindowMemory moved to langchain-community.
"""
from __future__ import annotations

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage


class AgentMemory:
    """Simple windowed conversation memory compatible with LangChain 0.3+."""

    def __init__(self, window_size: int = 10):
        self._history = ChatMessageHistory()
        self._window_size = window_size

    def add_interaction(self, human_input: str, ai_output: str) -> None:
        self._history.add_user_message(human_input)
        self._history.add_ai_message(ai_output)
        messages = self._history.messages
        max_msgs = self._window_size * 2
        if len(messages) > max_msgs:
            self._history.messages = messages[-max_msgs:]

    def get_history(self) -> list:
        return self._history.messages

    def get_history_as_string(self) -> str:
        messages = self.get_history()
        if not messages:
            return "(no prior conversation)"
        lines = []
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._history.clear()

    @property
    def langchain_memory(self):
        """Expose the raw ChatMessageHistory for use in the agent."""
        return self._history