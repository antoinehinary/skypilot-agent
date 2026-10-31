"""A smolagents model that replays pre-written replies (no LLM).

Used to run mission reference solutions and to test the agent stack deterministically.
"""

from __future__ import annotations

from typing import Any

from smolagents import Model
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import TokenUsage


def code_reply(source: str) -> str:
    """A model reply containing one Python code block, in the format CodeAgent parses."""
    return f"Thought: next step.\n```python\n{source}\n```"


class ScriptedModel(Model):
    """Returns `replies` in order and records every prompt it receives."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(model_id="scripted")
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, messages: list, stop_sequences: Any = None, **kwargs: Any) -> ChatMessage:
        self.prompts.append(str(messages))
        reply = self.replies.pop(0) if self.replies else "Thought: out of script.\n"
        return ChatMessage(role=MessageRole.ASSISTANT, content=reply, token_usage=TokenUsage(10, 5))

    @classmethod
    def from_code(cls, steps: list[str]) -> ScriptedModel:
        """One code block per agent step."""
        return cls([code_reply(step) for step in steps])
