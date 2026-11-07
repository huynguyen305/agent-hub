"""Expose any ``LLMProvider`` as a LangChain chat model.

Lets LangChain components (prompt templates, retrievers, Guardrails AI's
LangChain integration) run on the Claude subscription. Tool calling is not
routed through this adapter: tools are explicit graph nodes (ADR-004).
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from hubkit.llm.base import LLMProvider


class ProviderChatModel(BaseChatModel):
    provider: Any  # LLMProvider (Any keeps pydantic from validating the ABC)

    @property
    def _llm_type(self) -> str:
        return f"hubkit:{getattr(self.provider, 'name', 'provider')}"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        system = "\n\n".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
        convo = "\n\n".join(
            f"{m.type.upper()}: {m.content}" for m in messages if not isinstance(m, SystemMessage)
        )
        result = self.provider.complete(system, convo)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result.text))])


def as_chat_model(provider: LLMProvider) -> ProviderChatModel:
    return ProviderChatModel(provider=provider)
