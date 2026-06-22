from abc import ABC, abstractmethod
from typing import Callable

from .types import LLMResponse, Message, ToolDefinition


class LLMClient(ABC):

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDefinition] | None = None,
        stream: bool = True,
        on_chunk: Callable[[str, str], None] | None = None,
    ) -> LLMResponse:
        ...

    @abstractmethod
    def supports_tools(self) -> bool:
        ...


def create_client(provider: str, base_url: str, api_key: str, model: str) -> LLMClient:
    """Factory: create the right client based on provider name."""
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(base_url, api_key, model)
    elif provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(base_url, api_key, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
