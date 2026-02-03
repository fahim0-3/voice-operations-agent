"""
LLM provider factory - the one place that turns a provider name +
settings into a concrete LLMProvider instance.

why a factory function instead of importing each provider class directly
in main.py/llm.py: it keeps provider-specific imports (boto3, ollama,
openai) lazy and localized - selecting "ollama" never has to import
boto3, and vice versa. It's also the single place that knows the full
list of valid provider names, which is what /providers and error messages
both read from.
"""

from __future__ import annotations

from app.providers.base import LLMProvider, ToolCall

VALID_PROVIDERS = ("bedrock", "ollama", "openai")


def create_provider(provider: str | None = None, model: str | None = None) -> LLMProvider:
    """
    provider: "bedrock" | "ollama" | "openai" | None (None -> settings.llm_provider default)
    model: optional model-id override within the chosen provider.
    """
    from app.config import settings

    name = (provider or settings.llm_provider).lower()

    if name == "bedrock":
        from app.providers.bedrock import BedrockProvider

        return BedrockProvider(model_id=model)
    if name == "ollama":
        from app.providers.ollama import OllamaProvider

        return OllamaProvider(model=model)
    if name == "openai":
        from app.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model=model)

    raise ValueError(f"Unknown LLM provider '{name}'. Valid options: {', '.join(VALID_PROVIDERS)}")


__all__ = ["create_provider", "LLMProvider", "ToolCall", "VALID_PROVIDERS"]
