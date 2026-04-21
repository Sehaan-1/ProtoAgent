"""Factory helpers constructing ModelClient adapters from ModelConfig."""

from __future__ import annotations

import httpx

from ag2.config.models import AnthropicConfig, GeminiConfig, ModelConfig, OpenAICompletionsConfig
from ag2.providers.anthropic import AnthropicClient
from ag2.providers.base import ModelClient
from ag2.providers.gemini import GeminiClient
from ag2.providers.openai import OpenAICompletionsClient


def create_model_client(
    config: ModelConfig,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> ModelClient:
    """Instantiate the concrete ModelClient for a ModelConfig."""
    if isinstance(config, OpenAICompletionsConfig):
        return OpenAICompletionsClient(config, http_client=http_client)
    if isinstance(config, AnthropicConfig):
        return AnthropicClient(config, http_client=http_client)
    if isinstance(config, GeminiConfig):
        return GeminiClient(config, http_client=http_client)
    raise NotImplementedError(f"Provider adapter for {config.provider!r} is not implemented")
