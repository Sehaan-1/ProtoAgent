"""Model and provider configuration schemas for AG2."""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class ModelConfig(BaseModel):
    """Base configuration shared by all model provider adapters."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    provider: str
    model: str = Field(min_length=1)
    api_key: SecretStr | None = None
    base_url: str | None = None
    timeout: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model", "base_url")
    @classmethod
    def _strip_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be blank")
        return stripped


class OpenAICompletionsConfig(ModelConfig):
    """Configuration for OpenAI and OpenAI-compatible chat completions APIs."""

    provider: Literal["openai"] = "openai"
    organization: str | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    parallel_tool_calls: bool | None = None


class AnthropicConfig(ModelConfig):
    """Configuration for Anthropic Messages API models."""

    provider: Literal["anthropic"] = "anthropic"
    max_tokens: int = Field(default=4096, gt=0)
    top_k: int | None = Field(default=None, ge=1)


class GeminiConfig(ModelConfig):
    """Configuration for Google Gemini API models."""

    provider: Literal["gemini"] = "gemini"
    api_version: str | None = None
    top_k: int | None = Field(default=None, ge=1)


AnyModelConfig = Annotated[
    OpenAICompletionsConfig | AnthropicConfig | GeminiConfig,
    Field(discriminator="provider"),
]


def resolve_api_key(config: ModelConfig, env_var: str | None = None) -> str | None:
    """Return the plaintext API key from ``config`` or an environment variable fallback."""
    if config.api_key is not None:
        return config.api_key.get_secret_value()
    if env_var is not None:
        value = os.getenv(env_var)
        if value:
            return value
    return None
