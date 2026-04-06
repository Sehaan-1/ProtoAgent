"""Unit tests for ag2.config schemas and dependency isolation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import SecretStr, TypeAdapter, ValidationError

import ag2.config
from ag2.config import (
    AnthropicConfig,
    AnyModelConfig,
    GeminiConfig,
    ModelConfig,
    OpenAICompletionsConfig,
    resolve_api_key,
)


def test_openai_completions_config_defaults_and_secret_masking() -> None:
    cfg = OpenAICompletionsConfig(
        model="gpt-4o-mini",
        api_key=SecretStr("sk-secret-12345"),
        reasoning_effort="medium",
        parallel_tool_calls=True,
    )
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.timeout == 60.0
    assert cfg.max_retries == 3
    assert "sk-secret-12345" not in repr(cfg)
    assert "sk-secret-12345" not in str(cfg)
    assert cfg.api_key is not None
    assert cfg.api_key.get_secret_value() == "sk-secret-12345"


def test_discriminated_union_parses_provider_configs() -> None:
    adapter: TypeAdapter[AnyModelConfig] = TypeAdapter(AnyModelConfig)

    openai_cfg = adapter.validate_python({"provider": "openai", "model": "gpt-4o"})
    anthropic_cfg = adapter.validate_python(
        {"provider": "anthropic", "model": "claude-3-7-sonnet-latest"}
    )
    gemini_cfg = adapter.validate_python({"provider": "gemini", "model": "gemini-2.5-flash"})

    assert isinstance(openai_cfg, OpenAICompletionsConfig)
    assert isinstance(anthropic_cfg, AnthropicConfig)
    assert anthropic_cfg.max_tokens == 4096
    assert isinstance(gemini_cfg, GeminiConfig)


def test_model_config_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        OpenAICompletionsConfig(model="   ")

    with pytest.raises(ValidationError):
        OpenAICompletionsConfig(model="gpt-4o", timeout=0.0)

    with pytest.raises(ValidationError):
        OpenAICompletionsConfig(model="gpt-4o", temperature=2.5)

    with pytest.raises(ValidationError):
        ModelConfig.model_validate({"provider": "openai", "model": "gpt-4o", "unknown_field": 1})


def test_resolve_api_key_prefers_explicit_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_AG2_API_KEY", "from-env")

    explicit = OpenAICompletionsConfig(model="gpt-4o", api_key=SecretStr("from-config"))
    assert resolve_api_key(explicit, "TEST_AG2_API_KEY") == "from-config"

    fallback = OpenAICompletionsConfig(model="gpt-4o")
    assert resolve_api_key(fallback, "TEST_AG2_API_KEY") == "from-env"
    assert resolve_api_key(fallback, "MISSING_ENV_VAR") is None


def test_config_does_not_import_downstream_layers() -> None:
    forbidden = ("ag2.events", "ag2.providers", "ag2.agent", "ag2.tools", "ag2.mcp", "ag2.network")
    config_dir = Path(ag2.config.__file__).parent
    for py_file in config_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden), (
                        f"{py_file.name} must not import {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith(forbidden), (
                    f"{py_file.name} must not import {node.module}"
                )
