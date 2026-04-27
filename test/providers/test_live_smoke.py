"""Opt-in live integration smoke tests against real LLM APIs.

Skipped during normal offline CI runs (``addopts = "-m 'not live'"``).
Run explicitly whenever API keys or local Ollama are available:

    uv run pytest -m live -v -s
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, SecretStr

from ag2 import (
    Agent,
    AnthropicConfig,
    GeminiConfig,
    ModelConfig,
    OpenAICompletionsConfig,
    tool,
)

pytestmark = pytest.mark.live


class _CityFacts(BaseModel):
    city: str
    country: str


def _first_available_live_config() -> tuple[str, ModelConfig] | None:
    if key := os.getenv("OPENROUTER_API_KEY"):
        return "openrouter", OpenAICompletionsConfig(
            model=os.getenv("AG2_LIVE_MODEL", "openai/gpt-4o-mini"),
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(key),
            max_tokens=200,
        )
    if os.getenv("OPENAI_API_KEY"):
        return "openai", OpenAICompletionsConfig(
            model=os.getenv("AG2_LIVE_MODEL", "gpt-4o-mini"),
            max_tokens=200,
        )
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic", AnthropicConfig(
            model=os.getenv("AG2_LIVE_MODEL", "claude-3-5-haiku-latest"),
            max_tokens=200,
        )
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini", GeminiConfig(
            model=os.getenv("AG2_LIVE_MODEL", "gemini-2.5-flash"),
            max_tokens=200,
        )
    return None


@tool
def multiply_ints(x: int, y: int) -> int:
    """Multiply two integers."""
    return x * y


async def test_live_agent_streaming_tools_and_structured_output() -> None:
    resolved = _first_available_live_config()
    if resolved is None:
        pytest.skip(
            "No live LLM key set (OPENROUTER_API_KEY, OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, or GEMINI_API_KEY)"
        )

    _provider, config = resolved

    async with Agent(name="live-smoke", model=config, tools=[multiply_ints]) as agent:
        chunks: list[str] = []
        async with agent.stream_ask(
            "Use multiply_ints with x=13 and y=7, then state only the numeric product."
        ) as stream:
            async for ev in stream:
                chunks.append(ev.text)
            tool_reply = stream.reply

        assert "91" in tool_reply.text
        assert "".join(chunks) == tool_reply.text
        assert tool_reply.usage.total_tokens > 0

        structured_reply = await agent.ask(
            "Return the city Tokyo and its country Japan.",
            output_type=_CityFacts,
        )
        assert isinstance(structured_reply.output, _CityFacts)
        assert structured_reply.output.city.lower() == "tokyo"
        assert structured_reply.output.country.lower() == "japan"
