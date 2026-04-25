"""Multi-provider example — AnthropicClient and GeminiClient side-by-side.

Run:
    ANTHROPIC_API_KEY=... GEMINI_API_KEY=... uv run python examples/multi_provider.py
"""

from __future__ import annotations

import asyncio

from ag2 import Agent, AnthropicConfig, GeminiConfig


async def main() -> None:
    """Ask the same question concurrently to both Anthropic and Gemini."""
    prompt = "In one sentence, what makes async I/O useful for AI agents?"

    async with (
        Agent(
            name="anthropic",
            model=AnthropicConfig(model="claude-sonnet-4-20250514"),
        ) as anthropic_agent,
        Agent(
            name="gemini",
            model=GeminiConfig(model="gemini-2.5-flash"),
        ) as gemini_agent,
    ):
        anthropic_reply, gemini_reply = await asyncio.gather(
            anthropic_agent.ask(prompt),
            gemini_agent.ask(prompt),
        )

    print(f"[anthropic] {anthropic_reply.text}")
    print(f"[gemini]    {gemini_reply.text}")


if __name__ == "__main__":
    asyncio.run(main())
