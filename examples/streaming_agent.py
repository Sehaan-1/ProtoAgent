"""Streaming example — Agent.stream_ask() token-by-token output.

Run:
    OPENAI_API_KEY=sk-... uv run python examples/streaming_agent.py
"""

from __future__ import annotations

import asyncio
import sys

from ag2 import Agent, OpenAICompletionsConfig


async def main() -> None:
    """Stream a short poem token by token and inspect final usage."""
    config = OpenAICompletionsConfig(model="gpt-4o-mini")
    prompt = "Write a two-sentence haiku about asynchronous Python."
    print(f"Prompt: {prompt}\n---")

    async with Agent(name="writer", model=config) as agent, agent.stream_ask(prompt) as stream:
        async for event in stream:
            sys.stdout.write(event.text)
            sys.stdout.flush()
        reply = stream.reply

    print("\n---")
    print(f"Total tokens: {reply.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
