"""Quickstart example — basic Agent.ask() and continuation with OpenAI.

Run:
    OPENAI_API_KEY=sk-... uv run python examples/quickstart.py
"""

from __future__ import annotations

import asyncio

from ag2 import Agent, OpenAICompletionsConfig


async def main() -> None:
    """Run a two-turn conversation using Agent.ask() and reply.ask()."""
    config = OpenAICompletionsConfig(model="gpt-4o-mini")
    async with Agent(
        name="assistant",
        model=config,
        system_message="You are a concise assistant.",
    ) as agent:
        reply = await agent.ask("What is the capital of France?")
        print(f"[{agent.name}] {reply.text}")

        follow_up = await reply.ask("What river runs through it?")
        print(f"[{agent.name}] {follow_up.text}")


if __name__ == "__main__":
    asyncio.run(main())
