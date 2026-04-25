"""Knowledge + middleware example — KnowledgePlugin + AssemblerMiddleware.

Run:
    OPENAI_API_KEY=sk-... uv run python examples/knowledge_agent.py
"""

from __future__ import annotations

import asyncio

from ag2 import (
    Agent,
    AssemblerMiddleware,
    KnowledgePlugin,
    MemoryKnowledgeStore,
    OpenAICompletionsConfig,
    WorkingMemoryPolicy,
)


async def main() -> None:
    """Answer a question using retrieved knowledge injected via middleware."""
    store = MemoryKnowledgeStore()
    kb = KnowledgePlugin(store, default_limit=2)
    await kb.remember(
        "AG2 is an async-first, protocol-driven Python multi-agent framework. "
        "It supports OpenAI, Anthropic, and Gemini providers out of the box.",
        title="Framework Overview",
        tags=("architecture", "providers"),
    )
    await kb.remember(
        "AG2 provides conversation, consulting, discussion, and workflow "
        "channels on top of a Hub registry and AsyncWAL audit log.",
        title="Networking Channels",
        tags=("network", "channels"),
    )

    policy = WorkingMemoryPolicy(max_messages=20, preserve_recent=6)
    assembler = AssemblerMiddleware(knowledge=kb, working_memory=policy)

    async with Agent(
        name="knowledgeable",
        model=OpenAICompletionsConfig(model="gpt-4o-mini"),
        tools=kb.tools(),
        middleware=[assembler],
    ) as agent:
        reply = await agent.ask("What providers and channels does AG2 support?")
        print(f"[{agent.name}] {reply.text}")


if __name__ == "__main__":
    asyncio.run(main())
