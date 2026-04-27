"""Live real-LLM showcase running against OpenRouter, OpenAI, Anthropic, Gemini, or Ollama.

Auto-detects available provider credentials from the environment (in order:
OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY/GOOGLE_API_KEY)
or lets you choose explicitly with CLI flags:

    # Auto-detect (uses OPENROUTER_API_KEY if set, etc.)
    uv run python examples/live_demo.py

    # Explicit provider or local Ollama
    uv run python examples/live_demo.py --provider openrouter --model openai/gpt-4o-mini
    uv run python examples/live_demo.py --provider ollama --model llama3.2
    uv run python examples/live_demo.py --scenario tools
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from ag2 import (
    END,
    Agent,
    AnthropicConfig,
    AssemblerMiddleware,
    GeminiConfig,
    Hub,
    KnowledgePlugin,
    MemoryKnowledgeStore,
    MemoryStream,
    ModelConfig,
    OpenAICompletionsConfig,
    SpanCapture,
    ToolCallEvent,
    ToolResultEvent,
    TransitionGraph,
    WorkingMemoryPolicy,
    tool,
    workflow,
)

ProviderChoice = Literal["auto", "openrouter", "openai", "anthropic", "gemini", "ollama"]
ScenarioChoice = Literal["all", "tools", "structured", "knowledge", "workflow"]


class FrameworkBrief(BaseModel):
    """Structured report extracted by a live LLM response."""

    name: str
    top_capabilities: list[str] = Field(min_length=2, max_length=5)
    supports_streaming: bool


def resolve_live_config(
    *,
    provider: ProviderChoice = "auto",
    model: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 350,
) -> tuple[str, ModelConfig]:
    """Build a ``ModelConfig`` from CLI args or available environment variables."""
    chosen = provider
    if chosen == "auto":
        if os.getenv("OPENROUTER_API_KEY"):
            chosen = "openrouter"
        elif os.getenv("OPENAI_API_KEY"):
            chosen = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            chosen = "anthropic"
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            chosen = "gemini"
        else:
            raise SystemExit(
                "No live LLM API key detected. Set OPENROUTER_API_KEY, OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, or GEMINI_API_KEY — or pass --provider ollama --model <name>."
            )

    if chosen == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set.")
        return chosen, OpenAICompletionsConfig(
            model=model or "openai/gpt-4o-mini",
            base_url=base_url or "https://openrouter.ai/api/v1",
            api_key=SecretStr(key),
            max_tokens=max_tokens,
        )
    if chosen == "openai":
        return chosen, OpenAICompletionsConfig(
            model=model or "gpt-4o-mini",
            base_url=base_url,
            max_tokens=max_tokens,
        )
    if chosen == "ollama":
        return chosen, OpenAICompletionsConfig(
            model=model or "llama3.2",
            base_url=base_url or "http://localhost:11434/v1",
            api_key=SecretStr("ollama"),
            max_tokens=max_tokens,
        )
    if chosen == "anthropic":
        return chosen, AnthropicConfig(
            model=model or "claude-3-5-haiku-latest",
            base_url=base_url,
            max_tokens=max_tokens,
        )
    return chosen, GeminiConfig(
        model=model or "gemini-2.5-flash",
        base_url=base_url,
        max_tokens=max_tokens,
    )


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers accurately."""
    return a * b


@tool
def list_ag2_packages() -> list[str]:
    """List the top-level subpackages present inside the local ag2/ directory."""
    ag2_root = Path(__file__).resolve().parents[1] / "ag2"
    return sorted(p.name for p in ag2_root.iterdir() if p.is_dir() and not p.name.startswith("_"))


async def run_tools_and_streaming(config: ModelConfig) -> int:
    """Demo 1: Live token streaming + real local @tool execution."""
    print("\n[1/4] Streaming + Real Tools")
    print("-" * 40)
    async with (
        MemoryStream(history_limit=50) as stream,
        Agent(
            name="inspector",
            model=config,
            system_message="Always use tools when asked for workspace data or math.",
            tools=[list_ag2_packages, multiply_numbers],
            stream=stream,
        ) as agent,
    ):
        prompt = (
            "Call list_ag2_packages and multiply_numbers(a=len(packages), b=10). "
            "Report how many packages exist and the product in one concise sentence."
        )
        print("Streaming reply: ", end="", flush=True)
        async with agent.stream_ask(prompt) as turn_stream:
            async for chunk in turn_stream:
                sys.stdout.write(chunk.text)
                sys.stdout.flush()
            reply = turn_stream.reply
        print()

        for event in stream.history:
            if isinstance(event, ToolCallEvent):
                print(f"  -> tool_call: {event.tool_name}({event.arguments})")
            elif isinstance(event, ToolResultEvent):
                print(f"  <- tool_result: {event.tool_name} => {event.content}")
        print(f"  usage: {reply.usage.total_tokens} tokens (model={reply.model})")
        return reply.usage.total_tokens


async def run_structured_output(config: ModelConfig) -> int:
    """Demo 2: Real LLM structured JSON output validated into a Pydantic model."""
    print("\n[2/4] Structured Output (Pydantic)")
    print("-" * 40)
    async with Agent(name="extractor", model=config) as agent:
        reply = await agent.ask(
            "Summarize AG2 (ProtoAgent): an async-first Python multi-agent framework "
            "supporting OpenAI, Anthropic, Gemini, MCP tools, Hub workflows, and streaming.",
            output_type=FrameworkBrief,
        )
        brief = reply.output
        print(f"  parsed: {brief!r}")
        print(f"  usage: {reply.usage.total_tokens} tokens")
        return reply.usage.total_tokens


async def run_knowledge_middleware(config: ModelConfig) -> int:
    """Demo 3: KnowledgeStore retrieval + AssemblerMiddleware injection."""
    print("\n[3/4] Knowledge + AssemblerMiddleware")
    print("-" * 40)
    store = MemoryKnowledgeStore()
    kb = KnowledgePlugin(store, default_limit=2)
    await kb.remember(
        "AG2 uses AsyncWAL backed by aiosqlite inside Hub to persist conversation "
        "and workflow handoffs.",
        title="Hub WAL Persistence",
        tags=("network", "wal"),
    )
    assembler = AssemblerMiddleware(
        knowledge=kb,
        working_memory=WorkingMemoryPolicy(max_messages=12, preserve_recent=4),
    )
    async with Agent(name="kb-assistant", model=config, middleware=[assembler]) as agent:
        reply = await agent.ask("In one sentence, how does Hub persist audit logs in AG2?")
        print(f"  reply: {reply.text}")
        print(f"  usage: {reply.usage.total_tokens} tokens")
        return reply.usage.total_tokens


async def run_multi_agent_workflow(config: ModelConfig) -> int:
    """Demo 4: Multi-agent WorkflowChannel + AsyncWAL + SpanCapture telemetry."""
    print("\n[4/4] Multi-Agent Workflow + AsyncWAL + OpenTelemetry")
    print("-" * 40)
    graph = (
        TransitionGraph()
        .set_entry("architect")
        .add_edge("architect", "reviewer")
        .add_edge("reviewer", END)
    )
    async with Hub(close_agents=True) as hub:
        hub.register(
            Agent(
                name="architect",
                model=config,
                system_message="Propose a 1-sentence design principle for async agent systems.",
                stream=hub.stream,
            )
        )
        hub.register(
            Agent(
                name="reviewer",
                model=config,
                system_message="Tighten the proposed sentence into a crisp 10-word motto.",
                stream=hub.stream,
            )
        )
        with SpanCapture() as cap:
            result = await workflow(
                hub,
                graph,
                "Give us a guiding principle for async multi-agent orchestration.",
                run_id="live-demo-wf",
            )
        wal_entries = await hub.read_wal(run_id="live-demo-wf")
        print(f"  turns: {[t.speaker for t in result.turns]}")
        print(f"  final_text: {result.final_text}")
        print(f"  wal_entries: {len(wal_entries)} ({[e.kind for e in wal_entries]})")
        print(f"  otel_spans: {cap.span_names}")
        print(f"  usage: {result.usage.total_tokens} tokens")
        return result.usage.total_tokens


async def main(argv: list[str] | None = None) -> None:
    """Run live real-LLM demonstrations against a real provider endpoint."""
    parser = argparse.ArgumentParser(description="Run live AG2 demos against a real LLM.")
    parser.add_argument(
        "--provider",
        choices=["auto", "openrouter", "openai", "anthropic", "gemini", "ollama"],
        default="auto",
    )
    parser.add_argument("--model", default=None, help="Override model ID.")
    parser.add_argument("--base-url", default=None, help="Override API base URL.")
    parser.add_argument("--max-tokens", type=int, default=350, help="Max completion tokens.")
    parser.add_argument(
        "--scenario",
        choices=["all", "tools", "structured", "knowledge", "workflow"],
        default="all",
    )
    args = parser.parse_args(argv)

    provider_name, config = resolve_live_config(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
    )
    print(f"Running live AG2 demo using provider={provider_name!r}, model={config.model!r}")

    total_tokens = 0
    if args.scenario in ("all", "tools"):
        total_tokens += await run_tools_and_streaming(config)
    if args.scenario in ("all", "structured"):
        total_tokens += await run_structured_output(config)
    if args.scenario in ("all", "knowledge"):
        total_tokens += await run_knowledge_middleware(config)
    if args.scenario in ("all", "workflow"):
        total_tokens += await run_multi_agent_workflow(config)

    print(f"\nLive demo finished cleanly (total_tokens={total_tokens}).")


if __name__ == "__main__":
    asyncio.run(main())
