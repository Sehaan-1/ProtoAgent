# AG2 (ProtoAgent)

Clean, async-first, protocol-driven multi-agent framework built around the modern `import ag2` (v1.x) architecture.

## Highlights

- **Async-first Agent loop** (`Agent.ask`, `Agent.stream_ask`, `AgentReply.ask`) built on `anyio` structured concurrency.
- **Pluggable Model Providers**: `OpenAICompletionsClient`, `AnthropicClient`, and `GeminiClient` behind `ModelClient` / `StreamingModelClient` protocols, with retries (`tenacity`), structured output parsing (`ResponseFormat`), and incremental streaming (`StreamChunk`).
- **Typed Tools & Concurrent Execution**: `@tool` decorator with automatic JSON schema inference from Python type hints and parallel tool-call execution.
- **Human-in-the-Loop (HITL)**: Injected `Context` with `HITLHook` adapters (`CallbackHITLHook`, `QueueHITLHook`, `CLIHITLHook`).
- **Model Context Protocol (MCP)**: `MCPSession`, `MCPToolAdapter`, and `load_mcp_tools()` over stdio and HTTP transports.
- **Multi-Agent Networking**: `Hub` registry, SQLite-backed `AsyncWAL` audit log, typed channels (`conversation`, `consulting`, `discussion`), and declarative `WorkflowChannel` (`TransitionGraph` + `END` sentinel).
- **Agent Harness**: `MemoryKnowledgeStore` / `DiskKnowledgeStore`, `KnowledgePlugin`, `CompactTrigger` (`TailWindowCompact`, `SummarizeCompact`), and `AssemblerMiddleware` / `WorkingMemoryPolicy`.
- **Observability**: `MemoryStream` pub/sub event bus (`MessageEvent`, `TextEvent`, `ToolCallEvent`, `ToolResultEvent`, `ErrorEvent`) plus OpenTelemetry & `structlog` helpers (`configure_telemetry`, `SpanCapture`).

---

## Architecture

```
ag2/
├── config/       # Pydantic v2 configs: OpenAICompletionsConfig, AnthropicConfig, GeminiConfig
├── events/       # Immutable ChatMessage, Usage, ToolCall, typed events, MemoryStream
├── providers/    # ModelClient, StreamingModelClient, OpenAI / Anthropic / Gemini adapters
├── tools/        # @tool decorator, schema extraction, Tool execution & serialization
├── hitl/         # Context, HITLHook, CallbackHITLHook, QueueHITLHook, CLIHITLHook
├── mcp/          # MCPSession, MCPStdioConfig, MCPHttpConfig, MCPToolAdapter, load_mcp_tools
├── knowledge/    # KnowledgeStore (Memory & Disk), KnowledgePlugin retrieval tools
├── compact/      # CompactTrigger, TailWindowCompact, SummarizeCompact, token estimation
├── middleware/   # MessageMiddleware protocol, WorkingMemoryPolicy, AssemblerMiddleware
├── agent/        # Agent, AgentReply, AgentStream
├── network/      # Hub, AsyncWAL, Conversation/Consulting/Discussion/Workflow channels
└── telemetry/    # configure_telemetry, get_tracer, get_meter, SpanCapture
```

Dependency boundaries are enforced in tests so lower layers (`events`, `config`, `knowledge`, `compact`, `middleware`, `telemetry`) stay decoupled from `agent`, `mcp`, and `network`.

---

## Installation

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# Clone and install dev dependencies
uv sync --locked --dev
```

Install optional provider or integration extras when packaging or installing selectively:

```bash
pip install "ag2[openai]"
pip install "ag2[anthropic,gemini]"
pip install "ag2[mcp,storage,telemetry]"
# Or install everything:
pip install "ag2[all]"
```

Set provider API keys as environment variables (or pass `SecretStr` via `api_key`):
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Gemini: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)

---

## Quickstart

### Basic Agent & Multi-Turn Continuation

```python
import asyncio
from ag2 import Agent, OpenAICompletionsConfig


async def main() -> None:
    config = OpenAICompletionsConfig(model="gpt-4o-mini")
    async with Agent(
        name="assistant",
        model=config,
        system_message="You are a concise assistant.",
    ) as agent:
        reply = await agent.ask("What is the capital of France?")
        print(reply.text)

        follow_up = await reply.ask("What river runs through it?")
        print(follow_up.text)


asyncio.run(main())
```

### Streaming Responses

```python
async with agent.stream_ask("Write a short haiku.") as stream:
    async for event in stream:
        print(event.text, end="", flush=True)
    reply = stream.reply
```

### Typed Tools & Structured Output

```python
from pydantic import BaseModel
from ag2 import Agent, OpenAICompletionsConfig, tool


class WeatherReport(BaseModel):
    city: str
    summary: str


@tool
async def get_temperature(city: str) -> str:
    """Return the current temperature for a city."""
    return f"22C in {city}"


async with Agent(
    name="weather",
    model=OpenAICompletionsConfig(model="gpt-4o-mini"),
    tools=[get_temperature],
) as agent:
    reply = await agent.ask(
        "What is the weather in Paris?",
        output_type=WeatherReport,
    )
    print(reply.output)  # WeatherReport instance
```

### Multi-Agent Workflow (`Hub` + `TransitionGraph`)

```python
from ag2 import END, Agent, Hub, OpenAICompletionsConfig, TransitionGraph, workflow

model = OpenAICompletionsConfig(model="gpt-4o-mini")
graph = (
    TransitionGraph().set_entry("drafter").add_edge("drafter", "reviewer").add_edge("reviewer", END)
)

async with Hub(close_agents=True) as hub:
    hub.register(Agent(name="drafter", model=model, system_message="Draft text."))
    hub.register(Agent(name="reviewer", model=model, system_message="Polish text."))
    result = await workflow(hub, graph, "Explain async/await concisely.")
    print(result.final_text)
```

### Knowledge + Working-Memory Middleware

```python
from ag2 import (
    Agent,
    AssemblerMiddleware,
    KnowledgePlugin,
    MemoryKnowledgeStore,
    OpenAICompletionsConfig,
    WorkingMemoryPolicy,
)

store = MemoryKnowledgeStore()
kb = KnowledgePlugin(store)
await kb.remember("AG2 emits typed events on MemoryStream.", title="Events")

assembler = AssemblerMiddleware(
    knowledge=kb,
    working_memory=WorkingMemoryPolicy(max_messages=20, preserve_recent=6),
)
async with Agent(
    name="kb-agent",
    model=OpenAICompletionsConfig(model="gpt-4o-mini"),
    tools=kb.tools(),
    middleware=[assembler],
) as agent:
    reply = await agent.ask("How does AG2 emit events?")
```

### Telemetry & Observability

```python
from ag2 import SpanCapture, configure_telemetry

configure_telemetry("my-ag2-service", log_level="INFO")

with SpanCapture() as cap:
    reply = await agent.ask("Hello!")

print(cap.span_names)  # ['ag2.agent.ask', ...]
```

---

## Runnable Examples

See [`examples/`](examples/):

- [`examples/live_demo.py`](examples/live_demo.py) — **Real-LLM end-to-end showcase** (streaming + `@tool` calls, Pydantic `output_type`, `KnowledgePlugin` + `AssemblerMiddleware`, and `Hub` + `WorkflowChannel` + `AsyncWAL` + `SpanCapture`). Auto-detects `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or runs locally via `--provider ollama`:
  ```bash
  uv run python examples/live_demo.py
  # or with local Ollama:
  uv run python examples/live_demo.py --provider ollama --model llama3.2
  ```
- [`examples/quickstart.py`](examples/quickstart.py) — `Agent.ask()` and `reply.ask()`
- [`examples/multi_provider.py`](examples/multi_provider.py) — Concurrent `AnthropicConfig` + `GeminiConfig`
- [`examples/streaming_agent.py`](examples/streaming_agent.py) — `Agent.stream_ask()` token streaming
- [`examples/multi_agent_workflow.py`](examples/multi_agent_workflow.py) — `Hub` + `TransitionGraph` + `workflow()`
- [`examples/knowledge_agent.py`](examples/knowledge_agent.py) — `MemoryKnowledgeStore` + `KnowledgePlugin` + `AssemblerMiddleware`
- [`examples/telemetry_demo.py`](examples/telemetry_demo.py) — Self-contained telemetry & `SpanCapture` demo (no API key needed)

---

## Development Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy ag2 test examples
uv run pytest -v

# Opt-in live API integration tests (uses configured provider env key)
uv run pytest -m live -v -s
```
