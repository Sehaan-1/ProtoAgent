"""Core provider protocols, response models, streaming contracts, and structured output helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ag2.events.messages import ChatMessage, Usage


class ProviderError(RuntimeError):
    """Base error raised by model provider adapters."""


class RetryableProviderError(ProviderError):
    """Transient provider error that remained failing after retries were exhausted."""


class StructuredOutputError(ValueError):
    """Raised when model output cannot be parsed or validated against output_type."""


class ToolSpec(BaseModel):
    """Provider-agnostic tool schema descriptor passed to ModelClient.complete()."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


def enforce_strict_json_schema(node: Any) -> Any:
    """Recursively set additionalProperties=False on JSON Schema object nodes."""
    if isinstance(node, list):
        return [enforce_strict_json_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    updated: dict[str, Any] = {
        key: enforce_strict_json_schema(value) for key, value in node.items()
    }
    if updated.get("type") == "object" and "additionalProperties" not in updated:
        updated["additionalProperties"] = False
    return updated


class ResponseFormat(BaseModel):
    """Structured JSON Schema output specification for a model call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    json_schema: dict[str, Any]
    strict: bool = True

    @classmethod
    def from_type(cls, output_type: type[Any], *, strict: bool = True) -> ResponseFormat:
        """Build a ResponseFormat from any Pydantic-compatible Python type."""
        schema = TypeAdapter(output_type).json_schema()
        if strict:
            schema = enforce_strict_json_schema(schema)
        name = getattr(output_type, "__name__", "StructuredOutput")
        return cls(name=name, json_schema=schema, strict=strict)


class ModelResponse(BaseModel):
    """Normalized response returned by a ModelClient."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    message: ChatMessage
    usage: Usage = Field(default_factory=Usage)
    model: str
    finish_reason: str | None = None
    raw: Any = Field(default=None, exclude=True)


class StreamChunk(BaseModel):
    """Single incremental token delta or terminal response emitted during streaming."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text_delta: str | None = None
    response: ModelResponse | None = None


@runtime_checkable
class ModelClient(Protocol):
    """Async protocol implemented by all model provider adapters."""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        """Execute a single async chat completion turn."""
        ...

    async def aclose(self) -> None:
        """Release underlying HTTP/SDK resources."""
        ...


@runtime_checkable
class StreamingModelClient(ModelClient, Protocol):
    """Optional extension protocol for ModelClients supporting token streaming."""

    def stream_complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield incremental StreamChunk deltas ending with a final ModelResponse chunk."""
        ...


async def stream_model_response(
    client: ModelClient,
    messages: Sequence[ChatMessage],
    *,
    tools: Sequence[ToolSpec] = (),
    response_format: ResponseFormat | None = None,
) -> AsyncIterator[StreamChunk]:
    """Stream chunks from ``client``, falling back gracefully to ``complete()``."""
    if isinstance(client, StreamingModelClient):
        async for chunk in client.stream_complete(
            messages,
            tools=tools,
            response_format=response_format,
        ):
            yield chunk
        return

    response = await client.complete(
        messages,
        tools=tools,
        response_format=response_format,
    )
    if response.message.content:
        yield StreamChunk(text_delta=response.message.content)
    yield StreamChunk(response=response)
