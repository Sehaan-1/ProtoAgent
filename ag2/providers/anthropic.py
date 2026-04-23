"""Async Anthropic Messages API provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Self, cast

import anyio
import structlog
from opentelemetry import trace
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ag2.config.models import AnthropicConfig, resolve_api_key
from ag2.events.messages import ChatMessage, ToolCall, Usage
from ag2.providers.base import (
    ModelResponse,
    ProviderError,
    ResponseFormat,
    RetryableProviderError,
    StreamChunk,
    ToolSpec,
)

if TYPE_CHECKING:
    import anthropic

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.providers.anthropic")


def _import_anthropic() -> ModuleType:
    try:
        import anthropic as anthropic_mod
    except ImportError as exc:
        raise ImportError(
            "AnthropicClient requires the 'anthropic' optional dependency. "
            "Install it with: pip install 'ag2[anthropic]'"
        ) from exc
    return anthropic_mod


def _to_anthropic_messages(
    messages: Sequence[ChatMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert canonical ChatMessages into Anthropic ``(system, messages)`` format.

    System messages are concatenated into the top-level ``system`` prompt, and
    adjacent ``tool`` role messages are grouped into ``user`` messages with
    ``tool_result`` content blocks as required by the Messages API.
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def _flush_tool_results() -> None:
        if pending_tool_results:
            converted.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for msg in messages:
        if msg.role == "system":
            if msg.content:
                system_parts.append(msg.content)
            continue

        if msg.role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content or "",
                }
            )
            continue

        _flush_tool_results()
        if msg.role == "user":
            converted.append({"role": "user", "content": msg.content or ""})
            continue

        # Assistant message (may combine text content and tool_use blocks).
        blocks: list[dict[str, Any]] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        converted.append({"role": "assistant", "content": blocks})

    _flush_tool_results()
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, converted


def _to_anthropic_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters or {"type": "object", "properties": {}},
    }


def _parse_tool_input(raw_input: Any, tool_name: str) -> dict[str, Any]:
    if raw_input is None:
        return {}
    if isinstance(raw_input, dict):
        return raw_input
    if isinstance(raw_input, str):
        try:
            parsed = json.loads(raw_input) if raw_input.strip() else {}
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Anthropic returned invalid JSON tool input for {tool_name!r}: {exc}"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise ProviderError(
        f"Expected dict tool input for Anthropic tool {tool_name!r}, got {type(raw_input).__name__}"
    )


class AnthropicClient:
    """Async adapter wrapping ``anthropic.AsyncAnthropic`` with ``tenacity`` retries."""

    def __init__(
        self,
        config: AnthropicConfig,
        *,
        client: anthropic.AsyncAnthropic | None = None,
        http_client: Any = None,
    ) -> None:
        anthropic_mod = _import_anthropic()
        self._anthropic = anthropic_mod
        self.config = config
        self._owns_client = client is None

        if client is not None:
            self._client = client
        else:
            api_key = resolve_api_key(config, "ANTHROPIC_API_KEY")
            try:
                self._client = anthropic_mod.AsyncAnthropic(
                    api_key=api_key,
                    base_url=config.base_url,
                    timeout=config.timeout,
                    max_retries=0,
                    default_headers=config.extra_headers or None,
                    http_client=http_client,
                )
            except anthropic_mod.AnthropicError as exc:
                raise ProviderError(f"Failed to initialize Anthropic client: {exc}") from exc

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        """Execute an async Anthropic Messages request with configured retries."""
        anthropic_mod = self._anthropic
        retryable_errors: tuple[type[Exception], ...] = (
            anthropic_mod.APIConnectionError,
            anthropic_mod.APITimeoutError,
            anthropic_mod.RateLimitError,
            anthropic_mod.InternalServerError,
            anthropic_mod.OverloadedError,
            anthropic_mod.ServiceUnavailableError,
        )

        system_prompt, anthropic_messages = _to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt
        extra_body: dict[str, Any] = dict(self.config.extra_body)
        if self.config.temperature is not None:
            extra_body["temperature"] = self.config.temperature
        if self.config.top_k is not None:
            extra_body["top_k"] = self.config.top_k
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]
        if response_format is not None:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": response_format.json_schema,
                }
            }

        with tracer.start_as_current_span(
            "ag2.provider.anthropic.complete",
            attributes={"gen_ai.request.model": self.config.model},
        ):
            retrying = AsyncRetrying(
                stop=stop_after_attempt(self.config.max_retries + 1),
                wait=wait_exponential_jitter(initial=0.05, max=2.0),
                retry=retry_if_exception_type(retryable_errors),
                reraise=True,
            )
            try:
                async for attempt in retrying:
                    with attempt:
                        response = await self._client.messages.create(**kwargs)
            except retryable_errors as exc:
                logger.warning(
                    "ag2.provider.anthropic.retry_exhausted",
                    model=self.config.model,
                    error=str(exc),
                )
                raise RetryableProviderError(
                    f"Anthropic request failed after retries: {exc}"
                ) from exc
            except anthropic_mod.AnthropicError as exc:
                raise ProviderError(f"Anthropic request failed: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in getattr(response, "content", ()):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(str(getattr(block, "text", "")))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=_parse_tool_input(
                            getattr(block, "input", {}),
                            str(getattr(block, "name", "")),
                        ),
                    )
                )

        usage_obj = getattr(response, "usage", None)
        if usage_obj is not None:
            prompt_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
            completion_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
        else:
            usage = Usage()

        return ModelResponse(
            message=ChatMessage(
                role="assistant",
                content="".join(text_parts) if text_parts else None,
                tool_calls=tool_calls,
            ),
            usage=usage,
            model=cast(str, getattr(response, "model", None) or self.config.model),
            finish_reason=getattr(response, "stop_reason", None),
            raw=response,
        )

    async def stream_complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream Anthropic text deltas and yield the final assembled ModelResponse."""
        anthropic_mod = self._anthropic
        system_prompt, anthropic_messages = _to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt
        extra_body: dict[str, Any] = dict(self.config.extra_body)
        if self.config.temperature is not None:
            extra_body["temperature"] = self.config.temperature
        if self.config.top_k is not None:
            extra_body["top_k"] = self.config.top_k
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]
        if response_format is not None:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": response_format.json_schema,
                }
            }

        with tracer.start_as_current_span(
            "ag2.provider.anthropic.stream_complete",
            attributes={"gen_ai.request.model": self.config.model},
        ):
            try:
                async with self._client.messages.stream(**kwargs) as msg_stream:
                    async for event in msg_stream:
                        if getattr(event, "type", None) == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if getattr(delta, "type", None) == "text_delta":
                                delta_text = getattr(delta, "text", None)
                                if delta_text:
                                    yield StreamChunk(text_delta=str(delta_text))
                    final_msg = await msg_stream.get_final_message()
            except anthropic_mod.AnthropicError as exc:
                raise ProviderError(f"Anthropic stream request failed: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in getattr(final_msg, "content", ()):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(str(getattr(block, "text", "")))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=_parse_tool_input(
                            getattr(block, "input", {}),
                            str(getattr(block, "name", "")),
                        ),
                    )
                )
        usage_obj = getattr(final_msg, "usage", None)
        if usage_obj is not None:
            pt = int(getattr(usage_obj, "input_tokens", 0) or 0)
            ct = int(getattr(usage_obj, "output_tokens", 0) or 0)
            usage = Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
        else:
            usage = Usage()
        yield StreamChunk(
            response=ModelResponse(
                message=ChatMessage(
                    role="assistant",
                    content="".join(text_parts) if text_parts else None,
                    tool_calls=tool_calls,
                ),
                usage=usage,
                model=cast(str, getattr(final_msg, "model", None) or self.config.model),
                finish_reason=getattr(final_msg, "stop_reason", None),
                raw=final_msg,
            )
        )

    async def aclose(self) -> None:
        """Close the underlying AsyncAnthropic client if owned by this adapter."""
        if self._owns_client:
            with anyio.CancelScope(shield=True):
                await self._client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
