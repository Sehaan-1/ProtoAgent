"""Async OpenAI Chat Completions provider adapter."""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Self, cast

import anyio
import httpx
import structlog
from opentelemetry import trace
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ag2.config.models import OpenAICompletionsConfig, resolve_api_key
from ag2.events.messages import ChatMessage, ToolCall, Usage
from ag2.providers.base import (
    ModelResponse,
    ProviderError,
    ResponseFormat,
    RetryableProviderError,
    ToolSpec,
)

if TYPE_CHECKING:
    import openai

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.providers.openai")


def _import_openai() -> ModuleType:
    try:
        import openai as openai_mod
    except ImportError as exc:
        raise ImportError(
            "OpenAICompletionsClient requires the 'openai' optional dependency. "
            "Install it with: pip install 'ag2[openai]'"
        ) from exc
    return openai_mod


def _to_openai_message(msg: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": msg.role}
    if msg.content is not None:
        payload["content"] = msg.content
    elif msg.role != "assistant":
        payload["content"] = ""

    if msg.name is not None:
        payload["name"] = msg.name

    if msg.role == "tool":
        payload["tool_call_id"] = msg.tool_call_id

    if msg.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in msg.tool_calls
        ]
    return payload


def _to_openai_tool(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _parse_tool_arguments(raw_args: str | None, tool_name: str) -> dict[str, Any]:
    if raw_args is None or not raw_args.strip():
        return {}
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"Model returned invalid JSON arguments for tool {tool_name!r}: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ProviderError(
            f"Expected JSON object arguments for tool {tool_name!r}, got {type(parsed).__name__}"
        )
    return parsed


class OpenAICompletionsClient:
    """Async adapter wrapping ``openai.AsyncOpenAI`` with ``tenacity`` retries."""

    def __init__(
        self,
        config: OpenAICompletionsConfig,
        *,
        client: openai.AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        openai_mod = _import_openai()
        self._openai = openai_mod
        self.config = config
        self._owns_client = client is None

        if client is not None:
            self._client = client
        else:
            api_key = resolve_api_key(config, "OPENAI_API_KEY")
            try:
                self._client = openai_mod.AsyncOpenAI(
                    api_key=api_key,
                    base_url=config.base_url,
                    organization=config.organization,
                    timeout=config.timeout,
                    max_retries=0,
                    default_headers=config.extra_headers or None,
                    http_client=http_client,
                )
            except openai_mod.OpenAIError as exc:
                raise ProviderError(f"Failed to initialize OpenAI client: {exc}") from exc

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        """Run an async chat completion request with configured retries."""
        openai_mod = self._openai
        retryable_errors: tuple[type[Exception], ...] = (
            openai_mod.APIConnectionError,
            openai_mod.APITimeoutError,
            openai_mod.RateLimitError,
            openai_mod.InternalServerError,
        )

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [_to_openai_message(m) for m in messages],
        }
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            kwargs["max_tokens"] = self.config.max_tokens
        if self.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        if tools:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools]
            if self.config.parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = self.config.parallel_tool_calls
        if response_format is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.name,
                    "schema": response_format.json_schema,
                    "strict": response_format.strict,
                },
            }
        if self.config.extra_body:
            kwargs["extra_body"] = self.config.extra_body

        with tracer.start_as_current_span(
            "ag2.provider.openai.complete",
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
                        completion = await self._client.chat.completions.create(**kwargs)
            except retryable_errors as exc:
                logger.warning(
                    "ag2.provider.openai.retry_exhausted",
                    model=self.config.model,
                    error=str(exc),
                )
                raise RetryableProviderError(f"OpenAI request failed after retries: {exc}") from exc
            except openai_mod.OpenAIError as exc:
                raise ProviderError(f"OpenAI request failed: {exc}") from exc

        if not completion.choices:
            raise ProviderError("OpenAI response contained no choices")

        choice = completion.choices[0]
        raw_msg = choice.message
        parsed_calls: list[ToolCall] = []
        if raw_msg.tool_calls:
            for tc in raw_msg.tool_calls:
                func = getattr(tc, "function", None)
                if func is None:
                    continue
                parsed_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=func.name,
                        arguments=_parse_tool_arguments(func.arguments, func.name),
                    )
                )

        usage_obj = completion.usage
        usage = (
            Usage(
                prompt_tokens=usage_obj.prompt_tokens or 0,
                completion_tokens=usage_obj.completion_tokens or 0,
                total_tokens=usage_obj.total_tokens or 0,
            )
            if usage_obj is not None
            else Usage()
        )

        return ModelResponse(
            message=ChatMessage(
                role="assistant",
                content=raw_msg.content,
                tool_calls=parsed_calls,
            ),
            usage=usage,
            model=cast(str, completion.model or self.config.model),
            finish_reason=choice.finish_reason,
            raw=completion,
        )

    async def aclose(self) -> None:
        """Close the underlying AsyncOpenAI client if owned by this adapter."""
        if self._owns_client:
            with anyio.CancelScope(shield=True):
                await self._client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
