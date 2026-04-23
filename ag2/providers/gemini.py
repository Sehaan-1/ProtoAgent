"""Async Google Gemini (google-genai) provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Self, cast

import anyio
import httpx
import structlog
from opentelemetry import trace
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from ag2.config.models import GeminiConfig, resolve_api_key
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
    from google import genai

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.providers.gemini")


def _import_genai() -> tuple[ModuleType, ModuleType, ModuleType]:
    try:
        from google import genai as genai_mod
        from google.genai import errors as errors_mod
        from google.genai import types as types_mod
    except ImportError as exc:
        raise ImportError(
            "GeminiClient requires the 'gemini' optional dependency. "
            "Install it with: pip install 'ag2[gemini]'"
        ) from exc
    return genai_mod, types_mod, errors_mod


def _to_gemini_contents(
    messages: Sequence[ChatMessage],
    types_mod: ModuleType,
) -> tuple[str | None, list[Any]]:
    """Map canonical ChatMessages into ``(system_instruction, contents)``."""
    system_parts: list[str] = []
    contents: list[Any] = []
    call_names_by_id: dict[str, str] = {}

    for msg in messages:
        if msg.role == "system":
            if msg.content:
                system_parts.append(msg.content)
            continue

        if msg.role == "user":
            contents.append(
                types_mod.Content(
                    role="user",
                    parts=[types_mod.Part(text=msg.content or "")],
                )
            )
            continue

        if msg.role == "assistant":
            parts: list[Any] = []
            if msg.content:
                parts.append(types_mod.Part(text=msg.content))
            for tc in msg.tool_calls:
                call_names_by_id[tc.id] = tc.name
                parts.append(
                    types_mod.Part(
                        function_call=types_mod.FunctionCall(
                            id=tc.id,
                            name=tc.name,
                            args=tc.arguments,
                        )
                    )
                )
            if not parts:
                parts.append(types_mod.Part(text=""))
            contents.append(types_mod.Content(role="model", parts=parts))
            continue

        # Tool output message -> Gemini user function_response part.
        fn_name = msg.name or call_names_by_id.get(msg.tool_call_id or "", "tool")
        raw_content = msg.content or ""
        try:
            decoded = json.loads(raw_content)
            response_obj: dict[str, Any] = (
                decoded if isinstance(decoded, dict) else {"result": decoded}
            )
        except json.JSONDecodeError:
            response_obj = {"result": raw_content}
        contents.append(
            types_mod.Content(
                role="user",
                parts=[
                    types_mod.Part(
                        function_response=types_mod.FunctionResponse(
                            id=msg.tool_call_id,
                            name=fn_name,
                            response=response_obj,
                        )
                    )
                ],
            )
        )

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _is_retryable_genai_error(exc: BaseException) -> bool:
    code = getattr(exc, "code", None)
    return isinstance(code, int) and (code == 429 or code >= 500)


class GeminiClient:
    """Async adapter wrapping ``google.genai.Client`` with ``tenacity`` retries."""

    def __init__(
        self,
        config: GeminiConfig,
        *,
        client: genai.Client | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        genai_mod, types_mod, errors_mod = _import_genai()
        self._genai = genai_mod
        self._types = types_mod
        self._errors = errors_mod
        self.config = config
        self._owns_client = client is None

        if client is not None:
            self._client = client
        else:
            api_key = resolve_api_key(config, "GEMINI_API_KEY") or resolve_api_key(
                config, "GOOGLE_API_KEY"
            )
            http_opts_kwargs: dict[str, Any] = {}
            if config.base_url is not None:
                http_opts_kwargs["base_url"] = config.base_url
            if config.api_version is not None:
                http_opts_kwargs["api_version"] = config.api_version
            if config.extra_headers:
                http_opts_kwargs["headers"] = dict(config.extra_headers)
            if config.timeout:
                http_opts_kwargs["timeout"] = int(max(1.0, config.timeout) * 1000)
            if http_client is not None:
                http_opts_kwargs["httpx_async_client"] = http_client

            http_options = types_mod.HttpOptions(**http_opts_kwargs) if http_opts_kwargs else None
            try:
                self._client = genai_mod.Client(
                    api_key=api_key,
                    http_options=http_options,
                )
            except Exception as exc:
                raise ProviderError(f"Failed to initialize Gemini client: {exc}") from exc

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        """Execute an async Gemini ``generate_content`` request with retries."""
        types_mod = self._types
        api_error_cls = self._errors.APIError
        retryable_base: tuple[type[Exception], ...] = (
            api_error_cls,
            httpx.HTTPError,
        )

        system_instruction, contents = _to_gemini_contents(messages, types_mod)
        cfg_kwargs: dict[str, Any] = {}
        if system_instruction is not None:
            cfg_kwargs["system_instruction"] = system_instruction
        if self.config.temperature is not None:
            cfg_kwargs["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = self.config.max_tokens
        if self.config.top_k is not None:
            cfg_kwargs["top_k"] = self.config.top_k
        if tools:
            declarations = [
                types_mod.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters_json_schema=t.parameters or {"type": "object", "properties": {}},
                )
                for t in tools
            ]
            cfg_kwargs["tools"] = [types_mod.Tool(function_declarations=declarations)]
        if response_format is not None:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_json_schema"] = response_format.json_schema

        gen_config = types_mod.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None

        def _should_retry(exc: BaseException) -> bool:
            if isinstance(exc, httpx.HTTPError):
                return True
            if isinstance(exc, api_error_cls):
                return _is_retryable_genai_error(exc)
            return False

        with tracer.start_as_current_span(
            "ag2.provider.gemini.complete",
            attributes={"gen_ai.request.model": self.config.model},
        ):
            retrying = AsyncRetrying(
                stop=stop_after_attempt(self.config.max_retries + 1),
                wait=wait_exponential_jitter(initial=0.05, max=2.0),
                retry=retry_if_exception(_should_retry),
                reraise=True,
            )
            try:
                async for attempt in retrying:
                    with attempt:
                        response = await self._client.aio.models.generate_content(
                            model=self.config.model,
                            contents=contents,
                            config=gen_config,
                        )
            except retryable_base as exc:
                if _should_retry(exc):
                    logger.warning(
                        "ag2.provider.gemini.retry_exhausted",
                        model=self.config.model,
                        error=str(exc),
                    )
                    raise RetryableProviderError(
                        f"Gemini request failed after retries: {exc}"
                    ) from exc
                raise ProviderError(f"Gemini request failed: {exc}") from exc

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise ProviderError("Gemini response contained no candidates")

        candidate = candidates[0]
        content_obj = getattr(candidate, "content", None)
        parts = getattr(content_obj, "parts", None) or []

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for idx, part in enumerate(parts):
            text = getattr(part, "text", None)
            if isinstance(text, str):
                text_parts.append(text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                fc_id = getattr(fc, "id", None) or f"gemini_call_{idx}"
                fc_name = str(getattr(fc, "name", ""))
                raw_args = getattr(fc, "args", None)
                tool_calls.append(
                    ToolCall(
                        id=str(fc_id),
                        name=fc_name,
                        arguments=dict(raw_args) if isinstance(raw_args, dict) else {},
                    )
                )

        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
            completion_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
            total_tokens = int(
                getattr(usage_meta, "total_token_count", 0) or (prompt_tokens + completion_tokens)
            )
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        else:
            usage = Usage()

        finish_reason_raw = getattr(candidate, "finish_reason", None)
        finish_reason = (
            str(getattr(finish_reason_raw, "value", finish_reason_raw))
            if finish_reason_raw is not None
            else None
        )

        return ModelResponse(
            message=ChatMessage(
                role="assistant",
                content="".join(text_parts) if text_parts else None,
                tool_calls=tool_calls,
            ),
            usage=usage,
            model=cast(str, getattr(response, "model_version", None) or self.config.model),
            finish_reason=finish_reason,
            raw=response,
        )

    async def stream_complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream Gemini text deltas and yield the final assembled ModelResponse."""
        types_mod = self._types
        api_error_cls = self._errors.APIError
        system_instruction, contents = _to_gemini_contents(messages, types_mod)
        cfg_kwargs: dict[str, Any] = {}
        if system_instruction is not None:
            cfg_kwargs["system_instruction"] = system_instruction
        if self.config.temperature is not None:
            cfg_kwargs["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            cfg_kwargs["max_output_tokens"] = self.config.max_tokens
        if self.config.top_k is not None:
            cfg_kwargs["top_k"] = self.config.top_k
        if tools:
            declarations = [
                types_mod.FunctionDeclaration(
                    name=t.name,
                    description=t.description,
                    parameters_json_schema=t.parameters or {"type": "object", "properties": {}},
                )
                for t in tools
            ]
            cfg_kwargs["tools"] = [types_mod.Tool(function_declarations=declarations)]
        if response_format is not None:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_json_schema"] = response_format.json_schema
        gen_config = types_mod.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None

        with tracer.start_as_current_span(
            "ag2.provider.gemini.stream_complete",
            attributes={"gen_ai.request.model": self.config.model},
        ):
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            usage = Usage()
            model_version = self.config.model
            finish_reason: str | None = None
            try:
                stream = await self._client.aio.models.generate_content_stream(
                    model=self.config.model,
                    contents=contents,
                    config=gen_config,
                )
                async for chunk in stream:
                    if getattr(chunk, "model_version", None):
                        model_version = str(chunk.model_version)
                    usage_meta = getattr(chunk, "usage_metadata", None)
                    if usage_meta is not None:
                        pt = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
                        ct = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
                        tt = int(getattr(usage_meta, "total_token_count", 0) or (pt + ct))
                        usage = Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)
                    candidates = getattr(chunk, "candidates", None) or []
                    if not candidates:
                        continue
                    candidate = candidates[0]
                    fr = getattr(candidate, "finish_reason", None)
                    if fr is not None:
                        finish_reason = str(getattr(fr, "value", fr))
                    parts = getattr(getattr(candidate, "content", None), "parts", None) or []
                    for part in parts:
                        text = getattr(part, "text", None)
                        if isinstance(text, str) and text:
                            text_parts.append(text)
                            yield StreamChunk(text_delta=text)
                        fc = getattr(part, "function_call", None)
                        if fc is not None:
                            fc_id = getattr(fc, "id", None) or f"gemini_call_{len(tool_calls)}"
                            raw_args = getattr(fc, "args", None)
                            tool_calls.append(
                                ToolCall(
                                    id=str(fc_id),
                                    name=str(getattr(fc, "name", "")),
                                    arguments=dict(raw_args) if isinstance(raw_args, dict) else {},
                                )
                            )
            except (api_error_cls, httpx.HTTPError) as exc:
                raise ProviderError(f"Gemini stream request failed: {exc}") from exc

        yield StreamChunk(
            response=ModelResponse(
                message=ChatMessage(
                    role="assistant",
                    content="".join(text_parts) if text_parts else None,
                    tool_calls=tool_calls,
                ),
                usage=usage,
                model=model_version,
                finish_reason=finish_reason,
            )
        )

    async def aclose(self) -> None:
        """Close the underlying Gemini async client resources if owned."""
        if self._owns_client:
            with anyio.CancelScope(shield=True):
                await self._client.aio.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
