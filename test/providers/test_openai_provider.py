"""Unit tests for ag2.providers using real openai.AsyncOpenAI + httpx.MockTransport."""

from __future__ import annotations

import builtins
import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from ag2.config import AnthropicConfig, GeminiConfig, ModelConfig, OpenAICompletionsConfig
from ag2.events import ChatMessage
from ag2.providers import (
    AnthropicClient,
    GeminiClient,
    OpenAICompletionsClient,
    ProviderError,
    ResponseFormat,
    RetryableProviderError,
    ToolSpec,
    create_model_client,
)


class _Address(BaseModel):
    city: str
    country: str


class _Person(BaseModel):
    name: str
    address: _Address


def _completion_payload(
    *,
    content: str | None = "hello",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 5,
            "total_tokens": 17,
        },
    }


def test_response_format_enforces_strict_nested_object_schemas() -> None:
    fmt = ResponseFormat.from_type(_Person)
    assert fmt.name == "_Person"
    assert fmt.strict is True
    assert fmt.json_schema["additionalProperties"] is False
    defs = fmt.json_schema.get("$defs", {})
    assert defs["_Address"]["additionalProperties"] is False


async def test_openai_client_serializes_request_and_parses_tool_calls() -> None:
    captured_bodies: list[dict[str, Any]] = []
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        captured_headers.update(dict(request.headers))
        return httpx.Response(
            200,
            json=_completion_payload(
                content=None,
                tool_calls=[
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query": "ag2"}',
                        },
                    }
                ],
                finish_reason="tool_calls",
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        cfg = OpenAICompletionsConfig(
            model="gpt-4o-mini",
            api_key=SecretStr("sk-test"),
            temperature=0.2,
            reasoning_effort="low",
            parallel_tool_calls=False,
            extra_headers={"X-Custom": "1"},
        )
        async with OpenAICompletionsClient(cfg, http_client=http_client) as client:
            res = await client.complete(
                [ChatMessage(role="user", content="Find ag2")],
                tools=[
                    ToolSpec(
                        name="lookup",
                        description="Search docs",
                        parameters={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    )
                ],
                response_format=ResponseFormat.from_type(_Person),
            )

    assert len(captured_bodies) == 1
    body = captured_bodies[0]
    assert body["model"] == "gpt-4o-mini"
    assert body["temperature"] == 0.2
    assert body["reasoning_effort"] == "low"
    assert body["parallel_tool_calls"] is False
    assert body["tools"][0]["function"]["name"] == "lookup"
    assert body["response_format"]["type"] == "json_schema"
    assert captured_headers.get("x-custom") == "1"

    assert res.finish_reason == "tool_calls"
    assert res.usage.total_tokens == 17
    assert len(res.message.tool_calls) == 1
    assert res.message.tool_calls[0].name == "lookup"
    assert res.message.tool_calls[0].arguments == {"query": "ag2"}


async def test_openai_client_retries_rate_limits_then_succeeds() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json=_completion_payload(content="recovered"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        cfg = OpenAICompletionsConfig(
            model="gpt-4o-mini",
            api_key=SecretStr("sk-test"),
            max_retries=2,
        )
        async with OpenAICompletionsClient(cfg, http_client=http_client) as client:
            res = await client.complete([ChatMessage(role="user", content="hi")])

    assert attempts == 3
    assert res.message.content == "recovered"


async def test_openai_client_raises_retryable_error_when_retries_exhausted() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, json={"error": {"message": "server down"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        cfg = OpenAICompletionsConfig(
            model="gpt-4o-mini",
            api_key=SecretStr("sk-test"),
            max_retries=1,
        )
        async with OpenAICompletionsClient(cfg, http_client=http_client) as client:
            with pytest.raises(RetryableProviderError):
                await client.complete([ChatMessage(role="user", content="hi")])

    assert attempts == 2


async def test_openai_client_does_not_retry_bad_request() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        cfg = OpenAICompletionsConfig(
            model="gpt-4o-mini",
            api_key=SecretStr("sk-test"),
            max_retries=3,
        )
        async with OpenAICompletionsClient(cfg, http_client=http_client) as client:
            with pytest.raises(ProviderError) as exc_info:
                await client.complete([ChatMessage(role="user", content="hi")])

    assert not isinstance(exc_info.value, RetryableProviderError)
    assert attempts == 1


async def test_openai_client_handles_blank_and_null_tool_call_arguments() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion_payload(
                content=None,
                tool_calls=[
                    {
                        "id": "call_blank",
                        "type": "function",
                        "function": {"name": "ping", "arguments": ""},
                    },
                    {
                        "id": "call_null",
                        "type": "function",
                        "function": {"name": "status", "arguments": None},
                    },
                ],
                finish_reason="tool_calls",
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        cfg = OpenAICompletionsConfig(model="gpt-4o-mini", api_key=SecretStr("sk-test"))
        async with OpenAICompletionsClient(cfg, http_client=http_client) as client:
            res = await client.complete([ChatMessage(role="user", content="hi")])

    assert [tc.arguments for tc in res.message.tool_calls] == [{}, {}]


async def test_openai_client_rejects_invalid_tool_call_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion_payload(
                content=None,
                tool_calls=[
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{not-json"},
                    }
                ],
                finish_reason="tool_calls",
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        cfg = OpenAICompletionsConfig(model="gpt-4o-mini", api_key=SecretStr("sk-test"))
        async with OpenAICompletionsClient(cfg, http_client=http_client) as client:
            with pytest.raises(ProviderError, match="invalid JSON"):
                await client.complete([ChatMessage(role="user", content="hi")])


def test_factory_and_missing_openai_extra_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OpenAICompletionsConfig(model="gpt-4o-mini", api_key=SecretStr("sk-test"))
    client = create_model_client(cfg)
    assert isinstance(client, OpenAICompletionsClient)

    anthropic_cfg = AnthropicConfig(model="claude-3-7-sonnet-latest", api_key=SecretStr("sk-ant"))
    assert isinstance(create_model_client(anthropic_cfg), AnthropicClient)

    gemini_cfg = GeminiConfig(model="gemini-2.5-flash", api_key=SecretStr("gm-test"))
    assert isinstance(create_model_client(gemini_cfg), GeminiClient)

    custom_cfg = ModelConfig(provider="custom-unimplemented", model="m1")
    with pytest.raises(NotImplementedError):
        create_model_client(custom_cfg)

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"ag2\[openai\]"):
        OpenAICompletionsClient(cfg)
