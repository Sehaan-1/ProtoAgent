"""Unit and Agent E2E tests for ag2.providers.anthropic.AnthropicClient."""

from __future__ import annotations

import builtins
import json
import sys
from typing import Any

import anthropic
import pytest
from pydantic import BaseModel, SecretStr

from ag2.agent import Agent
from ag2.config import AnthropicConfig
from ag2.events import ChatMessage
from ag2.providers import (
    AnthropicClient,
    ProviderError,
    ResponseFormat,
    RetryableProviderError,
    ToolSpec,
)
from ag2.tools import tool


class _Verdict(BaseModel):
    approved: bool
    reason: str


def _httpx2() -> Any:
    return sys.modules[anthropic.DefaultAsyncHttpxClient.__bases__[0].__module__]


def _anthropic_message_payload(
    *,
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    return {
        "id": "msg_test_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-7-sonnet-latest",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 18,
            "output_tokens": 9,
        },
    }


async def test_anthropic_client_serializes_system_tools_and_parses_tool_use() -> None:
    captured_bodies: list[dict[str, Any]] = []
    httpx2_mod = _httpx2()

    def sync_handler(request: Any) -> Any:
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx2_mod.Response(
            200,
            json=_anthropic_message_payload(
                content=[
                    {"type": "text", "text": "Checking weather..."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
                stop_reason="tool_use",
            ),
        )

    transport = httpx2_mod.MockTransport(sync_handler)
    async with anthropic.DefaultAsyncHttpxClient(transport=transport) as http_client:
        cfg = AnthropicConfig(
            model="claude-3-7-sonnet-latest",
            api_key=SecretStr("sk-ant-test"),
            temperature=0.1,
            top_k=20,
        )
        async with AnthropicClient(cfg, http_client=http_client) as client:
            res = await client.complete(
                [
                    ChatMessage(role="system", content="Be brief."),
                    ChatMessage(role="user", content="Weather in Paris?"),
                ],
                tools=[
                    ToolSpec(
                        name="get_weather",
                        description="Lookup weather",
                        parameters={
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    )
                ],
                response_format=ResponseFormat.from_type(_Verdict),
            )

    body = captured_bodies[0]
    assert body["system"] == "Be brief."
    assert body["temperature"] == pytest.approx(0.1)
    assert body["top_k"] == 20
    assert body["tools"][0]["name"] == "get_weather"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert res.usage.total_tokens == 27
    assert res.message.content == "Checking weather..."
    assert len(res.message.tool_calls) == 1
    assert res.message.tool_calls[0].name == "get_weather"
    assert res.message.tool_calls[0].arguments == {"city": "Paris"}


async def test_anthropic_client_retries_rate_limits_and_propagates_bad_request() -> None:
    httpx2_mod = _httpx2()
    attempts = 0

    def retry_handler(_request: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            return httpx2_mod.Response(
                429,
                json={
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "slow down"},
                },
            )
        return httpx2_mod.Response(
            200,
            json=_anthropic_message_payload(content=[{"type": "text", "text": "ok-after-retry"}]),
        )

    cfg = AnthropicConfig(
        model="claude-3-7-sonnet-latest",
        api_key=SecretStr("sk-ant"),
        max_retries=2,
    )
    async with (
        anthropic.DefaultAsyncHttpxClient(transport=httpx2_mod.MockTransport(retry_handler)) as hc,
        AnthropicClient(cfg, http_client=hc) as client,
    ):
        res = await client.complete([ChatMessage(role="user", content="hi")])
    assert attempts == 2
    assert res.message.content == "ok-after-retry"

    def bad_handler(_request: Any) -> Any:
        return httpx2_mod.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}},
        )

    async with (
        anthropic.DefaultAsyncHttpxClient(
            transport=httpx2_mod.MockTransport(bad_handler)
        ) as hc_bad,
        AnthropicClient(cfg, http_client=hc_bad) as client,
    ):
        with pytest.raises(ProviderError) as exc_info:
            await client.complete([ChatMessage(role="user", content="hi")])
        assert not isinstance(exc_info.value, RetryableProviderError)


async def test_anthropic_agent_tool_loop_groups_tool_results() -> None:
    httpx2_mod = _httpx2()

    @tool
    def double(n: int) -> int:
        return n * 2

    turn = 0

    def handler(request: Any) -> Any:
        nonlocal turn
        turn += 1
        payload = json.loads(request.content.decode("utf-8"))
        if turn == 1:
            return httpx2_mod.Response(
                200,
                json=_anthropic_message_payload(
                    content=[
                        {
                            "type": "tool_use",
                            "id": "tu_1",
                            "name": "double",
                            "input": {"n": 21},
                        }
                    ],
                    stop_reason="tool_use",
                ),
            )
        last_user = payload["messages"][-1]
        assert last_user["role"] == "user"
        assert last_user["content"][0]["type"] == "tool_result"
        assert last_user["content"][0]["tool_use_id"] == "tu_1"
        assert last_user["content"][0]["content"] == "42"
        return httpx2_mod.Response(
            200,
            json=_anthropic_message_payload(content=[{"type": "text", "text": "Result is 42."}]),
        )

    cfg = AnthropicConfig(model="claude-3-7-sonnet-latest", api_key=SecretStr("sk-ant"))
    async with (
        anthropic.DefaultAsyncHttpxClient(transport=httpx2_mod.MockTransport(handler)) as hc,
        AnthropicClient(cfg, http_client=hc) as client,
        Agent(name="claude-worker", model=client, tools=[double]) as agent,
    ):
        reply = await agent.ask("Double 21")
        assert reply.text == "Result is 42."
        assert turn == 2


def test_missing_anthropic_extra_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = AnthropicConfig(model="claude-3-7-sonnet-latest", api_key=SecretStr("sk-ant"))
    with pytest.raises(ImportError, match=r"ag2\[anthropic\]"):
        AnthropicClient(cfg)
