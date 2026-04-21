"""Unit and Agent E2E tests for ag2.providers.gemini.GeminiClient."""

from __future__ import annotations

import builtins
import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, SecretStr

from ag2.agent import Agent
from ag2.config import GeminiConfig
from ag2.events import ChatMessage
from ag2.providers import (
    GeminiClient,
    ProviderError,
    ResponseFormat,
    RetryableProviderError,
    ToolSpec,
)
from ag2.tools import tool


class _Answer(BaseModel):
    city: str
    population_millions: float


def _gemini_payload(
    *,
    parts: list[dict[str, Any]],
    finish_reason: str = "STOP",
) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": parts},
                "finishReason": finish_reason,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 14,
            "candidatesTokenCount": 7,
            "totalTokenCount": 21,
        },
        "modelVersion": "gemini-2.5-flash",
    }


async def test_gemini_client_serializes_config_tools_and_structured_output() -> None:
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json=_gemini_payload(
                parts=[
                    {
                        "functionCall": {
                            "id": "call_geo_1",
                            "name": "lookup_city",
                            "args": {"name": "Tokyo"},
                        }
                    }
                ]
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        cfg = GeminiConfig(
            model="gemini-2.5-flash",
            api_key=SecretStr("gm-test"),
            temperature=0.2,
            top_k=32,
        )
        async with GeminiClient(cfg, http_client=http_client) as client:
            res = await client.complete(
                [
                    ChatMessage(role="system", content="Stay concise."),
                    ChatMessage(role="user", content="Lookup Tokyo."),
                ],
                tools=[
                    ToolSpec(
                        name="lookup_city",
                        description="Lookup city metadata",
                        parameters={
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                    )
                ],
                response_format=ResponseFormat.from_type(_Answer),
            )

    body = captured_bodies[0]
    assert body["systemInstruction"]["parts"][0]["text"] == "Stay concise."
    assert body["generationConfig"]["temperature"] == pytest.approx(0.2)
    assert body["generationConfig"]["topK"] == 32
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "lookup_city"

    assert res.usage.total_tokens == 21
    assert len(res.message.tool_calls) == 1
    assert res.message.tool_calls[0].name == "lookup_city"
    assert res.message.tool_calls[0].arguments == {"name": "Tokyo"}


async def test_gemini_client_retries_503_and_handles_400_without_retry() -> None:
    attempts = 0

    def retry_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            return httpx.Response(
                503,
                json={"error": {"code": 503, "message": "unavailable", "status": "UNAVAILABLE"}},
            )
        return httpx.Response(200, json=_gemini_payload(parts=[{"text": "recovered"}]))

    cfg = GeminiConfig(model="gemini-2.5-flash", api_key=SecretStr("gm-test"), max_retries=2)
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(retry_handler)) as hc,
        GeminiClient(cfg, http_client=hc) as client,
    ):
        res = await client.complete([ChatMessage(role="user", content="hi")])
    assert attempts == 2
    assert res.message.content == "recovered"

    def bad_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"code": 400, "message": "invalid arg", "status": "INVALID_ARGUMENT"}},
        )

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(bad_handler)) as hc_bad,
        GeminiClient(cfg, http_client=hc_bad) as client,
    ):
        with pytest.raises(ProviderError) as exc_info:
            await client.complete([ChatMessage(role="user", content="hi")])
        assert not isinstance(exc_info.value, RetryableProviderError)


async def test_gemini_agent_tool_call_and_function_response_e2e() -> None:
    @tool
    def square(n: int) -> int:
        return n * n

    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        body = json.loads(request.content.decode("utf-8"))
        if turn == 1:
            return httpx.Response(
                200,
                json=_gemini_payload(
                    parts=[{"functionCall": {"id": "c1", "name": "square", "args": {"n": 9}}}]
                ),
            )
        last_part = body["contents"][-1]["parts"][0]
        assert last_part["functionResponse"]["name"] == "square"
        assert last_part["functionResponse"]["response"] == {"result": 81}
        return httpx.Response(200, json=_gemini_payload(parts=[{"text": "9 squared is 81."}]))

    cfg = GeminiConfig(model="gemini-2.5-flash", api_key=SecretStr("gm-test"))
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as hc,
        GeminiClient(cfg, http_client=hc) as client,
        Agent(name="gemini-worker", model=client, tools=[square]) as agent,
    ):
        reply = await agent.ask("Compute 9 squared")
        assert reply.text == "9 squared is 81."
        assert turn == 2


def test_missing_gemini_extra_raises_helpful_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "google" or name.startswith("google.genai"):
            raise ImportError("No module named 'google.genai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = GeminiConfig(model="gemini-2.5-flash", api_key=SecretStr("gm-test"))
    with pytest.raises(ImportError, match=r"ag2\[gemini\]"):
        GeminiClient(cfg)
