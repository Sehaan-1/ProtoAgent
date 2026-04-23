"""Streaming tests across OpenAI, Anthropic, Gemini providers, fallback client, and Agent."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from typing import Any

import anthropic
import httpx
from pydantic import SecretStr

from ag2.agent import Agent
from ag2.config import AnthropicConfig, GeminiConfig, OpenAICompletionsConfig
from ag2.events import ChatMessage, MemoryStream, TextEvent
from ag2.providers import (
    AnthropicClient,
    GeminiClient,
    ModelResponse,
    OpenAICompletionsClient,
    ResponseFormat,
    StreamingModelClient,
    ToolSpec,
    stream_model_response,
)
from ag2.tools import tool


def _httpx2() -> Any:
    return sys.modules[anthropic.DefaultAsyncHttpxClient.__bases__[0].__module__]


async def test_openai_stream_complete_emits_deltas_and_tool_calls() -> None:
    def sse_handler(_request: httpx.Request) -> httpx.Response:
        chunks = [
            {
                "id": "c1",
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {"content": "Hel"}}],
            },
            {
                "id": "c1",
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {"content": "lo"}}],
            },
            {
                "id": "c1",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "ping", "arguments": '{"a":'},
                                }
                            ]
                        },
                    }
                ],
            },
            {
                "id": "c1",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": " 7}"}}]},
                    }
                ],
            },
            {
                "id": "c1",
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
        body = "".join(f"data: {json.dumps(item)}\n\n" for item in chunks) + "data: [DONE]\n\n"
        return httpx.Response(
            200, content=body.encode("utf-8"), headers={"content-type": "text/event-stream"}
        )

    cfg = OpenAICompletionsConfig(model="gpt-4o-mini", api_key=SecretStr("sk-stream"))
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(sse_handler)) as hc,
        OpenAICompletionsClient(cfg, http_client=hc) as client,
    ):
        assert isinstance(client, StreamingModelClient)
        deltas: list[str] = []
        final_resp: ModelResponse | None = None
        async for chunk in client.stream_complete([ChatMessage(role="user", content="hi")]):
            if chunk.text_delta:
                deltas.append(chunk.text_delta)
            if chunk.response is not None:
                final_resp = chunk.response

    assert deltas == ["Hel", "lo"]
    assert final_resp is not None
    assert final_resp.message.content == "Hello"
    assert final_resp.usage.total_tokens == 15
    assert final_resp.finish_reason == "tool_calls"
    assert final_resp.message.tool_calls[0].arguments == {"a": 7}


async def test_anthropic_stream_complete_emits_text_deltas() -> None:
    httpx2_mod = _httpx2()

    def sse_handler(_request: Any) -> Any:
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_s1",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-3-7-sonnet-latest",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 9, "output_tokens": 1},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Bon"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "jour"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 4},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
        return httpx2_mod.Response(
            200,
            content=body.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    cfg = AnthropicConfig(model="claude-3-7-sonnet-latest", api_key=SecretStr("sk-ant"))
    async with (
        anthropic.DefaultAsyncHttpxClient(transport=httpx2_mod.MockTransport(sse_handler)) as hc,
        AnthropicClient(cfg, http_client=hc) as client,
    ):
        deltas: list[str] = []
        final_resp: ModelResponse | None = None
        async for chunk in client.stream_complete([ChatMessage(role="user", content="greet")]):
            if chunk.text_delta:
                deltas.append(chunk.text_delta)
            if chunk.response is not None:
                final_resp = chunk.response

    assert deltas == ["Bon", "jour"]
    assert final_resp is not None
    assert final_resp.message.content == "Bonjour"
    assert final_resp.usage.total_tokens == 13


async def test_gemini_stream_complete_emits_text_deltas() -> None:
    def sse_handler(_request: httpx.Request) -> httpx.Response:
        frames = [
            {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Fast "}]}}],
                "modelVersion": "gemini-2.5-flash",
            },
            {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "stream"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 7,
                },
                "modelVersion": "gemini-2.5-flash",
            },
        ]
        body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)
        return httpx.Response(
            200, content=body.encode("utf-8"), headers={"content-type": "text/event-stream"}
        )

    cfg = GeminiConfig(model="gemini-2.5-flash", api_key=SecretStr("gm-stream"))
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(sse_handler)) as hc,
        GeminiClient(cfg, http_client=hc) as client,
    ):
        deltas: list[str] = []
        final_resp: ModelResponse | None = None
        async for chunk in client.stream_complete([ChatMessage(role="user", content="stream")]):
            if chunk.text_delta:
                deltas.append(chunk.text_delta)
            if chunk.response is not None:
                final_resp = chunk.response

    assert deltas == ["Fast ", "stream"]
    assert final_resp is not None
    assert final_resp.message.content == "Fast stream"
    assert final_resp.usage.total_tokens == 7


async def test_agent_stream_ask_and_fallback_non_streaming_client() -> None:
    class _NonStreamingClient:
        async def complete(
            self,
            messages: Sequence[ChatMessage],
            *,
            tools: Sequence[ToolSpec] = (),
            response_format: ResponseFormat | None = None,
        ) -> ModelResponse:
            del messages, tools, response_format
            return ModelResponse(
                message=ChatMessage(role="assistant", content="one-shot"), model="stub"
            )

        async def aclose(self) -> None:
            return None

    fallback_chunks = [
        c
        async for c in stream_model_response(
            _NonStreamingClient(),
            [ChatMessage(role="user", content="hi")],
        )
    ]
    assert [c.text_delta for c in fallback_chunks[:-1]] == ["one-shot"]
    assert fallback_chunks[-1].response is not None

    @tool
    def mult(a: int, b: int) -> int:
        return a * b

    turn = 0

    def sse_agent_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        if turn == 1:
            frames = [
                {
                    "id": "c1",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_m",
                                        "type": "function",
                                        "function": {
                                            "name": "mult",
                                            "arguments": '{"a": 6, "b": 7}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            ]
        else:
            frames = [
                {
                    "id": "c2",
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {"content": "Answer: "}}],
                },
                {
                    "id": "c2",
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {"content": "42"}, "finish_reason": "stop"}],
                },
            ]
        body = "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames) + "data: [DONE]\n\n"
        return httpx.Response(
            200, content=body.encode("utf-8"), headers={"content-type": "text/event-stream"}
        )

    cfg = OpenAICompletionsConfig(model="gpt-4o-mini", api_key=SecretStr("sk-e2e"))
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(sse_agent_handler)) as hc,
        OpenAICompletionsClient(cfg, http_client=hc) as model_client,
        MemoryStream(history_limit=30) as stream,
        Agent(name="streamer", model=model_client, tools=[mult], stream=stream) as agent,
    ):
        collected_deltas: list[str] = []
        async with agent.stream_ask("Multiply 6 and 7") as turn_stream:
            async for ev in turn_stream:
                assert ev.delta is True
                collected_deltas.append(ev.text)
            reply = turn_stream.reply

    assert collected_deltas == ["Answer: ", "42"]
    assert reply.text == "Answer: 42"
    delta_events = [ev for ev in stream.history if isinstance(ev, TextEvent) and ev.delta]
    final_events = [ev for ev in stream.history if isinstance(ev, TextEvent) and not ev.delta]
    assert [e.text for e in delta_events] == ["Answer: ", "42"]
    assert [e.text for e in final_events] == ["Answer: 42"]
