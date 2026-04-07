"""Unit and end-to-end tests for Agent tool-calling loop."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import anyio
import httpx
import pytest
from pydantic import BaseModel, SecretStr

from ag2.agent import Agent
from ag2.config import OpenAICompletionsConfig
from ag2.events import (
    ChatMessage,
    ErrorEvent,
    MemoryStream,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
)
from ag2.providers import ModelResponse, OpenAICompletionsClient, ResponseFormat, ToolSpec
from ag2.tools import MaxToolRoundsError, tool


class _Summary(BaseModel):
    answer: int
    note: str


class _ScriptedModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.recorded_messages: list[list[ChatMessage]] = []
        self.recorded_tools: list[list[ToolSpec]] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        del response_format
        self.recorded_messages.append(list(messages))
        self.recorded_tools.append(list(tools))
        return self._responses.pop(0)

    async def aclose(self) -> None:
        return None


async def test_agent_executes_tools_concurrently_and_accumulates_usage() -> None:
    started_a = anyio.Event()
    started_b = anyio.Event()

    @tool
    async def fetch_a(key: str) -> str:
        started_a.set()
        await started_b.wait()  # Would deadlock if tools ran sequentially.
        return f"A:{key}"

    @tool
    async def fetch_b(key: str) -> str:
        started_b.set()
        await started_a.wait()
        return f"B:{key}"

    client = _ScriptedModelClient(
        [
            ModelResponse(
                message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(id="c1", name="fetch_a", arguments={"key": "x"}),
                        ToolCall(id="c2", name="fetch_b", arguments={"key": "y"}),
                    ],
                ),
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="fake",
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=ChatMessage(role="assistant", content='{"answer": 7, "note": "A:x+B:y"}'),
                usage=Usage(prompt_tokens=20, completion_tokens=6, total_tokens=26),
                model="fake",
                finish_reason="stop",
            ),
        ]
    )

    async with MemoryStream(history_limit=50) as stream:
        async with Agent(
            name="runner",
            model=client,
            tools=[fetch_a, fetch_b],
            stream=stream,
        ) as agent:
            with anyio.fail_after(2.0):
                reply = await agent.ask("Run both", output_type=_Summary)

        assert isinstance(reply.output, _Summary)
        assert reply.output.answer == 7
        assert reply.usage.total_tokens == 41
        assert [m.role for m in reply.history] == [
            "user",
            "assistant",
            "tool",
            "tool",
            "assistant",
        ]
        assert reply.history[2].content == "A:x"
        assert reply.history[3].content == "B:y"

        call_events = [ev for ev in stream.history if isinstance(ev, ToolCallEvent)]
        result_events = [ev for ev in stream.history if isinstance(ev, ToolResultEvent)]
        assert [e.tool_name for e in call_events] == ["fetch_a", "fetch_b"]
        assert [e.content for e in result_events] == ["A:x", "B:y"]
        assert all(not e.is_error for e in result_events)


async def test_agent_recovers_from_unknown_tool_validation_and_runtime_errors() -> None:
    @tool
    def divide(a: int, b: int) -> float:
        if b == 0:
            raise ZeroDivisionError("division by zero")
        return a / b

    client = _ScriptedModelClient(
        [
            ModelResponse(
                message=ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ToolCall(id="c1", name="missing_tool", arguments={}),
                        ToolCall(id="c2", name="divide", arguments={"a": "bad", "b": 2}),
                        ToolCall(id="c3", name="divide", arguments={"a": 10, "b": 0}),
                    ],
                ),
                model="fake",
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=ChatMessage(
                    role="assistant",
                    tool_calls=[ToolCall(id="c4", name="divide", arguments={"a": 10, "b": 2})],
                ),
                model="fake",
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=ChatMessage(role="assistant", content="Result is 5.0"),
                model="fake",
                finish_reason="stop",
            ),
        ]
    )

    async with MemoryStream(history_limit=50) as stream:
        async with Agent(name="safe", model=client, tools=[divide], stream=stream) as agent:
            reply = await agent.ask("Divide 10 by 2")

        assert reply.text == "Result is 5.0"
        result_events = [ev for ev in stream.history if isinstance(ev, ToolResultEvent)]
        assert [e.is_error for e in result_events] == [True, True, True, False]
        assert "unknown tool" in result_events[0].content
        assert "Invalid arguments" in result_events[1].content
        assert "division by zero" in result_events[2].content
        assert result_events[3].content == "5.0"


async def test_agent_raises_max_tool_rounds_error() -> None:
    @tool
    def ping() -> str:
        return "pong"

    loop_reply = ModelResponse(
        message=ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="ping", arguments={})],
        ),
        model="fake",
        finish_reason="tool_calls",
    )
    client = _ScriptedModelClient([loop_reply, loop_reply, loop_reply])

    async with MemoryStream(history_limit=20) as stream:
        async with Agent(
            name="looper",
            model=client,
            tools=[ping],
            max_tool_rounds=2,
            stream=stream,
        ) as agent:
            with pytest.raises(MaxToolRoundsError, match="max_tool_rounds=2"):
                await agent.ask("Keep pinging")

        err_events = [ev for ev in stream.history if isinstance(ev, ErrorEvent)]
        assert len(err_events) == 1
        assert err_events[0].error_type == "MaxToolRoundsError"


async def test_agent_propagates_cancellation_inside_tool() -> None:
    @tool
    async def hang() -> str:
        await anyio.sleep(10)
        return "never"

    client = _ScriptedModelClient(
        [
            ModelResponse(
                message=ChatMessage(
                    role="assistant",
                    tool_calls=[ToolCall(id="c1", name="hang", arguments={})],
                ),
                model="fake",
            )
        ]
    )

    async with Agent(name="cancel-test", model=client, tools=[hang]) as agent:
        with anyio.move_on_after(0.05) as scope:
            await agent.ask("hang")
        assert scope.cancelled_caught is True


@pytest.mark.e2e
async def test_e2e_openai_client_tool_loop_and_continuation() -> None:
    """End-to-end test: OpenAICompletionsClient + httpx.MockTransport + Agent + @tool."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @tool
    async def format_badge(label: str, value: int) -> str:
        """Format a numeric badge."""
        await anyio.lowlevel.checkpoint()
        return f"[{label}={value}]"

    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        payload = json.loads(request.content.decode("utf-8"))
        messages: list[dict[str, Any]] = payload["messages"]

        if turn == 1:
            assert any(t["function"]["name"] == "add" for t in payload["tools"])
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-1",
                    "object": "chat.completion",
                    "created": 1700000001,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_add",
                                        "type": "function",
                                        "function": {
                                            "name": "add",
                                            "arguments": '{"a": 19, "b": 23}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
                },
            )

        if turn == 2:
            assert messages[-1]["role"] == "tool"
            assert messages[-1]["content"] == "42"
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-2",
                    "object": "chat.completion",
                    "created": 1700000002,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "The sum is 42."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 25, "completion_tokens": 6, "total_tokens": 31},
                },
            )

        if turn == 3:
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-3",
                    "object": "chat.completion",
                    "created": 1700000003,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_badge",
                                        "type": "function",
                                        "function": {
                                            "name": "format_badge",
                                            "arguments": '{"label": "sum", "value": 42}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37},
                },
            )

        assert messages[-1]["role"] == "tool"
        assert messages[-1]["content"] == "[sum=42]"
        return httpx.Response(
            200,
            json={
                "id": "cmpl-4",
                "object": "chat.completion",
                "created": 1700000004,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Badge: [sum=42]"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45},
            },
        )

    cfg = OpenAICompletionsConfig(model="gpt-4o-mini", api_key=SecretStr("sk-e2e"))
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client,
        OpenAICompletionsClient(cfg, http_client=http_client) as model_client,
        MemoryStream(history_limit=50) as stream,
        Agent(
            name="math-agent",
            model=model_client,
            tools=[add, format_badge],
            stream=stream,
        ) as agent,
    ):
        reply1 = await agent.ask("Compute 19 + 23")
        assert reply1.text == "The sum is 42."
        assert reply1.usage.total_tokens == 54

        reply2 = await reply1.ask("Format that result as a badge.")
        assert reply2.text == "Badge: [sum=42]"
        assert reply2.run_id == reply1.run_id
        assert turn == 4
