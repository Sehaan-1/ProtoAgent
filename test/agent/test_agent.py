"""Unit tests for ag2.agent.Agent and AgentReply."""

from __future__ import annotations

from collections.abc import Sequence

import anyio
import pytest
from pydantic import BaseModel

from ag2.agent import Agent
from ag2.events import (
    ChatMessage,
    ErrorEvent,
    MemoryStream,
    MessageEvent,
    TextEvent,
    Usage,
)
from ag2.providers import (
    ModelResponse,
    ProviderError,
    ResponseFormat,
    StructuredOutputError,
    ToolSpec,
)


class _Plan(BaseModel):
    title: str
    steps: list[str]


class _FakeModelClient:
    def __init__(self, responses: list[ModelResponse | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[ChatMessage], ResponseFormat | None]] = []
        self.closed = False

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        del tools
        self.calls.append((list(messages), response_format))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True


async def test_agent_ask_and_reply_continuation_preserves_isolated_threads() -> None:
    client = _FakeModelClient(
        [
            ModelResponse(
                message=ChatMessage(role="assistant", content="Hi Alice!"),
                usage=Usage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
                model="fake-gpt",
                finish_reason="stop",
            ),
            ModelResponse(
                message=ChatMessage(role="assistant", content="Your name is Alice."),
                usage=Usage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
                model="fake-gpt",
                finish_reason="stop",
            ),
            ModelResponse(
                message=ChatMessage(role="assistant", content="Hi Bob!"),
                usage=Usage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
                model="fake-gpt",
                finish_reason="stop",
            ),
        ]
    )

    async with Agent(
        name="assistant",
        model=client,
        system_message="Be concise.",
    ) as agent:
        reply1 = await agent.ask("I am Alice.")
        assert reply1.text == "Hi Alice!"
        assert reply1.output == "Hi Alice!"
        assert reply1.usage.total_tokens == 13
        assert len(reply1.history) == 3

        reply2 = await reply1.ask("What is my name?")
        assert reply2.text == "Your name is Alice."
        assert reply2.run_id == reply1.run_id
        assert [m.role for m in reply2.history] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]

        # Independent ask() does not leak Alice's thread.
        other_reply = await agent.ask("I am Bob.")
        assert other_reply.text == "Hi Bob!"
        assert other_reply.run_id != reply1.run_id
        assert len(other_reply.history) == 3

    # Caller-provided client is not closed by Agent.
    assert client.closed is False
    assert agent.stream.closed is True


async def test_agent_structured_output_parsing_and_validation_errors() -> None:
    client = _FakeModelClient(
        [
            ModelResponse(
                message=ChatMessage(
                    role="assistant",
                    content='{"title": "Ship AG2", "steps": ["scaffold", "test"]}',
                ),
                model="fake-gpt",
            ),
            ModelResponse(
                message=ChatMessage(role="assistant", content="not-valid-json"),
                model="fake-gpt",
            ),
        ]
    )

    async with MemoryStream(history_limit=20) as stream:
        agent = Agent(name="planner", model=client, stream=stream)
        reply = await agent.ask("Make a plan", output_type=_Plan)
        assert isinstance(reply.output, _Plan)
        assert reply.output.title == "Ship AG2"
        assert reply.output.steps == ["scaffold", "test"]
        assert client.calls[0][1] is not None
        assert client.calls[0][1].name == "_Plan"

        with pytest.raises(StructuredOutputError):
            await agent.ask("Broken plan", output_type=_Plan)

        await agent.aclose()
        # Caller-provided stream stays open until caller closes it.
        assert stream.closed is False

        error_events = [ev for ev in stream.history if isinstance(ev, ErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].error_type == "StructuredOutputError"


async def test_agent_emits_events_and_propagates_provider_errors() -> None:
    client = _FakeModelClient(
        [
            ModelResponse(
                message=ChatMessage(role="assistant", content="streamed answer"),
                model="fake-gpt",
            ),
            ProviderError("upstream down"),
        ]
    )

    async with MemoryStream(history_limit=20) as stream:
        agent = Agent(name="worker", model=client, stream=stream)
        async with stream.where(TextEvent, source="worker").subscribe() as text_sub:
            reply = await agent.ask("Hello", run_id="run-1")
            assert reply.text == "streamed answer"
            text_ev = await anext(text_sub)
            assert text_ev.text == "streamed answer"
            assert text_ev.run_id == "run-1"

        msg_events = [ev for ev in stream.history if isinstance(ev, MessageEvent)]
        assert len(msg_events) == 2

        with pytest.raises(ProviderError, match="upstream down"):
            await agent.ask("Fail", run_id="run-2")

        err_events = [ev for ev in stream.history if isinstance(ev, ErrorEvent)]
        assert len(err_events) == 1
        assert err_events[0].run_id == "run-2"


async def test_agent_cancellation_during_model_call() -> None:
    started = anyio.Event()

    class _SlowClient:
        async def complete(
            self,
            messages: Sequence[ChatMessage],
            *,
            tools: Sequence[ToolSpec] = (),
            response_format: ResponseFormat | None = None,
        ) -> ModelResponse:
            del messages, tools, response_format
            started.set()
            await anyio.sleep(10)
            return ModelResponse(message=ChatMessage(role="assistant", content="late"), model="x")

        async def aclose(self) -> None:
            return None

    async with Agent(name="slow", model=_SlowClient()) as agent:
        with anyio.move_on_after(0.05) as scope:
            await agent.ask("wait")
        assert scope.cancelled_caught is True
