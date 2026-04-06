"""Unit tests for ag2.events models and async MemoryStream pub/sub."""

from __future__ import annotations

import ast
from pathlib import Path

import anyio
import pytest
from pydantic import TypeAdapter, ValidationError

import ag2.events
from ag2.events import (
    EV_COMPACTION_COMPLETED,
    EV_HUMAN_INPUT_REQUEST,
    EV_HUMAN_MESSAGE,
    EV_TEXT,
    AnyEvent,
    BaseEvent,
    ChatMessage,
    CompactionCompleted,
    CompactionFailed,
    CompactionStarted,
    ErrorEvent,
    HumanInputRequest,
    HumanMessage,
    MemoryStream,
    MessageEvent,
    StreamClosedError,
    TextEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
)


def test_chat_message_validation() -> None:
    tool_call = ToolCall(id="call_1", name="search", arguments={"q": "ag2"})
    assistant_msg = ChatMessage(role="assistant", tool_calls=[tool_call])
    assert assistant_msg.content is None
    assert len(assistant_msg.tool_calls) == 1

    tool_msg = ChatMessage(role="tool", tool_call_id="call_1", content="ok")
    assert tool_msg.tool_call_id == "call_1"

    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="missing call id")

    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="bad", tool_calls=[tool_call])

    with pytest.raises(ValidationError):
        ChatMessage(role="user", content=None)


def test_event_models_and_discriminated_union_roundtrip() -> None:
    adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)
    events: list[AnyEvent] = [
        TextEvent(text="hello", source="planner"),
        MessageEvent(message=ChatMessage(role="user", content="hi")),
        ToolCallEvent(call_id="c1", tool_name="calc", arguments={"x": 1}),
        ToolResultEvent(call_id="c1", tool_name="calc", content="2"),
        HumanInputRequest(prompt="Approve?", choices=["yes", "no"], timeout=30.0),
        HumanMessage(request_id="req1", content="yes"),
        CompactionStarted(strategy="summarize", original_message_count=20),
        CompactionCompleted(
            strategy="summarize",
            original_message_count=20,
            compacted_message_count=4,
            summary="condensed",
        ),
        CompactionFailed(strategy="summarize", error="context limit"),
        ErrorEvent(error_type="RateLimit", message="slow down", recoverable=True),
    ]

    for ev in events:
        payload = ev.model_dump()
        restored = adapter.validate_python(payload)
        assert restored == ev

    assert events[0].event_type == EV_TEXT
    assert events[4].event_type == EV_HUMAN_INPUT_REQUEST
    assert events[5].event_type == EV_HUMAN_MESSAGE
    assert events[7].event_type == EV_COMPACTION_COMPLETED


async def test_memory_stream_pub_sub_and_where_filters() -> None:
    async with MemoryStream() as stream:
        async with (
            stream.where(TextEvent).subscribe() as text_sub,
            stream.where(EV_HUMAN_INPUT_REQUEST, source="reviewer").subscribe() as req_sub,
            stream.where(lambda ev: ev.run_id == "run-42")
            .where(source="coder")
            .subscribe() as chained_sub,
        ):
            await stream.emit(TextEvent(text="first", source="coder", run_id="run-42"))
            await stream.publish(
                HumanInputRequest(prompt="Ship it?", source="reviewer", run_id="run-99")
            )
            await stream.publish(HumanMessage(content="yes", source="human", run_id="run-42"))

            first_text = await anext(text_sub)
            assert isinstance(first_text, TextEvent)
            assert first_text.text == "first"

            req = await anext(req_sub)
            assert isinstance(req, HumanInputRequest)
            assert req.prompt == "Ship it?"

            chained_ev = await anext(chained_sub)
            assert chained_ev.source == "coder"
            assert chained_ev.run_id == "run-42"

        assert stream.subscriber_count == 0


async def test_memory_stream_concurrent_subscribers_and_close() -> None:
    stream = MemoryStream()
    received_a: list[str] = []
    received_b: list[str] = []

    async def consumer(target: list[str]) -> None:
        async with stream.where(TextEvent).subscribe() as sub:
            async for ev in sub:
                target.append(ev.text)

    async with anyio.create_task_group() as tg:
        tg.start_soon(consumer, received_a)
        tg.start_soon(consumer, received_b)
        await anyio.lowlevel.checkpoint()

        await stream.emit(TextEvent(text="one"))
        await stream.emit(TextEvent(text="two"))
        await stream.aclose()

    assert received_a == ["one", "two"]
    assert received_b == ["one", "two"]
    assert stream.closed is True

    with pytest.raises(StreamClosedError):
        await stream.emit(TextEvent(text="after-close"))

    with pytest.raises(StreamClosedError):
        stream.subscribe()


async def test_memory_stream_history_replay() -> None:
    async with MemoryStream(history_limit=3) as stream:
        await stream.emit(TextEvent(text="m1", source="a"))
        await stream.emit(TextEvent(text="m2", source="b"))
        await stream.emit(TextEvent(text="m3", source="a"))
        await stream.emit(TextEvent(text="m4", source="a"))

        assert [ev.text for ev in stream.history if isinstance(ev, TextEvent)] == ["m2", "m3", "m4"]

        async with stream.where(TextEvent, source="a").subscribe(replay=True) as sub:
            await stream.emit(TextEvent(text="m5", source="a"))
            got = [
                (await anext(sub)).text,
                (await anext(sub)).text,
                (await anext(sub)).text,
            ]
            assert got == ["m3", "m4", "m5"]


async def test_memory_stream_drop_newest_overflow_policy() -> None:
    async with (
        MemoryStream(overflow_policy="drop_newest") as stream,
        stream.where(TextEvent).subscribe(max_buffer_size=2) as sub,
    ):
        await stream.emit(TextEvent(text="1"))
        await stream.emit(TextEvent(text="2"))
        await stream.emit(TextEvent(text="3"))  # dropped due to full buffer

        assert (await anext(sub)).text == "1"
        assert (await anext(sub)).text == "2"

        await stream.emit(TextEvent(text="4"))
        assert (await anext(sub)).text == "4"


async def test_memory_stream_cancellation_cleanup() -> None:
    async with MemoryStream() as stream:
        with anyio.CancelScope() as scope:
            async with stream.subscribe() as sub:
                assert stream.subscriber_count == 1
                scope.cancel()
                await anext(sub)

        assert stream.subscriber_count == 0


async def test_memory_stream_handles_early_unsubscribe_during_emit() -> None:
    async with MemoryStream() as stream:
        sub1 = stream.subscribe()
        sub2 = stream.subscribe()
        await sub1.aclose()
        assert stream.subscriber_count == 1

        await stream.emit(BaseEvent(event_type="custom"))
        ev = await anext(sub2)
        assert ev.event_type == "custom"
        await sub2.aclose()


def test_events_validation_errors_on_invalid_filter() -> None:
    stream = MemoryStream()
    with pytest.raises(TypeError):
        stream.where(123)  # type: ignore[call-overload]


def test_events_does_not_import_downstream_layers() -> None:
    forbidden = (
        "ag2.config",
        "ag2.providers",
        "ag2.agent",
        "ag2.tools",
        "ag2.mcp",
        "ag2.network",
        "ag2.knowledge",
    )
    events_dir = Path(ag2.events.__file__).parent
    for py_file in events_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden), (
                        f"{py_file.name} must not import {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith(forbidden), (
                    f"{py_file.name} must not import {node.module}"
                )
