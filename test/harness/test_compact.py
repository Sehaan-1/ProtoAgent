"""Unit tests for CompactTrigger, TailWindowCompact, and SummarizeCompact."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ag2.compact import (
    CompactTrigger,
    SummarizeCompact,
    TailWindowCompact,
    estimate_tokens,
    partition_messages,
)
from ag2.events import (
    ChatMessage,
    CompactionCompleted,
    CompactionFailed,
    CompactionStarted,
    MemoryStream,
    ToolCall,
)
from ag2.providers import ModelResponse, ProviderError, ResponseFormat, ToolSpec


class _FakeSummarizerClient:
    def __init__(self, response: str | Exception) -> None:
        self._response = response
        self.calls: list[list[ChatMessage]] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        del tools, response_format
        self.calls.append(list(messages))
        if isinstance(self._response, Exception):
            raise self._response
        return ModelResponse(
            message=ChatMessage(role="assistant", content=self._response),
            model="summarizer-mini",
        )

    async def aclose(self) -> None:
        return None


def test_partition_messages_preserves_system_and_tool_call_pairs() -> None:
    messages = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="u1"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(
            role="assistant",
            tool_calls=[ToolCall(id="c1", name="search", arguments={"q": "x"})],
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="result-1"),
        ChatMessage(role="assistant", content="final"),
    ]
    # preserve_recent=2 lands on tool output at index 4 -> must shift cut back to index 3.
    leading, middle, tail = partition_messages(messages, preserve_recent=2)
    assert [m.content for m in leading] == ["sys"]
    assert [m.content for m in middle] == ["u1", "a1"]
    assert len(tail) == 3
    assert tail[0].role == "assistant" and tail[0].tool_calls
    assert tail[1].role == "tool"


async def test_tail_window_and_summarize_compact_emit_events() -> None:
    messages = [
        ChatMessage(role="system", content="Be accurate."),
        ChatMessage(role="user", content="Turn 1"),
        ChatMessage(role="assistant", content="Ack 1"),
        ChatMessage(role="user", content="Turn 2"),
        ChatMessage(role="assistant", content="Ack 2"),
        ChatMessage(role="user", content="Latest question?"),
    ]
    trigger = CompactTrigger(max_messages=4, preserve_recent=2, min_messages_to_compact=2)
    assert trigger.should_compact(messages) is True
    assert estimate_tokens(messages) > 0

    async with MemoryStream(history_limit=20) as stream:
        tail_compacted = await TailWindowCompact(preserve_recent=2).compact(
            messages,
            stream=stream,
            source="agent-a",
            run_id="r1",
        )
        assert len(tail_compacted) == 4
        assert "[Compacted 3 earlier messages]" in (tail_compacted[1].content or "")

        summarizer = _FakeSummarizerClient("User completed Turn 1 and Turn 2.")
        sum_compacted = await SummarizeCompact(summarizer, preserve_recent=2).compact(
            messages,
            stream=stream,
            source="agent-a",
            run_id="r2",
        )
        assert len(sum_compacted) == 4
        assert sum_compacted[0].content == "Be accurate."
        assert "User completed Turn 1 and Turn 2." in (sum_compacted[1].content or "")
        assert [m.content for m in sum_compacted[2:]] == ["Ack 2", "Latest question?"]

        started = [e for e in stream.history if isinstance(e, CompactionStarted)]
        completed = [e for e in stream.history if isinstance(e, CompactionCompleted)]
        assert len(started) == 2
        assert len(completed) == 2
        assert completed[1].summary == "User completed Turn 1 and Turn 2."

        failing = SummarizeCompact(
            _FakeSummarizerClient(ProviderError("timeout")), preserve_recent=2
        )
        with pytest.raises(ProviderError):
            await failing.compact(messages, stream=stream, source="agent-a", run_id="r3")

        failed = [e for e in stream.history if isinstance(e, CompactionFailed)]
        assert len(failed) == 1
        assert "timeout" in failed[0].error
