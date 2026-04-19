"""Conversation history compaction triggers and summarization strategies."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import structlog
from opentelemetry import trace

from ag2.events.messages import ChatMessage
from ag2.events.models import CompactionCompleted, CompactionFailed, CompactionStarted
from ag2.events.stream import MemoryStream
from ag2.providers.base import ModelClient

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.compact")


def estimate_tokens(messages: Sequence[ChatMessage]) -> int:
    """Estimate approximate prompt token count across a sequence of ChatMessages."""
    total = 0
    for msg in messages:
        content_len = len(msg.content or "")
        calls_len = len(json.dumps([tc.model_dump(mode="json") for tc in msg.tool_calls]))
        total += max(4, (content_len + calls_len) // 4 + 4)
    return total


@dataclass(frozen=True, slots=True)
class CompactTrigger:
    """Threshold policy deciding when conversation history requires compaction."""

    max_messages: int | None = None
    max_tokens: int | None = None
    preserve_recent: int = 4
    min_messages_to_compact: int = 2

    def __post_init__(self) -> None:
        if self.max_messages is not None and self.max_messages < 2:
            raise ValueError("max_messages must be >= 2")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.preserve_recent < 1:
            raise ValueError("preserve_recent must be >= 1")
        if self.min_messages_to_compact < 1:
            raise ValueError("min_messages_to_compact must be >= 1")

    def should_compact(self, messages: Sequence[ChatMessage]) -> bool:
        """Return True if message count or estimated tokens exceed configured budgets."""
        if len(messages) <= self.preserve_recent:
            return False
        _, middle, _ = partition_messages(messages, preserve_recent=self.preserve_recent)
        if len(middle) < self.min_messages_to_compact:
            return False
        if self.max_messages is not None and len(messages) > self.max_messages:
            return True
        return self.max_tokens is not None and estimate_tokens(messages) > self.max_tokens


def partition_messages(
    messages: Sequence[ChatMessage],
    *,
    preserve_recent: int,
) -> tuple[list[ChatMessage], list[ChatMessage], list[ChatMessage]]:
    """Split ``messages`` into ``(leading_system, compactable_middle, recent_tail)``.

    Adjusts the tail cut backwards so an assistant ``tool_calls`` message is never
    separated from its corresponding ``tool`` response messages.
    """
    msgs = list(messages)
    sys_end = 0
    while sys_end < len(msgs) and msgs[sys_end].role == "system":
        sys_end += 1

    leading_system = msgs[:sys_end]
    rest = msgs[sys_end:]
    if len(rest) <= preserve_recent:
        return leading_system, [], rest

    cut = len(rest) - preserve_recent
    # Never start the preserved tail on orphaned tool outputs.
    while cut > 0 and rest[cut].role == "tool":
        cut -= 1

    return leading_system, rest[:cut], rest[cut:]


@runtime_checkable
class CompactionStrategy(Protocol):
    """Async protocol implemented by conversation history compactors."""

    async def compact(
        self,
        messages: Sequence[ChatMessage],
        *,
        stream: MemoryStream | None = None,
        source: str | None = None,
        run_id: str | None = None,
    ) -> list[ChatMessage]:
        """Return a compacted list of ChatMessages, emitting compaction lifecycle events."""
        ...


class TailWindowCompact:
    """Zero-LLM deterministic compactor replacing old middle turns with a brief marker."""

    strategy_name = "tail_window"

    def __init__(self, *, preserve_recent: int = 4) -> None:
        if preserve_recent < 1:
            raise ValueError("preserve_recent must be >= 1")
        self.preserve_recent = preserve_recent

    async def compact(
        self,
        messages: Sequence[ChatMessage],
        *,
        stream: MemoryStream | None = None,
        source: str | None = None,
        run_id: str | None = None,
    ) -> list[ChatMessage]:
        """Drop compactable middle messages while keeping system prompts and recent tail."""
        leading, middle, tail = partition_messages(messages, preserve_recent=self.preserve_recent)
        if not middle:
            return list(messages)

        if stream is not None:
            await stream.emit(
                CompactionStarted(
                    strategy=self.strategy_name,
                    original_message_count=len(messages),
                    original_token_estimate=estimate_tokens(messages),
                    source=source,
                    run_id=run_id,
                )
            )

        note = f"[Compacted {len(middle)} earlier messages]"
        compacted = [
            *leading,
            ChatMessage(role="system", content=note),
            *tail,
        ]
        if stream is not None:
            await stream.emit(
                CompactionCompleted(
                    strategy=self.strategy_name,
                    original_message_count=len(messages),
                    compacted_message_count=len(compacted),
                    summary=note,
                    source=source,
                    run_id=run_id,
                )
            )
        return compacted


class SummarizeCompact:
    """LLM-backed compactor summarizing dropped history turns into a system digest."""

    strategy_name = "summarize"

    def __init__(
        self,
        client: ModelClient,
        *,
        preserve_recent: int = 4,
        summary_instruction: str = (
            "Summarize the key facts, decisions, and open tasks from this conversation excerpt "
            "concisely so another assistant turn can continue accurately."
        ),
    ) -> None:
        if preserve_recent < 1:
            raise ValueError("preserve_recent must be >= 1")
        self._client = client
        self.preserve_recent = preserve_recent
        self.summary_instruction = summary_instruction

    async def compact(
        self,
        messages: Sequence[ChatMessage],
        *,
        stream: MemoryStream | None = None,
        source: str | None = None,
        run_id: str | None = None,
    ) -> list[ChatMessage]:
        """Summarize compactable middle messages using ``client`` and emit audit events."""
        leading, middle, tail = partition_messages(messages, preserve_recent=self.preserve_recent)
        if not middle:
            return list(messages)

        if stream is not None:
            await stream.emit(
                CompactionStarted(
                    strategy=self.strategy_name,
                    original_message_count=len(messages),
                    original_token_estimate=estimate_tokens(messages),
                    source=source,
                    run_id=run_id,
                )
            )

        transcript_lines = [
            f"{m.role}: {m.content or f'[tool_calls:{len(m.tool_calls)}]'}" for m in middle
        ]
        prompt_messages = [
            ChatMessage(role="system", content=self.summary_instruction),
            ChatMessage(role="user", content="\n".join(transcript_lines)),
        ]

        try:
            with tracer.start_as_current_span(
                "ag2.compact.summarize",
                attributes={"ag2.compact.dropped_messages": len(middle)},
            ):
                response = await self._client.complete(prompt_messages)
        except Exception as exc:
            logger.warning("ag2.compact.failed", strategy=self.strategy_name, error=str(exc))
            if stream is not None and not stream.closed:
                await stream.emit(
                    CompactionFailed(
                        strategy=self.strategy_name,
                        error=str(exc),
                        source=source,
                        run_id=run_id,
                    )
                )
            raise

        summary_text = (response.message.content or "").strip() or "(empty summary)"
        summary_msg = ChatMessage(
            role="system",
            content=f"[Conversation Summary]\n{summary_text}",
        )
        compacted = [*leading, summary_msg, *tail]

        if stream is not None:
            await stream.emit(
                CompactionCompleted(
                    strategy=self.strategy_name,
                    original_message_count=len(messages),
                    compacted_message_count=len(compacted),
                    summary=summary_text,
                    source=source,
                    run_id=run_id,
                )
            )
        return compacted
