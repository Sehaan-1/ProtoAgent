"""Composable agent message-assembly middleware and working-memory policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ag2.compact.compactor import CompactionStrategy, CompactTrigger, TailWindowCompact
from ag2.events.messages import ChatMessage
from ag2.events.stream import MemoryStream
from ag2.knowledge.plugin import KnowledgePlugin


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Per-turn context passed through the Agent middleware pipeline."""

    prompt: ChatMessage
    source: str
    run_id: str
    stream: MemoryStream


@runtime_checkable
class MessageMiddleware(Protocol):
    """Async hook transforming model-bound messages prior to ModelClient.complete()."""

    async def transform(
        self,
        messages: list[ChatMessage],
        ctx: TurnContext,
    ) -> list[ChatMessage]:
        """Return transformed messages for the upcoming model request."""
        ...


class WorkingMemoryPolicy:
    """MessageMiddleware enforcing working-memory bounds with optional compaction."""

    def __init__(
        self,
        *,
        trigger: CompactTrigger | None = None,
        compactor: CompactionStrategy | None = None,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        preserve_recent: int = 6,
    ) -> None:
        if trigger is None:
            self.trigger = CompactTrigger(
                max_messages=max_messages,
                max_tokens=max_tokens,
                preserve_recent=preserve_recent,
            )
        else:
            self.trigger = trigger
        self.compactor: CompactionStrategy = (
            compactor
            if compactor is not None
            else TailWindowCompact(preserve_recent=self.trigger.preserve_recent)
        )

    async def transform(
        self,
        messages: list[ChatMessage],
        ctx: TurnContext,
    ) -> list[ChatMessage]:
        """Compact ``messages`` when ``trigger.should_compact(messages)`` holds."""
        if not self.trigger.should_compact(messages):
            return messages
        return await self.compactor.compact(
            messages,
            stream=ctx.stream,
            source=ctx.source,
            run_id=ctx.run_id,
        )


class AssemblerMiddleware:
    """Prompt assembly pipeline combining policy compaction, knowledge lookup, and guidance."""

    def __init__(
        self,
        *,
        knowledge: KnowledgePlugin | None = None,
        working_memory: WorkingMemoryPolicy | None = None,
        system_prefix: str | None = None,
        system_suffix: str | None = None,
        extra_middlewares: Sequence[MessageMiddleware] = (),
        knowledge_limit: int = 4,
    ) -> None:
        if knowledge_limit < 1:
            raise ValueError("knowledge_limit must be >= 1")
        self.knowledge = knowledge
        self.working_memory = working_memory
        self.system_prefix = system_prefix
        self.system_suffix = system_suffix
        self.extra_middlewares = tuple(extra_middlewares)
        self.knowledge_limit = knowledge_limit

    async def transform(
        self,
        messages: list[ChatMessage],
        ctx: TurnContext,
    ) -> list[ChatMessage]:
        """Assemble model-ready messages cleanly without mutating caller history."""
        current = list(messages)

        if self.working_memory is not None:
            current = await self.working_memory.transform(current, ctx)

        for mw in self.extra_middlewares:
            current = await mw.transform(current, ctx)

        if self.system_prefix:
            current.insert(0, ChatMessage(role="system", content=self.system_prefix))

        if self.knowledge is not None and ctx.prompt.content:
            kb_block = await self.knowledge.format_context(
                ctx.prompt.content,
                limit=self.knowledge_limit,
            )
            if kb_block:
                # Inject retrieved context right before the trailing messages / latest prompt.
                insert_idx = max(0, len(current) - 1)
                current.insert(insert_idx, ChatMessage(role="system", content=kb_block))

        if self.system_suffix:
            current.append(ChatMessage(role="system", content=self.system_suffix))

        return current
