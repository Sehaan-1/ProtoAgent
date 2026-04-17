"""Typed multi-agent communication channels: conversation, consulting, and discussion."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

import anyio
import structlog
from opentelemetry import trace

from ag2.agent.agent import Agent, AgentReply
from ag2.events.messages import Usage
from ag2.network.hub import Hub

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.network.channels")

ChannelKind = Literal["conversation", "consulting", "discussion", "workflow"]
StopPredicate = Callable[["ChannelTurn", tuple["ChannelTurn", ...]], bool]
SpeakerSelector = Callable[
    [tuple["ChannelTurn", ...], tuple[Agent, ...]],
    str | Agent | Awaitable[str | Agent],
]


def _add_usage(acc: Usage, delta: Usage) -> Usage:
    return Usage(
        prompt_tokens=acc.prompt_tokens + delta.prompt_tokens,
        completion_tokens=acc.completion_tokens + delta.completion_tokens,
        total_tokens=acc.total_tokens + delta.total_tokens,
    )


def _resolve_agent(hub: Hub, ref: Agent | str) -> Agent:
    if isinstance(ref, Agent):
        if ref.name not in hub:
            hub.register(ref)
        return ref
    return hub.get(ref)


@dataclass(frozen=True, slots=True)
class ChannelTurn:
    """Single agent execution turn recorded inside a multi-agent channel."""

    turn_index: int
    speaker: str
    text: str
    reply: AgentReply[Any]
    usage: Usage


@dataclass(frozen=True, slots=True)
class ChannelResult:
    """Final aggregated outcome from running a typed multi-agent channel."""

    channel: str
    channel_type: ChannelKind
    run_id: str
    final_text: str
    final_speaker: str
    turns: tuple[ChannelTurn, ...]
    usage: Usage
    termination_reason: str


async def _run_and_log_turn(
    hub: Hub,
    agent: Agent,
    prompt: str,
    *,
    turn_index: int,
    run_id: str,
    channel_name: str,
    channel_type: ChannelKind,
    recipient: str | None = None,
) -> ChannelTurn:
    reply = await agent.ask(prompt, run_id=run_id)
    turn = ChannelTurn(
        turn_index=turn_index,
        speaker=agent.name,
        text=reply.text,
        reply=reply,
        usage=reply.usage,
    )
    await hub.append_wal(
        run_id=run_id,
        channel=channel_name,
        kind=f"{channel_type}.turn",
        sender=agent.name,
        recipient=recipient,
        payload={
            "turn_index": turn_index,
            "prompt": prompt,
            "text": reply.text,
            "tokens": reply.usage.total_tokens,
        },
    )
    return turn


class ConversationChannel:
    """Two-agent alternating dialogue channel with max_turns and predicate termination."""

    def __init__(
        self,
        hub: Hub,
        *,
        name: str = "conversation",
        initiator: Agent | str,
        responder: Agent | str,
        max_turns: int = 6,
        stop_when: StopPredicate | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.hub = hub
        self.name = name
        self.initiator = _resolve_agent(hub, initiator)
        self.responder = _resolve_agent(hub, responder)
        self.max_turns = max_turns
        self.stop_when = stop_when

    async def run(self, prompt: str, *, run_id: str | None = None) -> ChannelResult:
        """Run alternating turns between initiator and responder."""
        effective_run_id = run_id or uuid4().hex
        turns: list[ChannelTurn] = []
        total_usage = Usage()
        current_prompt = prompt
        termination = "max_turns"

        peers = (self.initiator, self.responder)
        with tracer.start_as_current_span(
            "ag2.network.channel.conversation",
            attributes={"ag2.channel": self.name, "ag2.run_id": effective_run_id},
        ):
            for idx in range(self.max_turns):
                speaker = peers[idx % 2]
                peer = peers[(idx + 1) % 2]
                turn = await _run_and_log_turn(
                    self.hub,
                    speaker,
                    current_prompt,
                    turn_index=idx,
                    run_id=effective_run_id,
                    channel_name=self.name,
                    channel_type="conversation",
                    recipient=peer.name,
                )
                turns.append(turn)
                total_usage = _add_usage(total_usage, turn.usage)
                current_prompt = turn.text
                if self.stop_when is not None and self.stop_when(turn, tuple(turns)):
                    termination = "stop_predicate"
                    break

        last = turns[-1]
        return ChannelResult(
            channel=self.name,
            channel_type="conversation",
            run_id=effective_run_id,
            final_text=last.text,
            final_speaker=last.speaker,
            turns=tuple(turns),
            usage=total_usage,
            termination_reason=termination,
        )


class ConsultingChannel:
    """Hub-and-spoke channel querying consultants concurrently, then synthesizing."""

    def __init__(
        self,
        hub: Hub,
        *,
        name: str = "consulting",
        requester: Agent | str,
        consultants: Sequence[Agent | str],
    ) -> None:
        if not consultants:
            raise ValueError("consultants must contain at least one agent")
        self.hub = hub
        self.name = name
        self.requester = _resolve_agent(hub, requester)
        self.consultants = tuple(_resolve_agent(hub, item) for item in consultants)

    async def run(self, question: str, *, run_id: str | None = None) -> ChannelResult:
        """Consult all specialists concurrently and ask requester to synthesize."""
        effective_run_id = run_id or uuid4().hex
        consultant_turns: list[ChannelTurn | None] = [None] * len(self.consultants)

        with tracer.start_as_current_span(
            "ag2.network.channel.consulting",
            attributes={"ag2.channel": self.name, "ag2.run_id": effective_run_id},
        ):

            async def _ask_consultant(index: int, consultant: Agent) -> None:
                brief = (
                    f"Consulting request from {self.requester.name}:\n"
                    f"{question}\n\n"
                    "Provide focused expert input."
                )
                consultant_turns[index] = await _run_and_log_turn(
                    self.hub,
                    consultant,
                    brief,
                    turn_index=index,
                    run_id=effective_run_id,
                    channel_name=self.name,
                    channel_type="consulting",
                    recipient=self.requester.name,
                )

            async with anyio.create_task_group() as tg:
                for idx, consultant in enumerate(self.consultants):
                    tg.start_soon(_ask_consultant, idx, consultant)

            completed: list[ChannelTurn] = []
            for item in consultant_turns:
                assert item is not None
                completed.append(item)

            notes = "\n".join(f"- {turn.speaker}: {turn.text}" for turn in completed)
            synthesis_prompt = (
                f"Original question: {question}\n\n"
                f"Consultant responses:\n{notes}\n\n"
                "Synthesize a clear final answer."
            )
            synthesis_turn = await _run_and_log_turn(
                self.hub,
                self.requester,
                synthesis_prompt,
                turn_index=len(completed),
                run_id=effective_run_id,
                channel_name=self.name,
                channel_type="consulting",
                recipient=None,
            )

        all_turns = (*completed, synthesis_turn)
        total_usage = Usage()
        for t in all_turns:
            total_usage = _add_usage(total_usage, t.usage)

        return ChannelResult(
            channel=self.name,
            channel_type="consulting",
            run_id=effective_run_id,
            final_text=synthesis_turn.text,
            final_speaker=synthesis_turn.speaker,
            turns=all_turns,
            usage=total_usage,
            termination_reason="synthesized",
        )


class DiscussionChannel:
    """Shared-context group discussion with speaker selection and optional moderator."""

    def __init__(
        self,
        hub: Hub,
        *,
        name: str = "discussion",
        participants: Sequence[Agent | str],
        moderator: Agent | str | None = None,
        max_rounds: int = 6,
        speaker_selector: SpeakerSelector | None = None,
        stop_when: StopPredicate | None = None,
    ) -> None:
        if not participants:
            raise ValueError("participants must contain at least one agent")
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.hub = hub
        self.name = name
        self.participants = tuple(_resolve_agent(hub, item) for item in participants)
        self.moderator = _resolve_agent(hub, moderator) if moderator is not None else None
        self.max_rounds = max_rounds
        self.speaker_selector = speaker_selector
        self.stop_when = stop_when

    async def _select_speaker(self, round_idx: int, history: tuple[ChannelTurn, ...]) -> Agent:
        if self.speaker_selector is None:
            return self.participants[round_idx % len(self.participants)]

        selected = self.speaker_selector(history, self.participants)
        if inspect.isawaitable(selected):
            resolved = await selected
        else:
            resolved = selected
        if isinstance(resolved, Agent):
            return resolved
        for participant in self.participants:
            if participant.name == resolved:
                return participant
        return self.hub.get(resolved)

    async def run(self, topic: str, *, run_id: str | None = None) -> ChannelResult:
        """Execute up to ``max_rounds`` discussion turns plus optional moderator wrap-up."""
        effective_run_id = run_id or uuid4().hex
        turns: list[ChannelTurn] = []
        total_usage = Usage()
        termination = "max_rounds"

        with tracer.start_as_current_span(
            "ag2.network.channel.discussion",
            attributes={"ag2.channel": self.name, "ag2.run_id": effective_run_id},
        ):
            for idx in range(self.max_rounds):
                speaker = await self._select_speaker(idx, tuple(turns))
                transcript = "\n".join(f"[{t.speaker}] {t.text}" for t in turns)
                turn_prompt = (
                    f"Discussion topic: {topic}\n\n"
                    f"Transcript so far:\n{transcript or '(opening round)'}\n\n"
                    f"Respond as {speaker.name}."
                )
                turn = await _run_and_log_turn(
                    self.hub,
                    speaker,
                    turn_prompt,
                    turn_index=idx,
                    run_id=effective_run_id,
                    channel_name=self.name,
                    channel_type="discussion",
                )
                turns.append(turn)
                total_usage = _add_usage(total_usage, turn.usage)
                if self.stop_when is not None and self.stop_when(turn, tuple(turns)):
                    termination = "stop_predicate"
                    break

            if self.moderator is not None:
                final_transcript = "\n".join(f"[{t.speaker}] {t.text}" for t in turns)
                mod_prompt = (
                    f"Discussion topic: {topic}\n\n"
                    f"Full transcript:\n{final_transcript}\n\n"
                    "Summarize the outcome."
                )
                mod_turn = await _run_and_log_turn(
                    self.hub,
                    self.moderator,
                    mod_prompt,
                    turn_index=len(turns),
                    run_id=effective_run_id,
                    channel_name=self.name,
                    channel_type="discussion",
                )
                turns.append(mod_turn)
                total_usage = _add_usage(total_usage, mod_turn.usage)

        last = turns[-1]
        return ChannelResult(
            channel=self.name,
            channel_type="discussion",
            run_id=effective_run_id,
            final_text=last.text,
            final_speaker=last.speaker,
            turns=tuple(turns),
            usage=total_usage,
            termination_reason=termination,
        )


async def conversation(
    hub: Hub,
    prompt: str,
    *,
    initiator: Agent | str,
    responder: Agent | str,
    name: str = "conversation",
    max_turns: int = 6,
    stop_when: StopPredicate | None = None,
    run_id: str | None = None,
) -> ChannelResult:
    """Convenience function executing a :class:`ConversationChannel`."""
    return await ConversationChannel(
        hub,
        name=name,
        initiator=initiator,
        responder=responder,
        max_turns=max_turns,
        stop_when=stop_when,
    ).run(prompt, run_id=run_id)


async def consulting(
    hub: Hub,
    question: str,
    *,
    requester: Agent | str,
    consultants: Sequence[Agent | str],
    name: str = "consulting",
    run_id: str | None = None,
) -> ChannelResult:
    """Convenience function executing a :class:`ConsultingChannel`."""
    return await ConsultingChannel(
        hub,
        name=name,
        requester=requester,
        consultants=consultants,
    ).run(question, run_id=run_id)


async def discussion(
    hub: Hub,
    topic: str,
    *,
    participants: Sequence[Agent | str],
    moderator: Agent | str | None = None,
    name: str = "discussion",
    max_rounds: int = 6,
    speaker_selector: SpeakerSelector | None = None,
    stop_when: StopPredicate | None = None,
    run_id: str | None = None,
) -> ChannelResult:
    """Convenience function executing a :class:`DiscussionChannel`."""
    return await DiscussionChannel(
        hub,
        name=name,
        participants=participants,
        moderator=moderator,
        max_rounds=max_rounds,
        speaker_selector=speaker_selector,
        stop_when=stop_when,
    ).run(topic, run_id=run_id)
