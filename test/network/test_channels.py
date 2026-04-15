"""Unit and concurrency tests for conversation, consulting, and discussion channels."""

from __future__ import annotations

from collections.abc import Sequence

import anyio
import pytest

from ag2.agent import Agent
from ag2.events import ChatMessage, Usage
from ag2.network import (
    ChannelTurn,
    ConsultingChannel,
    ConversationChannel,
    DiscussionChannel,
    Hub,
    consulting,
    conversation,
    discussion,
)
from ag2.providers import ModelResponse, ResponseFormat, ToolSpec


class _ScriptedPeer:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.seen_prompts: list[str] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        del tools, response_format
        self.seen_prompts.append(messages[-1].content or "")
        text = self._responses.pop(0)
        return ModelResponse(
            message=ChatMessage(role="assistant", content=text),
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="fake",
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        return None


async def test_conversation_channel_alternates_turns_stops_early_and_logs_wal() -> None:
    alice_client = _ScriptedPeer(["draft v1", "DONE: approved"])
    bob_client = _ScriptedPeer(["looks good except formatting"])

    async with Hub() as hub:
        alice = Agent(name="alice", model=alice_client, stream=hub.stream)
        bob = Agent(name="bob", model=bob_client, stream=hub.stream)

        result = await conversation(
            hub,
            "Write release note",
            initiator=alice,
            responder=bob,
            name="release-chat",
            max_turns=6,
            stop_when=lambda turn, _hist: turn.text.startswith("DONE:"),
            run_id="run-conv-1",
        )

        assert result.channel_type == "conversation"
        assert result.termination_reason == "stop_predicate"
        assert result.final_speaker == "alice"
        assert result.final_text == "DONE: approved"
        assert len(result.turns) == 3
        assert result.usage.total_tokens == 45
        assert bob_client.seen_prompts[0] == "draft v1"
        assert alice_client.seen_prompts[1] == "looks good except formatting"

        wal_entries = await hub.read_wal(run_id="run-conv-1", channel="release-chat")
        assert len(wal_entries) == 3
        assert [e.sender for e in wal_entries] == ["alice", "bob", "alice"]


async def test_consulting_channel_runs_consultants_concurrently_then_synthesizes() -> None:
    sec_started = anyio.Event()
    perf_started = anyio.Event()

    class _BarrierClient:
        def __init__(self, my_event: anyio.Event, peer_event: anyio.Event, reply: str) -> None:
            self._my = my_event
            self._peer = peer_event
            self._reply = reply

        async def complete(
            self,
            messages: Sequence[ChatMessage],
            *,
            tools: Sequence[ToolSpec] = (),
            response_format: ResponseFormat | None = None,
        ) -> ModelResponse:
            del messages, tools, response_format
            self._my.set()
            await self._peer.wait()  # Deadlocks unless consultants execute concurrently.
            return ModelResponse(
                message=ChatMessage(role="assistant", content=self._reply),
                usage=Usage(prompt_tokens=8, completion_tokens=4, total_tokens=12),
                model="fake",
            )

        async def aclose(self) -> None:
            return None

    lead_client = _ScriptedPeer(["Combined plan: zero trust + caching"])

    async with Hub() as hub:
        lead = Agent(name="lead", model=lead_client, stream=hub.stream)
        sec = Agent(
            name="security",
            model=_BarrierClient(sec_started, perf_started, "Use mTLS"),
            stream=hub.stream,
        )
        perf = Agent(
            name="perf",
            model=_BarrierClient(perf_started, sec_started, "Add Redis cache"),
            stream=hub.stream,
        )

        with anyio.fail_after(2.0):
            res = await consulting(
                hub,
                "Design gateway",
                requester=lead,
                consultants=[sec, perf],
                name="arch-review",
                run_id="run-consult-1",
            )

        assert res.termination_reason == "synthesized"
        assert res.final_speaker == "lead"
        assert res.final_text == "Combined plan: zero trust + caching"
        assert [t.speaker for t in res.turns] == ["security", "perf", "lead"]
        assert "Use mTLS" in lead_client.seen_prompts[0]
        assert "Add Redis cache" in lead_client.seen_prompts[0]

        wal_entries = await hub.read_wal(run_id="run-consult-1", channel="arch-review")
        assert len(wal_entries) == 3


async def test_discussion_channel_supports_custom_selector_and_moderator() -> None:
    p1 = _ScriptedPeer(["Option A has lower latency"])
    p2 = _ScriptedPeer(["Option B is simpler"])
    mod = _ScriptedPeer(["Consensus: adopt Option B initially"])

    async def _pick_reverse(history: tuple[ChannelTurn, ...], _agents: tuple[Agent, ...]) -> str:
        return "beta" if len(history) == 0 else "alpha"

    async with Hub() as hub:
        alpha = Agent(name="alpha", model=p1, stream=hub.stream)
        beta = Agent(name="beta", model=p2, stream=hub.stream)
        chair = Agent(name="chair", model=mod, stream=hub.stream)
        hub.register(alpha)
        hub.register(beta)
        hub.register(chair)

        res = await discussion(
            hub,
            "Choose queueing library",
            participants=["alpha", "beta"],
            moderator="chair",
            max_rounds=2,
            speaker_selector=_pick_reverse,
            name="tech-council",
            run_id="run-disc-1",
        )

        assert res.final_speaker == "chair"
        assert res.final_text == "Consensus: adopt Option B initially"
        assert [t.speaker for t in res.turns] == ["beta", "alpha", "chair"]
        assert "[beta] Option B is simpler" in p1.seen_prompts[0]

        wal_entries = await hub.read_wal(run_id="run-disc-1", channel="tech-council")
        assert [e.sender for e in wal_entries] == ["beta", "alpha", "chair"]


def test_channel_validation_errors() -> None:
    hub = Hub()
    a = Agent(name="a", model=_ScriptedPeer([]))
    b = Agent(name="b", model=_ScriptedPeer([]))

    with pytest.raises(ValueError, match="max_turns"):
        ConversationChannel(hub, initiator=a, responder=b, max_turns=0)

    with pytest.raises(ValueError, match="consultants"):
        ConsultingChannel(hub, requester=a, consultants=[])

    with pytest.raises(ValueError, match="participants"):
        DiscussionChannel(hub, participants=[])
