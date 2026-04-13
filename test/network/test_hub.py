"""Unit and persistence tests for Hub registry and AsyncWAL."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

import anyio
import pytest

import ag2.network
from ag2.agent import Agent
from ag2.events import ChatMessage, MemoryStream, TextEvent
from ag2.network import (
    AgentNotFoundError,
    AsyncWAL,
    DuplicateAgentError,
    Hub,
    WALEntry,
)
from ag2.providers import ModelResponse, ResponseFormat, ToolSpec


class _StubClient:
    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text
        self.closed = False

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        del messages, tools, response_format
        return ModelResponse(
            message=ChatMessage(role="assistant", content=self._reply_text),
            model="stub-model",
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        self.closed = True


async def test_async_wal_persists_across_reopen_and_filters(tmp_path: Path) -> None:
    db_file = tmp_path / "wal" / "events.db"

    async with AsyncWAL(db_file) as wal:
        assert (await wal.journal_mode()).lower() == "wal"
        e1 = await wal.append(
            run_id="run-1",
            channel="room-a",
            kind="message",
            sender="alice",
            recipient="bob",
            payload={"text": "hello"},
        )
        e2 = await wal.append(
            run_id="run-1",
            channel="room-b",
            kind="handoff",
            sender="bob",
            payload={"next": "carol"},
        )
        e3 = await wal.append(
            run_id="run-2",
            channel="room-a",
            kind="message",
            sender="dave",
            payload={"text": "other"},
        )
        assert [e1.seq, e2.seq, e3.seq] == [1, 2, 3]
        assert isinstance(e1, WALEntry)

    # Re-open from disk and verify durable recovery & query filters.
    async with AsyncWAL(db_file) as reopened:
        all_entries = await reopened.read_wal()
        assert len(all_entries) == 3
        assert [e.entry_id for e in all_entries] == [e1.entry_id, e2.entry_id, e3.entry_id]

        run1_after1 = await reopened.read_wal(run_id="run-1", after_seq=1)
        assert [e.seq for e in run1_after1] == [2]
        assert run1_after1[0].kind == "handoff"

        room_a_first = await reopened.read_wal(channel="room-a", limit=1)
        assert len(room_a_first) == 1
        assert room_a_first[0].sender == "alice"


async def test_async_wal_concurrent_writes_preserve_monotonic_sequence(tmp_path: Path) -> None:
    db_file = tmp_path / "concurrent.db"

    async with AsyncWAL(db_file) as wal:

        async def _writer(worker_id: int) -> None:
            for step in range(5):
                await wal.append(
                    run_id="concurrent-run",
                    kind="tick",
                    sender=f"w-{worker_id}",
                    payload={"step": step},
                )

        async with anyio.create_task_group() as tg:
            for idx in range(4):
                tg.start_soon(_writer, idx)

        entries = await wal.read_wal(run_id="concurrent-run")
        assert len(entries) == 20
        assert [e.seq for e in entries] == list(range(1, 21))


async def test_hub_registry_audit_recording_and_lifecycle(tmp_path: Path) -> None:
    db_file = tmp_path / "hub.db"
    c1 = _StubClient("hi from alpha")
    c2 = _StubClient("hi from beta")

    async with MemoryStream(history_limit=50) as shared_stream:
        alpha = Agent(name="alpha", model=c1, stream=shared_stream)
        beta = Agent(name="beta", model=c2, stream=shared_stream)

        async with Hub([alpha], stream=shared_stream, wal_path=db_file, close_agents=True) as hub:
            hub.register(beta)
            assert len(hub) == 2
            assert hub.agent_names == ("alpha", "beta")
            assert "alpha" in hub
            assert hub["beta"] is beta

            with pytest.raises(DuplicateAgentError):
                hub.register(Agent(name="alpha", model=c1))

            reply = await hub["alpha"].ask("ping", run_id="run-hub-1")
            await hub.record_event(
                TextEvent(text=reply.text, source="alpha", run_id=reply.run_id),
                channel="direct",
                recipient="user",
            )

            wal_entries = await hub.read_wal(run_id="run-hub-1")
            assert len(wal_entries) == 1
            assert wal_entries[0].sender == "alpha"
            assert wal_entries[0].recipient == "user"
            assert wal_entries[0].payload["text"] == "hi from alpha"

            hub.unregister("beta")
            assert "beta" not in hub
            with pytest.raises(AgentNotFoundError):
                hub.get("beta")

        # Caller-provided stream remains open; close_agents=True leaves remaining alpha closed.
        assert hub.closed is True
        assert hub.wal.closed is True
        assert shared_stream.closed is False


def test_network_does_not_import_mcp_or_harness_layers() -> None:
    forbidden = ("ag2.mcp", "ag2.knowledge", "ag2.middleware", "ag2.compact")
    network_dir = Path(ag2.network.__file__).parent
    for py_file in network_dir.glob("*.py"):
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
