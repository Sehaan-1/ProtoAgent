"""Multi-agent network Hub managing agent registry, lifecycle, stream, and WAL audit log."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import anyio
import structlog

from ag2.agent.agent import Agent
from ag2.events.models import BaseEvent
from ag2.events.stream import MemoryStream
from ag2.network.wal import AsyncWAL, WALEntry

logger = structlog.get_logger(__name__)


class HubError(RuntimeError):
    """Base error for Hub registry and lifecycle operations."""


class AgentNotFoundError(HubError, KeyError):
    """Raised when looking up an unregistered agent name in a Hub."""


class DuplicateAgentError(HubError, ValueError):
    """Raised when registering an agent whose name already exists in the Hub."""


class Hub:
    """Coordinator holding registered agents, shared event stream, and async WAL."""

    def __init__(
        self,
        agents: Sequence[Agent] = (),
        *,
        stream: MemoryStream | None = None,
        wal: AsyncWAL | None = None,
        wal_path: str | Path | None = None,
        close_agents: bool = False,
    ) -> None:
        if wal is not None and wal_path is not None:
            raise ValueError("Provide at most one of 'wal' or 'wal_path'")

        if stream is None:
            self.stream = MemoryStream()
            self._owns_stream = True
        else:
            self.stream = stream
            self._owns_stream = False

        if wal is not None:
            self.wal = wal
            self._owns_wal = False
        else:
            self.wal = AsyncWAL(wal_path if wal_path is not None else ":memory:")
            self._owns_wal = True

        self._close_agents = close_agents
        self._agents: dict[str, Agent] = {}
        self._closed = False

        for agent in agents:
            self.register(agent)

    @property
    def closed(self) -> bool:
        """Whether this Hub has been closed."""
        return self._closed

    @property
    def agent_names(self) -> tuple[str, ...]:
        """Ordered tuple of registered agent names."""
        return tuple(self._agents.keys())

    @property
    def agents(self) -> tuple[Agent, ...]:
        """Ordered tuple of registered Agent instances."""
        return tuple(self._agents.values())

    def register(self, agent: Agent, *, name: str | None = None) -> Agent:
        """Register an agent under ``name`` (or ``agent.name``), rejecting duplicates."""
        key = (name if name is not None else agent.name).strip()
        if not key:
            raise ValueError("Agent registration name must not be empty")
        if key in self._agents:
            raise DuplicateAgentError(f"Agent {key!r} is already registered in this Hub")
        self._agents[key] = agent
        logger.debug("ag2.network.hub.registered", agent=key)
        return agent

    def unregister(self, name: str) -> Agent:
        """Remove and return a registered agent by name."""
        agent = self._agents.pop(name, None)
        if agent is None:
            raise AgentNotFoundError(f"Agent {name!r} is not registered in this Hub")
        return agent

    def get(self, name: str) -> Agent:
        """Return the registered agent for ``name``, or raise AgentNotFoundError."""
        agent = self._agents.get(name)
        if agent is None:
            raise AgentNotFoundError(f"Agent {name!r} is not registered in this Hub")
        return agent

    def __getitem__(self, name: str) -> Agent:
        return self.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    async def append_wal(
        self,
        *,
        run_id: str,
        kind: str,
        channel: str | None = None,
        sender: str | None = None,
        recipient: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> WALEntry:
        """Append an entry to this Hub's write-ahead log."""
        return await self.wal.append(
            run_id=run_id,
            kind=kind,
            channel=channel,
            sender=sender,
            recipient=recipient,
            payload=payload,
        )

    async def record_event(
        self,
        event: BaseEvent,
        *,
        run_id: str | None = None,
        channel: str | None = None,
        sender: str | None = None,
        recipient: str | None = None,
    ) -> WALEntry:
        """Persist a typed ``BaseEvent`` to the WAL (and preserve event metadata)."""
        effective_run_id = run_id or event.run_id or "default"
        return await self.wal.append(
            entry_id=event.event_id,
            run_id=effective_run_id,
            channel=channel,
            kind=event.event_type,
            sender=sender if sender is not None else event.source,
            recipient=recipient,
            payload=event.model_dump(mode="json"),
            timestamp=event.timestamp,
        )

    async def read_wal(
        self,
        *,
        run_id: str | None = None,
        channel: str | None = None,
        kind: str | None = None,
        after_seq: int = 0,
        limit: int | None = None,
    ) -> list[WALEntry]:
        """Read entries from this Hub's write-ahead log in commit order."""
        return await self.wal.read_wal(
            run_id=run_id,
            channel=channel,
            kind=kind,
            after_seq=after_seq,
            limit=limit,
        )

    async def aclose(self) -> None:
        """Close owned WAL, owned stream, and optionally registered agents."""
        if self._closed:
            return
        self._closed = True
        with anyio.CancelScope(shield=True):
            if self._close_agents:
                for agent in self._agents.values():
                    await agent.aclose()
            if self._owns_wal:
                await self.wal.aclose()
            if self._owns_stream:
                await self.stream.aclose()

    async def __aenter__(self) -> Self:
        await self.wal.initialize()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
