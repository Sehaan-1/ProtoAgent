"""Async SQLite write-ahead log (WAL) for AG2 network audit trails and replay."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

import aiosqlite
import anyio
import structlog
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.network.wal")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS wal_entries (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    run_id TEXT NOT NULL,
    channel TEXT,
    kind TEXT NOT NULL,
    sender TEXT,
    recipient TEXT,
    payload_json TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wal_run_seq ON wal_entries (run_id, seq);
CREATE INDEX IF NOT EXISTS idx_wal_channel_seq ON wal_entries (channel, seq);
"""


class WALEntry(BaseModel):
    """Immutable record persisted in the network write-ahead log."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=1)
    entry_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    channel: str | None = None
    kind: str = Field(min_length=1)
    sender: str | None = None
    recipient: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


class AsyncWAL:
    """Async write-ahead log backed by ``aiosqlite`` with SQLite WAL journaling."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._raw_path = str(path)
        self._is_memory = self._raw_path == ":memory:"
        self._conn: aiosqlite.Connection | None = None
        self._lock = anyio.Lock()
        self._closed = False

    @property
    def path(self) -> str:
        """Database path (or ``':memory:'`` for in-memory WAL)."""
        return self._raw_path

    @property
    def closed(self) -> bool:
        """Whether this WAL has been closed."""
        return self._closed

    async def initialize(self) -> None:
        """Open the SQLite connection (if needed) and ensure the WAL schema exists."""
        async with self._lock:
            await self._ensure_open_locked()

    async def _ensure_open_locked(self) -> aiosqlite.Connection:
        if self._closed:
            raise RuntimeError("AsyncWAL is closed")
        if self._conn is not None:
            return self._conn

        if not self._is_memory:
            Path(self._raw_path).parent.mkdir(parents=True, exist_ok=True)

        conn = await aiosqlite.connect(self._raw_path)
        try:
            conn.row_factory = aiosqlite.Row
            if not self._is_memory:
                await conn.execute("PRAGMA journal_mode=WAL;")
                await conn.execute("PRAGMA synchronous=NORMAL;")
            await conn.executescript(_SCHEMA_SQL)
            await conn.commit()
        except Exception:
            await conn.close()
            raise

        self._conn = conn
        logger.debug("ag2.network.wal.opened", path=self._raw_path)
        return conn

    async def append(
        self,
        *,
        run_id: str,
        kind: str,
        channel: str | None = None,
        sender: str | None = None,
        recipient: str | None = None,
        payload: dict[str, Any] | None = None,
        entry_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> WALEntry:
        """Append an immutable entry to the WAL and return it with assigned ``seq``."""
        if not run_id:
            raise ValueError("run_id must not be empty")
        if not kind:
            raise ValueError("kind must not be empty")

        actual_entry_id = entry_id or uuid4().hex
        actual_ts = timestamp or datetime.now(UTC)
        actual_payload = dict(payload) if payload is not None else {}
        payload_json = json.dumps(actual_payload, default=str)

        with tracer.start_as_current_span(
            "ag2.network.wal.append",
            attributes={"ag2.run_id": run_id, "ag2.wal.kind": kind},
        ):
            async with self._lock:
                conn = await self._ensure_open_locked()
                cursor = await conn.execute(
                    """
                    INSERT INTO wal_entries (
                        entry_id, run_id, channel, kind, sender, recipient, payload_json, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        actual_entry_id,
                        run_id,
                        channel,
                        kind,
                        sender,
                        recipient,
                        payload_json,
                        actual_ts.isoformat(),
                    ),
                )
                await conn.commit()
                seq = cursor.lastrowid
                assert seq is not None and seq >= 1

        return WALEntry(
            seq=seq,
            entry_id=actual_entry_id,
            run_id=run_id,
            channel=channel,
            kind=kind,
            sender=sender,
            recipient=recipient,
            payload=actual_payload,
            timestamp=actual_ts,
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
        """Read WAL entries in ascending sequence order with optional filters."""
        if after_seq < 0:
            raise ValueError("after_seq must be >= 0")
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")

        clauses = ["seq > ?"]
        params: list[Any] = [after_seq]
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if channel is not None:
            clauses.append("channel = ?")
            params.append(channel)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)

        sql = (
            "SELECT seq, entry_id, run_id, channel, kind, sender, recipient, "
            "payload_json, timestamp "
            f"FROM wal_entries WHERE {' AND '.join(clauses)} ORDER BY seq ASC"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        with tracer.start_as_current_span("ag2.network.wal.read"):
            async with self._lock:
                conn = await self._ensure_open_locked()
                async with conn.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()

        return [
            WALEntry(
                seq=int(row["seq"]),
                entry_id=str(row["entry_id"]),
                run_id=str(row["run_id"]),
                channel=row["channel"],
                kind=str(row["kind"]),
                sender=row["sender"],
                recipient=row["recipient"],
                payload=json.loads(row["payload_json"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            for row in rows
        ]

    async def journal_mode(self) -> str:
        """Return the active SQLite PRAGMA journal_mode."""
        async with self._lock:
            conn = await self._ensure_open_locked()
            async with conn.execute("PRAGMA journal_mode;") as cursor:
                row = await cursor.fetchone()
                return str(row[0]) if row is not None else "unknown"

    async def aclose(self) -> None:
        """Close the underlying SQLite connection safely."""
        with anyio.CancelScope(shield=True):
            async with self._lock:
                if self._closed:
                    return
                self._closed = True
                if self._conn is not None:
                    await self._conn.close()
                    self._conn = None

    async def __aenter__(self) -> Self:
        await self.initialize()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
