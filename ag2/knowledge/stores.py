"""Async knowledge document models and memory/disk KnowledgeStore backends."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable
from uuid import uuid4

import aiofiles
import anyio
import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class KnowledgeDocument(BaseModel):
    """Immutable document stored inside an AG2 KnowledgeStore."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    content: str = Field(min_length=1)
    title: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class KnowledgeSearchResult(BaseModel):
    """Scored document returned by a KnowledgeStore search."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document: KnowledgeDocument
    score: float = Field(ge=0.0)


@runtime_checkable
class KnowledgeStore(Protocol):
    """Async storage protocol for agent knowledge persistence and retrieval."""

    async def upsert(self, document: KnowledgeDocument) -> KnowledgeDocument:
        """Insert or update a document and return the persisted copy."""
        ...

    async def get(self, doc_id: str) -> KnowledgeDocument | None:
        """Retrieve a document by ``doc_id`` or return ``None``."""
        ...

    async def delete(self, doc_id: str) -> bool:
        """Delete a document by ``doc_id``, returning whether it existed."""
        ...

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: Sequence[str] | None = None,
    ) -> list[KnowledgeSearchResult]:
        """Search documents matching ``query`` ordered by descending relevance."""
        ...

    async def list_documents(self) -> list[KnowledgeDocument]:
        """Return all stored documents ordered by most recently updated."""
        ...


def score_document(query: str, document: KnowledgeDocument) -> float:
    """Compute a normalized relevance score in ``[0.0, 1.0+]`` for lexical matching."""
    q_clean = query.strip().lower()
    if not q_clean:
        return 0.0

    q_tokens = _tokenize(q_clean)
    if not q_tokens:
        return 0.0

    content_tokens = _tokenize(document.content)
    title_tokens = _tokenize(document.title or "")
    tag_tokens = _tokenize(" ".join(document.tags))

    content_hits = len(q_tokens & content_tokens) / len(q_tokens)
    title_hits = len(q_tokens & title_tokens) / len(q_tokens)
    tag_hits = len(q_tokens & tag_tokens) / len(q_tokens)
    phrase_bonus = 0.25 if q_clean in document.content.lower() else 0.0

    return round(content_hits * 0.6 + title_hits * 0.25 + tag_hits * 0.15 + phrase_bonus, 6)


def _search_in_collection(
    documents: Sequence[KnowledgeDocument],
    query: str,
    *,
    limit: int,
    tags: Sequence[str] | None,
) -> list[KnowledgeSearchResult]:
    if limit < 1:
        raise ValueError("limit must be >= 1")
    required_tags = set(tags) if tags else None

    matches: list[KnowledgeSearchResult] = []
    for doc in documents:
        if required_tags and not (required_tags & set(doc.tags)):
            continue
        score = score_document(query, doc)
        if score > 0.0:
            matches.append(KnowledgeSearchResult(document=doc, score=score))

    matches.sort(key=lambda item: (-item.score, -item.document.updated_at.timestamp()))
    return matches[:limit]


class MemoryKnowledgeStore:
    """Thread/task-safe in-memory KnowledgeStore protected by an anyio Lock."""

    def __init__(self, documents: Sequence[KnowledgeDocument] = ()) -> None:
        self._lock = anyio.Lock()
        self._docs: dict[str, KnowledgeDocument] = {doc.doc_id: doc for doc in documents}

    async def upsert(self, document: KnowledgeDocument) -> KnowledgeDocument:
        """Store or replace ``document`` in memory."""
        async with self._lock:
            self._docs[document.doc_id] = document
        return document

    async def get(self, doc_id: str) -> KnowledgeDocument | None:
        """Fetch a document by ``doc_id``."""
        async with self._lock:
            return self._docs.get(doc_id)

    async def delete(self, doc_id: str) -> bool:
        """Remove a document by ``doc_id``."""
        async with self._lock:
            return self._docs.pop(doc_id, None) is not None

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: Sequence[str] | None = None,
    ) -> list[KnowledgeSearchResult]:
        """Score and rank matching documents in memory."""
        async with self._lock:
            snapshot = list(self._docs.values())
        return _search_in_collection(snapshot, query, limit=limit, tags=tags)

    async def list_documents(self) -> list[KnowledgeDocument]:
        """List stored documents sorted newest first."""
        async with self._lock:
            snapshot = list(self._docs.values())
        snapshot.sort(key=lambda d: d.updated_at.timestamp(), reverse=True)
        return snapshot


class DiskKnowledgeStore:
    """Persistent JSON-backed KnowledgeStore using async ``aiofiles`` I/O."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = anyio.Lock()
        self._loaded = False
        self._docs: dict[str, KnowledgeDocument] = {}

    @property
    def path(self) -> Path:
        """Filesystem path storing the serialized knowledge index."""
        return self._path

    async def initialize(self) -> None:
        """Load existing documents from disk if the store file exists."""
        async with self._lock:
            await self._ensure_loaded_locked()

    async def _ensure_loaded_locked(self) -> None:
        if self._loaded:
            return
        if self._path.exists():
            async with aiofiles.open(self._path, encoding="utf-8") as fh:
                raw = await fh.read()
            if raw.strip():
                items = json.loads(raw)
                self._docs = {
                    doc.doc_id: doc
                    for item in items
                    if (doc := KnowledgeDocument.model_validate(item))
                }
        self._loaded = True

    async def _flush_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_name(f"{self._path.name}.tmp")
        payload = json.dumps(
            [doc.model_dump(mode="json") for doc in self._docs.values()],
            indent=2,
        )
        async with aiofiles.open(tmp_path, mode="w", encoding="utf-8") as fh:
            await fh.write(payload)
        await anyio.Path(tmp_path).replace(self._path)

    async def upsert(self, document: KnowledgeDocument) -> KnowledgeDocument:
        """Persist ``document`` atomically to disk."""
        async with self._lock:
            await self._ensure_loaded_locked()
            self._docs[document.doc_id] = document
            await self._flush_locked()
        logger.debug("ag2.knowledge.disk.upserted", doc_id=document.doc_id)
        return document

    async def get(self, doc_id: str) -> KnowledgeDocument | None:
        """Retrieve a persisted document by ``doc_id``."""
        async with self._lock:
            await self._ensure_loaded_locked()
            return self._docs.get(doc_id)

    async def delete(self, doc_id: str) -> bool:
        """Delete a document and flush the updated store to disk."""
        async with self._lock:
            await self._ensure_loaded_locked()
            existed = self._docs.pop(doc_id, None) is not None
            if existed:
                await self._flush_locked()
            return existed

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: Sequence[str] | None = None,
    ) -> list[KnowledgeSearchResult]:
        """Load if needed and search persisted documents."""
        async with self._lock:
            await self._ensure_loaded_locked()
            snapshot = list(self._docs.values())
        return _search_in_collection(snapshot, query, limit=limit, tags=tags)

    async def list_documents(self) -> list[KnowledgeDocument]:
        """List all persisted documents sorted newest first."""
        async with self._lock:
            await self._ensure_loaded_locked()
            snapshot = list(self._docs.values())
        snapshot.sort(key=lambda d: d.updated_at.timestamp(), reverse=True)
        return snapshot

    async def aclose(self) -> None:
        """Ensure pending state is initialized; DiskKnowledgeStore holds no open handle."""
        return None

    async def __aenter__(self) -> Self:
        await self.initialize()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
