"""Unit and persistence tests for MemoryKnowledgeStore, DiskKnowledgeStore, and KnowledgePlugin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ag2.knowledge import (
    DiskKnowledgeStore,
    KnowledgeDocument,
    KnowledgePlugin,
    MemoryKnowledgeStore,
)


async def test_memory_knowledge_store_crud_tags_and_ranking() -> None:
    store = MemoryKnowledgeStore()
    d1 = KnowledgeDocument(
        doc_id="doc-async",
        title="Async Concurrency",
        content="AG2 uses anyio task groups for structured concurrency.",
        tags=("architecture", "async"),
    )
    d2 = KnowledgeDocument(
        doc_id="doc-mcp",
        title="MCP Adapter",
        content="Connect external servers via MCPSession.",
        tags=("mcp",),
    )
    await store.upsert(d1)
    await store.upsert(d2)

    assert await store.get("doc-async") == d1
    assert len(await store.list_documents()) == 2

    hits = await store.search("structured concurrency anyio", tags=("async",))
    assert len(hits) == 1
    assert hits[0].document.doc_id == "doc-async"
    assert hits[0].score > 0.4

    assert await store.delete("doc-mcp") is True
    assert await store.delete("doc-mcp") is False
    assert await store.get("doc-mcp") is None


async def test_disk_knowledge_store_persists_across_reopen_and_plugin_tools(
    tmp_path: Path,
) -> None:
    store_file = tmp_path / "kb" / "store.json"

    async with DiskKnowledgeStore(store_file) as store:
        plugin = KnowledgePlugin(store)
        tools = {t.name: t for t in plugin.tools()}
        assert set(tools) == {"knowledge_search", "knowledge_save"}

        saved_msg = await tools["knowledge_save"].invoke(
            {
                "title": "WAL Policy",
                "content": "Network Hub logs messages into SQLite WAL mode.",
                "tags": ["network", "wal"],
            }
        )
        assert saved_msg.startswith("Saved document ")

        found = await tools["knowledge_search"].invoke({"query": "SQLite WAL", "limit": 2})
        assert len(found) == 1
        assert found[0]["title"] == "WAL Policy"

        formatted = await plugin.format_context("SQLite WAL")
        assert formatted is not None
        assert "Relevant Knowledge:" in formatted
        assert "WAL Policy" in formatted

    assert store_file.exists()
    raw_items = json.loads(store_file.read_text(encoding="utf-8"))
    assert len(raw_items) == 1

    # Reopen from disk and confirm durability + read_only tools.
    async with DiskKnowledgeStore(store_file) as reopened:
        hits = await reopened.search("SQLite WAL")
        assert len(hits) == 1
        ro_plugin = KnowledgePlugin(reopened)
        assert [t.name for t in ro_plugin.tools(read_only=True)] == ["knowledge_search"]

    with pytest.raises(ValueError, match="default_limit"):
        KnowledgePlugin(MemoryKnowledgeStore(), default_limit=0)
