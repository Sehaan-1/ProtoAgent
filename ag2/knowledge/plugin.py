"""KnowledgePlugin exposing KnowledgeStore retrieval tools and context formatting."""

from __future__ import annotations

from collections.abc import Sequence

from ag2.knowledge.stores import (
    KnowledgeDocument,
    KnowledgeSearchResult,
    KnowledgeStore,
)
from ag2.tools.tool import Tool, tool


class KnowledgePlugin:
    """Bridge between a :class:`KnowledgeStore`, Agent tools, and middleware context."""

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        default_limit: int = 4,
        min_score: float = 0.05,
    ) -> None:
        if default_limit < 1:
            raise ValueError("default_limit must be >= 1")
        if min_score < 0.0:
            raise ValueError("min_score must be >= 0.0")
        self.store = store
        self.default_limit = default_limit
        self.min_score = min_score

    async def remember(
        self,
        content: str,
        *,
        title: str | None = None,
        tags: Sequence[str] = (),
        doc_id: str | None = None,
    ) -> KnowledgeDocument:
        """Persist a new or updated knowledge note in the backing store."""
        kwargs: dict[str, object] = {
            "content": content,
            "title": title,
            "tags": tuple(tags),
        }
        if doc_id is not None:
            kwargs["doc_id"] = doc_id
        doc = KnowledgeDocument.model_validate(kwargs)
        return await self.store.upsert(doc)

    async def query(
        self,
        query_text: str,
        *,
        limit: int | None = None,
        tags: Sequence[str] | None = None,
    ) -> list[KnowledgeSearchResult]:
        """Search the backing store and filter results below ``min_score``."""
        results = await self.store.search(
            query_text,
            limit=limit if limit is not None else self.default_limit,
            tags=tags,
        )
        return [item for item in results if item.score >= self.min_score]

    async def format_context(
        self,
        query_text: str,
        *,
        limit: int | None = None,
    ) -> str | None:
        """Format relevant knowledge hits into an injection block or return ``None``."""
        results = await self.query(query_text, limit=limit)
        if not results:
            return None
        lines = ["Relevant Knowledge:"]
        for item in results:
            doc = item.document
            header = f"- [{doc.title}]" if doc.title else f"- [{doc.doc_id}]"
            lines.append(f"{header} (score={item.score:.2f}): {doc.content}")
        return "\n".join(lines)

    def tools(self, *, read_only: bool = False) -> tuple[Tool, ...]:
        """Build native ``@tool`` callables for searching (and optionally writing) knowledge."""
        plugin = self

        @tool(
            name="knowledge_search",
            description="Search the knowledge base for facts relevant to a query.",
        )
        async def knowledge_search(query: str, limit: int = 4) -> list[dict[str, object]]:
            hits = await plugin.query(query, limit=limit)
            return [
                {
                    "doc_id": h.document.doc_id,
                    "title": h.document.title,
                    "content": h.document.content,
                    "tags": list(h.document.tags),
                    "score": h.score,
                }
                for h in hits
            ]

        if read_only:
            return (knowledge_search,)

        @tool(
            name="knowledge_save",
            description="Save a durable fact or note into the knowledge base.",
        )
        async def knowledge_save(
            content: str,
            title: str | None = None,
            tags: list[str] | None = None,
        ) -> str:
            saved = await plugin.remember(
                content,
                title=title,
                tags=tuple(tags) if tags is not None else (),
            )
            return f"Saved document {saved.doc_id}"

        return (knowledge_search, knowledge_save)
