"""End-to-end middleware tests and dependency isolation checks for Slice 8 harness layers."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path

import pytest

import ag2.compact
import ag2.knowledge
import ag2.middleware
from ag2.agent import Agent
from ag2.compact import CompactTrigger, SummarizeCompact
from ag2.events import ChatMessage, CompactionCompleted, MemoryStream
from ag2.knowledge import DiskKnowledgeStore, KnowledgePlugin
from ag2.middleware import AssemblerMiddleware, WorkingMemoryPolicy
from ag2.providers import ModelResponse, ResponseFormat, ToolSpec


class _RecordingAgentClient:
    def __init__(self, reply_text: str) -> None:
        self._reply_text = reply_text
        self.recorded_batches: list[list[ChatMessage]] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        del tools, response_format
        self.recorded_batches.append(list(messages))
        return ModelResponse(
            message=ChatMessage(role="assistant", content=self._reply_text),
            model="main-model",
        )

    async def aclose(self) -> None:
        return None


class _DigestClient:
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        del messages, tools, response_format
        return ModelResponse(
            message=ChatMessage(role="assistant", content="Earlier turns discussed auth tokens."),
            model="digest-model",
        )

    async def aclose(self) -> None:
        return None


@pytest.mark.e2e
async def test_e2e_agent_with_knowledge_working_memory_and_compaction(tmp_path: Path) -> None:
    kb_file = tmp_path / "kb.json"
    async with DiskKnowledgeStore(kb_file) as store, MemoryStream(history_limit=50) as stream:
        kb = KnowledgePlugin(store)
        await kb.remember(
            "AG2 uses MemoryStream pub/sub for typed event auditing.",
            title="Event Bus",
            tags=("events",),
        )

        policy = WorkingMemoryPolicy(
            trigger=CompactTrigger(max_messages=4, preserve_recent=2, min_messages_to_compact=2),
            compactor=SummarizeCompact(_DigestClient(), preserve_recent=2),
        )
        assembler = AssemblerMiddleware(
            knowledge=kb,
            working_memory=policy,
            system_prefix="Global policy: cite verified facts.",
        )

        client = _RecordingAgentClient("Event auditing uses MemoryStream.")
        async with Agent(
            name="harness-agent",
            model=client,
            tools=kb.tools(),
            stream=stream,
            middleware=[assembler],
        ) as agent:
            prior = [
                ChatMessage(role="user", content="msg 1"),
                ChatMessage(role="assistant", content="ans 1"),
                ChatMessage(role="user", content="msg 2"),
                ChatMessage(role="assistant", content="ans 2"),
            ]
            reply = await agent.ask(
                "How does MemoryStream event auditing work?",
                messages=prior,
                run_id="run-harness-1",
            )

        assert reply.text == "Event auditing uses MemoryStream."
        # Raw returned history stays complete while model-bound batch was compacted + enriched.
        assert len(reply.history) == 6
        sent = client.recorded_batches[0]
        contents = [m.content or "" for m in sent]
        assert contents[0] == "Global policy: cite verified facts."
        assert any("Earlier turns discussed auth tokens." in c for c in contents)
        assert any("Relevant Knowledge:" in c and "Event Bus" in c for c in contents)
        assert contents[-1] == "How does MemoryStream event auditing work?"

        compacted_events = [ev for ev in stream.history if isinstance(ev, CompactionCompleted)]
        assert len(compacted_events) == 1
        assert compacted_events[0].run_id == "run-harness-1"


def test_harness_layer_dependency_boundaries() -> None:
    forbidden_map = {
        ag2.knowledge: ("ag2.agent", "ag2.mcp", "ag2.network", "ag2.middleware", "ag2.compact"),
        ag2.compact: ("ag2.agent", "ag2.mcp", "ag2.network", "ag2.middleware", "ag2.knowledge"),
        ag2.middleware: ("ag2.agent", "ag2.mcp", "ag2.network"),
    }
    for pkg, forbidden in forbidden_map.items():
        assert pkg.__file__ is not None
        pkg_dir = Path(pkg.__file__).parent
        for py_file in pkg_dir.glob("*.py"):
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
