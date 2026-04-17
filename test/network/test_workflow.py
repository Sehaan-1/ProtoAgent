"""Unit and workflow routing tests for TransitionGraph and WorkflowChannel."""

from __future__ import annotations

from collections.abc import Sequence

import anyio
import pytest

from ag2.agent import Agent
from ag2.events import ChatMessage, Usage
from ag2.network import (
    END,
    Hub,
    MaxWorkflowStepsError,
    TransitionGraph,
    WorkflowChannel,
    WorkflowError,
    WorkflowState,
    workflow,
)
from ag2.providers import ModelResponse, ResponseFormat, ToolSpec


class _ScriptedWorkflowClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        del tools, response_format
        self.prompts.append(messages[-1].content or "")
        return ModelResponse(
            message=ChatMessage(role="assistant", content=self._responses.pop(0)),
            usage=Usage(prompt_tokens=12, completion_tokens=6, total_tokens=18),
            model="fake",
        )

    async def aclose(self) -> None:
        return None


async def test_workflow_conditional_retry_loop_async_predicate_and_wal() -> None:
    planner_client = _ScriptedWorkflowClient(["Plan ready"])
    coder_client = _ScriptedWorkflowClient(["buggy code v1", "clean code v2"])
    reviewer_client = _ScriptedWorkflowClient(["NEEDS_FIX: missing test", "SHIP"])

    async def _needs_fix(state: WorkflowState) -> bool:
        await anyio.lowlevel.checkpoint()
        return state.last_text.startswith("NEEDS_FIX:")

    graph = (
        TransitionGraph()
        .set_entry("planner")
        .add_edge("planner", "coder")
        .add_edge("coder", "reviewer")
        .add_edge(
            "reviewer",
            "coder",
            when=_needs_fix,
            prompt=lambda st: f"Fix based on review: {st.last_text}",
            priority=10,
        )
        .add_edge(
            "reviewer",
            END,
            when=lambda st: st.last_text == "SHIP",
            priority=5,
        )
    )

    async with Hub() as hub:
        planner = Agent(name="planner", model=planner_client, stream=hub.stream)
        coder = Agent(name="coder", model=coder_client, stream=hub.stream)
        reviewer = Agent(name="reviewer", model=reviewer_client, stream=hub.stream)
        hub.register(planner)
        hub.register(coder)
        hub.register(reviewer)

        result = await workflow(
            hub,
            graph,
            "Build feature X",
            name="dev-pipeline",
            run_id="run-wf-1",
        )

        assert result.channel_type == "workflow"
        assert result.termination_reason == "end"
        assert result.final_speaker == "reviewer"
        assert result.final_text == "SHIP"
        assert [t.speaker for t in result.turns] == [
            "planner",
            "coder",
            "reviewer",
            "coder",
            "reviewer",
        ]
        assert result.usage.total_tokens == 90
        assert coder_client.prompts[1] == "Fix based on review: NEEDS_FIX: missing test"

        handoffs = await hub.read_wal(run_id="run-wf-1", kind="workflow.handoff")
        assert [(e.sender, e.recipient) for e in handoffs] == [
            ("planner", "coder"),
            ("coder", "reviewer"),
            ("reviewer", "coder"),
            ("coder", "reviewer"),
        ]


async def test_workflow_validation_terminal_node_and_max_steps_guard() -> None:
    a_client = _ScriptedWorkflowClient(["loop1", "loop2", "loop3"])
    b_client = _ScriptedWorkflowClient(["pong1", "pong2"])

    async with Hub() as hub:
        hub.register(Agent(name="a", model=a_client))
        hub.register(Agent(name="b", model=b_client))

        no_entry = TransitionGraph().add_edge("a", "b")
        with pytest.raises(WorkflowError, match="entry agent"):
            WorkflowChannel(hub, no_entry)

        unknown_target = TransitionGraph(entry="a").add_edge("a", "ghost")
        with pytest.raises(WorkflowError, match="ghost"):
            WorkflowChannel(hub, unknown_target)

        loop_graph = TransitionGraph(entry="a").add_edge("a", "b").add_edge("b", "a")
        with pytest.raises(MaxWorkflowStepsError, match="max_steps=3"):
            await workflow(hub, loop_graph, "start", max_steps=3)

        # Explicit terminal node overrides outgoing edges and exits cleanly.
        term_graph = TransitionGraph(entry="a").add_edge("a", "b").add_terminal("a")
        res = await workflow(hub, term_graph, "stop early")
        assert res.termination_reason == "completed"
        assert [t.speaker for t in res.turns] == ["a"]
