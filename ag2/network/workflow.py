"""Declarative multi-agent workflow channel backed by a conditional TransitionGraph."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Final, Self
from uuid import uuid4

import structlog
from opentelemetry import trace

from ag2.agent.agent import AgentReply
from ag2.events.messages import Usage
from ag2.network.channels import ChannelResult, ChannelTurn, _add_usage, _run_and_log_turn
from ag2.network.hub import Hub

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.network.workflow")

END: Final[str] = "__end__"

TransitionCondition = Callable[["WorkflowState"], bool | Awaitable[bool]]
PromptTransform = Callable[["WorkflowState"], str | Awaitable[str]]


class WorkflowError(RuntimeError):
    """Base error raised for invalid TransitionGraph definitions or execution failures."""


class MaxWorkflowStepsError(WorkflowError):
    """Raised when a workflow exceeds its configured ``max_steps`` budget."""


@dataclass(frozen=True, slots=True)
class WorkflowState:
    """Immutable snapshot evaluated when selecting the next TransitionGraph edge."""

    current_agent: str
    last_reply: AgentReply[Any]
    last_text: str
    history: tuple[ChannelTurn, ...]
    step: int
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Transition:
    """Directed edge in a :class:`TransitionGraph`."""

    source: str
    target: str | None = END
    condition: TransitionCondition | None = None
    prompt_fn: PromptTransform | None = None
    priority: int = 0


async def _eval_condition(fn: TransitionCondition | None, state: WorkflowState) -> bool:
    if fn is None:
        return True
    result = fn(state)
    if inspect.isawaitable(result):
        return bool(await result)
    return bool(result)


async def _eval_prompt(fn: PromptTransform | None, state: WorkflowState) -> str:
    if fn is None:
        return state.last_text
    result = fn(state)
    if inspect.isawaitable(result):
        return str(await result)
    return str(result)


class TransitionGraph:
    """Declarative conditional state-machine graph governing agent handoffs."""

    def __init__(self, *, entry: str | None = None) -> None:
        self._entry: str | None = entry.strip() if entry else None
        self._edges: dict[str, list[Transition]] = {}
        self._terminals: set[str] = set()

    @property
    def entry(self) -> str | None:
        """Configured starting agent name for workflow execution."""
        return self._entry

    @property
    def terminals(self) -> frozenset[str]:
        """Agent names explicitly marked as terminal workflow stops."""
        return frozenset(self._terminals)

    def set_entry(self, agent_name: str) -> Self:
        """Set the entry agent node and return ``self`` for fluent chaining."""
        cleaned = agent_name.strip()
        if not cleaned or cleaned == END:
            raise ValueError("Entry agent name must be a non-empty agent identifier")
        self._entry = cleaned
        return self

    def add_terminal(self, agent_name: str) -> Self:
        """Mark an agent node as terminal once its turn completes."""
        cleaned = agent_name.strip()
        if not cleaned or cleaned == END:
            raise ValueError("Terminal agent name must be a non-empty agent identifier")
        self._terminals.add(cleaned)
        return self

    def add_edge(
        self,
        source: str,
        target: str | None = END,
        *,
        when: TransitionCondition | None = None,
        prompt: PromptTransform | None = None,
        priority: int = 0,
    ) -> Self:
        """Add a directed conditional or unconditional handoff edge."""
        src = source.strip()
        if not src or src == END:
            raise ValueError("Transition source must be a non-empty agent name")
        tgt = END if target is None else target.strip()
        if not tgt:
            raise ValueError("Transition target must not be blank")

        edge = Transition(
            source=src,
            target=tgt,
            condition=when,
            prompt_fn=prompt,
            priority=priority,
        )
        bucket = self._edges.setdefault(src, [])
        bucket.append(edge)
        # Preserve higher priority first while maintaining registration order within ties.
        bucket.sort(key=lambda item: -item.priority)
        return self

    def outgoing(self, source: str) -> tuple[Transition, ...]:
        """Return ordered outgoing transitions registered for ``source``."""
        return tuple(self._edges.get(source, ()))

    def validate(self, known_agents: Iterable[str] | None = None) -> None:
        """Validate graph entry, target references, and non-trivial agent names."""
        if not self._entry:
            raise WorkflowError("TransitionGraph must define an entry agent")

        if known_agents is None:
            return

        known = set(known_agents)
        if self._entry not in known:
            raise WorkflowError(f"Workflow entry agent {self._entry!r} is not registered in Hub")

        for term in self._terminals:
            if term not in known:
                raise WorkflowError(f"Workflow terminal agent {term!r} is not registered in Hub")

        for src, edges in self._edges.items():
            if src not in known:
                raise WorkflowError(f"Transition source agent {src!r} is not registered in Hub")
            for edge in edges:
                if edge.target is not None and edge.target != END and edge.target not in known:
                    raise WorkflowError(
                        f"Transition target agent {edge.target!r} is not registered in Hub"
                    )

    async def next_transition(self, state: WorkflowState) -> Transition | None:
        """Select the highest-priority outgoing transition whose condition matches."""
        if state.current_agent in self._terminals:
            return None
        for edge in self._edges.get(state.current_agent, ()):
            if await _eval_condition(edge.condition, state):
                return edge
        return None


class WorkflowChannel:
    """Execute a :class:`TransitionGraph` across registered :class:`Hub` agents."""

    def __init__(
        self,
        hub: Hub,
        graph: TransitionGraph,
        *,
        name: str = "workflow",
        max_steps: int = 12,
        raise_on_max_steps: bool = True,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        graph.validate(hub.agent_names)
        self.hub = hub
        self.graph = graph
        self.name = name
        self.max_steps = max_steps
        self.raise_on_max_steps = raise_on_max_steps

    async def run(
        self,
        prompt: str,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChannelResult:
        """Drive workflow transitions until END, terminal node, or ``max_steps``."""
        self.graph.validate(self.hub.agent_names)
        entry_agent = self.graph.entry
        assert entry_agent is not None

        effective_run_id = run_id or uuid4().hex
        state_meta = dict(metadata) if metadata is not None else {}
        turns: list[ChannelTurn] = []
        total_usage = Usage()
        current_agent_name = entry_agent
        current_prompt = prompt
        termination = "completed"

        with tracer.start_as_current_span(
            "ag2.network.channel.workflow",
            attributes={"ag2.channel": self.name, "ag2.run_id": effective_run_id},
        ):
            for step in range(self.max_steps):
                agent = self.hub.get(current_agent_name)
                turn = await _run_and_log_turn(
                    self.hub,
                    agent,
                    current_prompt,
                    turn_index=step,
                    run_id=effective_run_id,
                    channel_name=self.name,
                    channel_type="workflow",
                )
                turns.append(turn)
                total_usage = _add_usage(total_usage, turn.usage)

                state = WorkflowState(
                    current_agent=current_agent_name,
                    last_reply=turn.reply,
                    last_text=turn.text,
                    history=tuple(turns),
                    step=step,
                    run_id=effective_run_id,
                    metadata=state_meta,
                )
                chosen = await self.graph.next_transition(state)
                if chosen is None or chosen.target is None or chosen.target == END:
                    await self.hub.append_wal(
                        run_id=effective_run_id,
                        channel=self.name,
                        kind="workflow.completed",
                        sender=current_agent_name,
                        payload={"step": step, "reason": "end"},
                    )
                    termination = "end" if chosen is not None else "completed"
                    break

                next_prompt = await _eval_prompt(chosen.prompt_fn, state)
                await self.hub.append_wal(
                    run_id=effective_run_id,
                    channel=self.name,
                    kind="workflow.handoff",
                    sender=current_agent_name,
                    recipient=chosen.target,
                    payload={"step": step, "next_prompt": next_prompt},
                )
                logger.debug(
                    "ag2.network.workflow.handoff",
                    channel=self.name,
                    run_id=effective_run_id,
                    source=current_agent_name,
                    target=chosen.target,
                    step=step,
                )
                current_agent_name = chosen.target
                current_prompt = next_prompt
            else:
                if self.raise_on_max_steps:
                    raise MaxWorkflowStepsError(
                        f"Workflow {self.name!r} exceeded max_steps={self.max_steps}"
                    )
                termination = "max_steps"

        last = turns[-1]
        return ChannelResult(
            channel=self.name,
            channel_type="workflow",
            run_id=effective_run_id,
            final_text=last.text,
            final_speaker=last.speaker,
            turns=tuple(turns),
            usage=total_usage,
            termination_reason=termination,
        )


async def workflow(
    hub: Hub,
    graph: TransitionGraph,
    prompt: str,
    *,
    name: str = "workflow",
    max_steps: int = 12,
    raise_on_max_steps: bool = True,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ChannelResult:
    """Convenience function executing a :class:`WorkflowChannel`."""
    return await WorkflowChannel(
        hub,
        graph,
        name=name,
        max_steps=max_steps,
        raise_on_max_steps=raise_on_max_steps,
    ).run(prompt, run_id=run_id, metadata=metadata)
