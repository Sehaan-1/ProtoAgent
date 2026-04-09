"""Tool execution Context supporting pause/resume human input and audit events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import anyio
import structlog
from opentelemetry import trace

from ag2.events.models import HumanInputRequest, HumanMessage
from ag2.events.stream import MemoryStream
from ag2.hitl.hooks import HITLCallable, HITLHook, coerce_hitl_hook

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.hitl")


class HITLError(RuntimeError):
    """Base error for Human-in-the-Loop interactions."""


class HITLNotConfiguredError(HITLError):
    """Raised when a tool requests human input without an active hitl_hook."""


class HITLTimeoutError(HITLError, TimeoutError):
    """Raised when human input does not arrive before timeout and no default applies."""


@dataclass(frozen=True, slots=True)
class Context:
    """Execution context injected into tools for HITL prompts and event auditing."""

    stream: MemoryStream = field(default_factory=MemoryStream)
    hitl_hook: HITLHook | None = None
    source: str | None = None
    run_id: str | None = None
    tool_name: str | None = None
    call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def create(
        cls,
        *,
        stream: MemoryStream | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
        source: str | None = None,
        run_id: str | None = None,
        tool_name: str | None = None,
        call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Context:
        """Construct a Context, normalizing any callable hitl_hook."""
        return cls(
            stream=stream if stream is not None else MemoryStream(),
            hitl_hook=coerce_hitl_hook(hitl_hook),
            source=source,
            run_id=run_id,
            tool_name=tool_name,
            call_id=call_id,
            metadata=dict(metadata) if metadata is not None else {},
        )

    def with_tool(self, *, tool_name: str, call_id: str) -> Context:
        """Return a shallow copy scoped to a specific tool invocation."""
        return replace(self, tool_name=tool_name, call_id=call_id)

    async def input(
        self,
        prompt: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - recorded on HumanInputRequest + default fallback
        choices: Sequence[str] | None = None,
        default: str | None = None,
        raise_on_timeout: bool = False,
    ) -> str:
        """Pause tool execution to request human input, emitting audit events."""
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be > 0")

        audit_meta = dict(self.metadata)
        if self.tool_name is not None:
            audit_meta["tool_name"] = self.tool_name
        if self.call_id is not None:
            audit_meta["call_id"] = self.call_id

        request = HumanInputRequest(
            prompt=prompt,
            timeout=timeout,
            choices=list(choices) if choices is not None else None,
            default=default,
            source=self.source,
            run_id=self.run_id,
            metadata=audit_meta,
        )
        await self.stream.emit(request)

        if self.hitl_hook is None:
            raise HITLNotConfiguredError(
                "Tool requested human input via context.input(), but no hitl_hook is configured"
            )

        with tracer.start_as_current_span(
            "ag2.hitl.input",
            attributes={
                "ag2.hitl.request_id": request.request_id,
                "ag2.tool.name": self.tool_name or "",
            },
        ):
            try:
                if timeout is not None:
                    with anyio.fail_after(timeout):
                        raw_response = await self.hitl_hook.request_input(request)
                else:
                    raw_response = await self.hitl_hook.request_input(request)
            except TimeoutError as exc:
                if default is not None and not raise_on_timeout:
                    timeout_msg = HumanMessage(
                        request_id=request.request_id,
                        content=default,
                        sender="system:timeout_default",
                        timed_out=True,
                        source=self.source,
                        run_id=self.run_id,
                        metadata=audit_meta,
                    )
                    await self.stream.emit(timeout_msg)
                    logger.info(
                        "ag2.hitl.timeout_default",
                        request_id=request.request_id,
                        tool=self.tool_name,
                    )
                    return default
                timeout_detail = f" after {timeout}s" if timeout is not None else ""
                raise HITLTimeoutError(
                    f"Human input request {request.request_id!r} timed out{timeout_detail}"
                ) from exc

        if isinstance(raw_response, HumanMessage):
            response_event = HumanMessage(
                request_id=raw_response.request_id or request.request_id,
                content=raw_response.content,
                sender=raw_response.sender,
                timed_out=raw_response.timed_out,
                source=self.source,
                run_id=self.run_id,
                metadata={**audit_meta, **raw_response.metadata},
            )
        else:
            response_event = HumanMessage(
                request_id=request.request_id,
                content=str(raw_response),
                sender="human",
                source=self.source,
                run_id=self.run_id,
                metadata=audit_meta,
            )

        if choices is not None and response_event.content not in choices:
            raise HITLError(
                f"Invalid human choice {response_event.content!r}; expected one of {list(choices)}"
            )

        await self.stream.emit(response_event)
        return response_event.content

    async def confirm(
        self,
        prompt: str,
        *,
        timeout: float | None = None,  # noqa: ASYNC109 - forwarded to input()
        default: bool | None = None,
    ) -> bool:
        """Request a yes/no confirmation from a human."""
        default_str: str | None = None
        if default is not None:
            default_str = "yes" if default else "no"
        answer = await self.input(
            prompt,
            timeout=timeout,
            choices=["yes", "no"],
            default=default_str,
        )
        return answer == "yes"
