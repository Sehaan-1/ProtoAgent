"""Pluggable Human-in-the-Loop (HITL) hook protocols and adapters."""

from __future__ import annotations

import builtins
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, Self, runtime_checkable

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from ag2.events.models import HumanInputRequest, HumanMessage


@runtime_checkable
class HITLHook(Protocol):
    """Async protocol implemented by Human-in-the-Loop input handlers."""

    async def request_input(self, request: HumanInputRequest) -> str | HumanMessage:
        """Fulfill a human input request and return the response string or HumanMessage."""
        ...


HITLCallable = Callable[[HumanInputRequest], str | HumanMessage | Awaitable[str | HumanMessage]]


class CallbackHITLHook:
    """Adapter wrapping a sync or async callable as an HITLHook."""

    def __init__(self, callback: HITLCallable) -> None:
        if not callable(callback):
            raise TypeError("HITL callback must be callable")
        self._callback = callback
        is_async_fn = inspect.iscoroutinefunction(callback)
        is_async_obj = not inspect.isfunction(callback) and inspect.iscoroutinefunction(
            type(callback).__call__
        )
        self._is_async = is_async_fn or is_async_obj

    async def request_input(self, request: HumanInputRequest) -> str | HumanMessage:
        if self._is_async:
            result = self._callback(request)
            assert inspect.isawaitable(result)
            return await result
        result = await anyio.to_thread.run_sync(self._callback, request)
        if inspect.isawaitable(result):
            return await result
        return result


@dataclass(slots=True)
class _PendingPrompt:
    event: anyio.Event
    response: HumanMessage | None = None


class QueueHITLHook:
    """Async queue-backed HITLHook for UIs, web backends, and concurrent tests."""

    def __init__(self, *, max_buffer_size: int = 128) -> None:
        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be >= 1")
        send_stream, receive_stream = anyio.create_memory_object_stream[HumanInputRequest](
            max_buffer_size
        )
        self._send_stream: MemoryObjectSendStream[HumanInputRequest] = send_stream
        self._receive_stream: MemoryObjectReceiveStream[HumanInputRequest] = receive_stream
        self._pending: dict[str, _PendingPrompt] = {}

    @property
    def pending_count(self) -> int:
        """Number of requests currently awaiting human input."""
        return len(self._pending)

    async def next_request(self) -> HumanInputRequest:
        """Await the next HumanInputRequest enqueued by a tool."""
        return await self._receive_stream.receive()

    async def submit(
        self,
        request_id: str,
        content: str,
        *,
        sender: str = "human",
    ) -> None:
        """Deliver a human response for a pending ``request_id``."""
        pending = self._pending.get(request_id)
        if pending is None:
            raise KeyError(f"No pending HITL request with request_id={request_id!r}")
        pending.response = HumanMessage(
            request_id=request_id,
            content=content,
            sender=sender,
        )
        pending.event.set()

    async def request_input(self, request: HumanInputRequest) -> HumanMessage:
        """Enqueue ``request`` and suspend until ``submit()`` resolves it."""
        waiter = _PendingPrompt(event=anyio.Event())
        self._pending[request.request_id] = waiter
        try:
            await self._send_stream.send(request)
            await waiter.event.wait()
            assert waiter.response is not None
            return waiter.response
        finally:
            self._pending.pop(request.request_id, None)

    async def aclose(self) -> None:
        """Close the internal request queue streams."""
        with anyio.CancelScope(shield=True):
            await self._send_stream.aclose()
            await self._receive_stream.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()


class CLIHITLHook:
    """Interactive terminal prompt HITLHook offloaded to an anyio worker thread."""

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = builtins.input,
        sender: str = "cli-user",
    ) -> None:
        self._input_fn = input_fn
        self._sender = sender

    async def request_input(self, request: HumanInputRequest) -> HumanMessage:
        prompt_parts = [request.prompt]
        if request.choices:
            prompt_parts.append(f"choices={list(request.choices)}")
        if request.default is not None:
            prompt_parts.append(f"default={request.default!r}")
        rendered = " ".join(prompt_parts) + ": "

        raw = await anyio.to_thread.run_sync(
            self._input_fn,
            rendered,
            abandon_on_cancel=True,
        )
        content = raw if raw or request.default is None else request.default
        return HumanMessage(
            request_id=request.request_id,
            content=content,
            sender=self._sender,
        )


def coerce_hitl_hook(hook: HITLHook | HITLCallable | None) -> HITLHook | None:
    """Normalize an HITLHook instance or callable into an HITLHook."""
    if hook is None:
        return None
    if isinstance(hook, HITLHook):
        return hook
    if callable(hook):
        return CallbackHITLHook(hook)
    raise TypeError("hitl_hook must implement HITLHook or be callable")
