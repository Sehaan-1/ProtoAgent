"""Async in-memory pub/sub event stream backed by anyio memory object streams."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, Self, TypeVar, overload
from uuid import uuid4

import anyio
import structlog
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from ag2.events.models import BaseEvent

logger = structlog.get_logger(__name__)

E_co = TypeVar("E_co", bound=BaseEvent, covariant=True)
E_sub = TypeVar("E_sub", bound=BaseEvent)

EventPredicate = Callable[[BaseEvent], bool]
OverflowPolicy = Literal["block", "drop_newest"]


class StreamClosedError(RuntimeError):
    """Raised when publishing to or subscribing from a closed MemoryStream."""


@dataclass(slots=True)
class _SubscriberEntry:
    subscriber_id: str
    predicate: EventPredicate
    send_stream: MemoryObjectSendStream[Any]


class Subscription(Generic[E_co]):
    """Async iterator and context manager yielding filtered events from MemoryStream."""

    def __init__(
        self,
        stream: MemoryStream,
        subscriber_id: str,
        receive_stream: MemoryObjectReceiveStream[E_co],
    ) -> None:
        self._stream = stream
        self._subscriber_id = subscriber_id
        self._receive_stream = receive_stream
        self._closed = False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> E_co:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._receive_stream.receive()
        except anyio.EndOfStream as exc:
            await self.aclose()
            raise StopAsyncIteration from exc

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the subscription and unregister it from the parent stream."""
        if self._closed:
            return
        self._closed = True
        self._stream._unregister(self._subscriber_id)
        with anyio.CancelScope(shield=True):
            await self._receive_stream.aclose()


def _build_predicate(
    matcher: type[BaseEvent] | str | EventPredicate | None = None,
    *,
    event_type: str | None = None,
    source: str | None = None,
    run_id: str | None = None,
) -> EventPredicate:
    checks: list[EventPredicate] = []

    if matcher is not None:
        if isinstance(matcher, type):
            if not issubclass(matcher, BaseEvent):
                raise TypeError("matcher class must be a BaseEvent subclass")
            target_cls = matcher

            def _match_cls(ev: BaseEvent) -> bool:
                return isinstance(ev, target_cls)

            checks.append(_match_cls)
        elif isinstance(matcher, str):
            target_type = matcher

            def _match_type(ev: BaseEvent) -> bool:
                return ev.event_type == target_type

            checks.append(_match_type)
        elif callable(matcher):
            checks.append(matcher)
        else:
            raise TypeError(
                "matcher must be a BaseEvent subclass, event_type string, or callable predicate"
            )

    if event_type is not None:
        expected_type = event_type

        def _match_kw_type(ev: BaseEvent) -> bool:
            return ev.event_type == expected_type

        checks.append(_match_kw_type)
    if source is not None:
        expected_source = source

        def _match_source(ev: BaseEvent) -> bool:
            return ev.source == expected_source

        checks.append(_match_source)
    if run_id is not None:
        expected_run_id = run_id

        def _match_run_id(ev: BaseEvent) -> bool:
            return ev.run_id == expected_run_id

        checks.append(_match_run_id)

    if not checks:
        return lambda _ev: True

    def _combined(event: BaseEvent) -> bool:
        return all(check(event) for check in checks)

    return _combined


class EventFilter(Generic[E_co]):
    """Composable filter view over a MemoryStream supporting chained .where() calls."""

    def __init__(self, stream: MemoryStream, predicate: EventPredicate) -> None:
        self._stream = stream
        self._predicate = predicate

    @overload
    def where(
        self,
        matcher: type[E_sub],
        /,
        *,
        event_type: str | None = None,
        source: str | None = None,
        run_id: str | None = None,
    ) -> EventFilter[E_sub]: ...

    @overload
    def where(
        self,
        matcher: str | EventPredicate | None = None,
        /,
        *,
        event_type: str | None = None,
        source: str | None = None,
        run_id: str | None = None,
    ) -> EventFilter[E_co]: ...

    def where(
        self,
        matcher: type[BaseEvent] | str | EventPredicate | None = None,
        /,
        *,
        event_type: str | None = None,
        source: str | None = None,
        run_id: str | None = None,
    ) -> EventFilter[Any]:
        """Return a narrowed EventFilter combining this filter with additional criteria."""
        next_predicate = _build_predicate(
            matcher,
            event_type=event_type,
            source=source,
            run_id=run_id,
        )
        prev_predicate = self._predicate
        return EventFilter(
            self._stream,
            lambda ev: prev_predicate(ev) and next_predicate(ev),
        )

    def subscribe(
        self,
        *,
        max_buffer_size: int = 256,
        replay: bool = False,
    ) -> Subscription[E_co]:
        """Subscribe to events matching this filter."""
        return self._stream._subscribe(
            predicate=self._predicate,
            max_buffer_size=max_buffer_size,
            replay=replay,
        )


class MemoryStream:
    """Async in-memory pub/sub event bus with typed filtering and bounded replay."""

    def __init__(
        self,
        *,
        history_limit: int = 0,
        overflow_policy: OverflowPolicy = "block",
    ) -> None:
        if history_limit < 0:
            raise ValueError("history_limit must be >= 0")
        self._history_limit = history_limit
        self._overflow_policy: OverflowPolicy = overflow_policy
        self._history: deque[BaseEvent] = deque(maxlen=history_limit if history_limit > 0 else 0)
        self._subscribers: dict[str, _SubscriberEntry] = {}
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether the stream has been closed."""
        return self._closed

    @property
    def history(self) -> tuple[BaseEvent, ...]:
        """Snapshot of retained events when ``history_limit > 0``."""
        return tuple(self._history)

    @property
    def subscriber_count(self) -> int:
        """Number of currently active subscribers."""
        return len(self._subscribers)

    @overload
    def where(
        self,
        matcher: type[E_sub],
        /,
        *,
        event_type: str | None = None,
        source: str | None = None,
        run_id: str | None = None,
    ) -> EventFilter[E_sub]: ...

    @overload
    def where(
        self,
        matcher: str | EventPredicate | None = None,
        /,
        *,
        event_type: str | None = None,
        source: str | None = None,
        run_id: str | None = None,
    ) -> EventFilter[BaseEvent]: ...

    def where(
        self,
        matcher: type[BaseEvent] | str | EventPredicate | None = None,
        /,
        *,
        event_type: str | None = None,
        source: str | None = None,
        run_id: str | None = None,
    ) -> EventFilter[Any]:
        """Create a filtered view over events published to this stream."""
        predicate = _build_predicate(
            matcher,
            event_type=event_type,
            source=source,
            run_id=run_id,
        )
        return EventFilter(self, predicate)

    def subscribe(
        self,
        *,
        max_buffer_size: int = 256,
        replay: bool = False,
    ) -> Subscription[BaseEvent]:
        """Subscribe to all events published to this stream."""
        return self._subscribe(
            predicate=lambda _ev: True,
            max_buffer_size=max_buffer_size,
            replay=replay,
        )

    def _subscribe(
        self,
        *,
        predicate: EventPredicate,
        max_buffer_size: int,
        replay: bool,
    ) -> Subscription[Any]:
        if self._closed:
            raise StreamClosedError("Cannot subscribe to a closed MemoryStream")
        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be >= 1")

        replay_items = [ev for ev in self._history if predicate(ev)] if replay else []
        effective_buffer = max(max_buffer_size, len(replay_items))
        send_stream, receive_stream = anyio.create_memory_object_stream[BaseEvent](effective_buffer)

        for item in replay_items:
            send_stream.send_nowait(item)

        subscriber_id = uuid4().hex
        self._subscribers[subscriber_id] = _SubscriberEntry(
            subscriber_id=subscriber_id,
            predicate=predicate,
            send_stream=send_stream,
        )
        return Subscription(self, subscriber_id, receive_stream)

    def _unregister(self, subscriber_id: str) -> None:
        entry = self._subscribers.pop(subscriber_id, None)
        if entry is not None:
            entry.send_stream.close()

    async def publish(self, event: BaseEvent) -> None:
        """Publish an event to all matching subscribers."""
        if self._closed:
            raise StreamClosedError("Cannot publish to a closed MemoryStream")

        if self._history_limit > 0:
            self._history.append(event)

        # Snapshot entries so unregistrations during await do not mutate while iterating.
        entries = list(self._subscribers.values())
        for entry in entries:
            if not entry.predicate(event):
                continue
            try:
                if self._overflow_policy == "drop_newest":
                    try:
                        entry.send_stream.send_nowait(event)
                    except anyio.WouldBlock:
                        logger.warning(
                            "ag2.events.dropped",
                            subscriber_id=entry.subscriber_id,
                            event_type=event.event_type,
                        )
                else:
                    await entry.send_stream.send(event)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                self._unregister(entry.subscriber_id)

    async def emit(self, event: BaseEvent) -> None:
        """Alias for ``publish(event)``."""
        await self.publish(event)

    async def aclose(self) -> None:
        """Close the stream and terminate all active subscriptions."""
        if self._closed:
            return
        self._closed = True
        entries = list(self._subscribers.values())
        self._subscribers.clear()
        with anyio.CancelScope(shield=True):
            for entry in entries:
                await entry.send_stream.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
