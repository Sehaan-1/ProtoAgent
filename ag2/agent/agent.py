"""Core async Agent and AgentReply conversation and tool-calling loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Generic, Self, TypeVar, cast, overload
from uuid import uuid4

import anyio
import structlog
from opentelemetry import trace
from pydantic import TypeAdapter, ValidationError

from ag2.config.models import ModelConfig
from ag2.events.messages import ChatMessage, ToolCall, Usage
from ag2.events.models import (
    ErrorEvent,
    MessageEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ag2.events.stream import MemoryStream
from ag2.hitl.context import Context, HITLNotConfiguredError
from ag2.hitl.hooks import HITLCallable, HITLHook, coerce_hitl_hook
from ag2.middleware.pipeline import MessageMiddleware, TurnContext
from ag2.providers.base import (
    ModelClient,
    ModelResponse,
    ProviderError,
    ResponseFormat,
    StructuredOutputError,
    ToolSpec,
    stream_model_response,
)
from ag2.providers.factory import create_model_client
from ag2.tools.tool import MaxToolRoundsError, Tool, index_tools, serialize_tool_result

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.agent")

T = TypeVar("T")


def _coerce_user_message(prompt: str | ChatMessage) -> ChatMessage:
    if isinstance(prompt, ChatMessage):
        return prompt
    return ChatMessage(role="user", content=prompt)


def _add_usage(acc: Usage, delta: Usage) -> Usage:
    return Usage(
        prompt_tokens=acc.prompt_tokens + delta.prompt_tokens,
        completion_tokens=acc.completion_tokens + delta.completion_tokens,
        total_tokens=acc.total_tokens + delta.total_tokens,
    )


@dataclass(frozen=True, slots=True)
class AgentReply(Generic[T]):
    """Immutable reply from an Agent turn supporting multi-turn continuation."""

    agent: Agent
    text: str
    output: T
    message: ChatMessage
    history: tuple[ChatMessage, ...]
    usage: Usage
    run_id: str
    model: str
    finish_reason: str | None = None

    @overload
    async def ask(
        self,
        prompt: str | ChatMessage,
        *,
        output_type: type[T],
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
        stream: bool = False,
    ) -> AgentReply[T]: ...

    @overload
    async def ask(
        self,
        prompt: str | ChatMessage,
        *,
        output_type: None = None,
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
        stream: bool = False,
    ) -> AgentReply[Any]: ...

    async def ask(
        self,
        prompt: str | ChatMessage,
        *,
        output_type: type[Any] | None = None,
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
        stream: bool = False,
    ) -> AgentReply[Any]:
        """Continue this exact conversation thread with a follow-up prompt."""
        return await self.agent._run_turn(
            prompt,
            prior_history=self.history,
            output_type=output_type,
            run_id=self.run_id,
            include_system=False,
            tools=tools,
            max_tool_rounds=max_tool_rounds,
            hitl_hook=hitl_hook,
            stream=stream,
        )


class AgentStream(Generic[T]):
    """Async context manager yielding live TextEvent deltas and exposing the final reply."""

    def __init__(
        self,
        agent: Agent,
        prompt: str | ChatMessage,
        *,
        messages: Sequence[ChatMessage] | None = None,
        output_type: type[T] | None = None,
        run_id: str | None = None,
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
    ) -> None:
        self._agent = agent
        self._prompt = prompt
        self._messages = messages
        self._output_type = output_type
        self._run_id = run_id or uuid4().hex
        self._tools = tools
        self._max_tool_rounds = max_tool_rounds
        self._hitl_hook = hitl_hook
        self._reply: AgentReply[T] | None = None
        self._error: BaseException | None = None
        self._send_stream: Any = None
        self._recv_stream: Any = None
        self._tg: Any = None

    @property
    def run_id(self) -> str:
        """Run identifier correlating streamed TextEvents and the final AgentReply."""
        return self._run_id

    @property
    def reply(self) -> AgentReply[T]:
        """Return the completed AgentReply after iteration finishes."""
        if self._error is not None:
            raise self._error
        if self._reply is None:
            raise RuntimeError("AgentStream has not finished yet")
        return self._reply

    async def __aenter__(self) -> Self:
        send_stream, recv_stream = anyio.create_memory_object_stream[TextEvent](256)
        self._send_stream = send_stream
        self._recv_stream = recv_stream
        self._tg = anyio.create_task_group()
        await self._tg.__aenter__()

        async def _producer() -> None:
            try:
                self._reply = cast(
                    AgentReply[T],
                    await self._agent._run_turn(
                        self._prompt,
                        prior_history=self._messages or (),
                        output_type=self._output_type,
                        run_id=self._run_id,
                        include_system=True,
                        tools=self._tools,
                        max_tool_rounds=self._max_tool_rounds,
                        hitl_hook=self._hitl_hook,
                        stream=True,
                        _delta_sink=send_stream,
                    ),
                )
            except BaseException as exc:
                if isinstance(exc, anyio.get_cancelled_exc_class()):
                    raise
                self._error = exc
            finally:
                with anyio.CancelScope(shield=True):
                    await send_stream.aclose()

        self._tg.start_soon(_producer)
        return self

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> TextEvent:
        assert self._recv_stream is not None
        try:
            return cast(TextEvent, await self._recv_stream.receive())
        except anyio.EndOfStream as exc:
            if self._error is not None:
                raise self._error from None
            raise StopAsyncIteration from exc

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        try:
            if self._tg is not None:
                self._tg.cancel_scope.cancel()
                try:
                    await self._tg.__aexit__(exc_type, exc_val, exc_tb)
                except BaseExceptionGroup as eg:
                    if (
                        exc_val is not None
                        and len(eg.exceptions) == 1
                        and eg.exceptions[0] is exc_val
                    ):
                        return False
                    raise
        finally:
            if self._recv_stream is not None:
                with anyio.CancelScope(shield=True):
                    await self._recv_stream.aclose()
        return None


class Agent:
    """Async protocol-driven conversational agent with tool-calling and HITL support."""

    def __init__(
        self,
        *,
        name: str = "agent",
        model: ModelConfig | ModelClient,
        system_message: str | None = None,
        stream: MemoryStream | None = None,
        output_type: type[Any] | None = None,
        tools: Sequence[Tool] = (),
        max_tool_rounds: int = 10,
        hitl_hook: HITLHook | HITLCallable | None = None,
        middleware: Sequence[MessageMiddleware] = (),
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be >= 1")

        self.name = name
        self.system_message = system_message
        self.default_output_type = output_type
        self.tools = tuple(tools)
        index_tools(self.tools)
        self.max_tool_rounds = max_tool_rounds
        self.hitl_hook: HITLHook | None = coerce_hitl_hook(hitl_hook)
        self.middleware: tuple[MessageMiddleware, ...] = tuple(middleware)

        if isinstance(model, ModelConfig):
            self._client: ModelClient = create_model_client(model)
            self._owns_client = True
        else:
            self._client = model
            self._owns_client = False

        if stream is None:
            self.stream = MemoryStream()
            self._owns_stream = True
        else:
            self.stream = stream
            self._owns_stream = False

    @overload
    async def ask(
        self,
        prompt: str | ChatMessage,
        *,
        messages: Sequence[ChatMessage] | None = None,
        output_type: type[T],
        run_id: str | None = None,
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
        stream: bool = False,
    ) -> AgentReply[T]: ...

    @overload
    async def ask(
        self,
        prompt: str | ChatMessage,
        *,
        messages: Sequence[ChatMessage] | None = None,
        output_type: None = None,
        run_id: str | None = None,
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
        stream: bool = False,
    ) -> AgentReply[Any]: ...

    async def ask(
        self,
        prompt: str | ChatMessage,
        *,
        messages: Sequence[ChatMessage] | None = None,
        output_type: type[Any] | None = None,
        run_id: str | None = None,
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
        stream: bool = False,
    ) -> AgentReply[Any]:
        """Send a prompt to the agent and await its typed AgentReply."""
        return await self._run_turn(
            prompt,
            prior_history=messages or (),
            output_type=output_type,
            run_id=run_id,
            include_system=True,
            tools=tools,
            max_tool_rounds=max_tool_rounds,
            hitl_hook=hitl_hook,
            stream=stream,
        )

    @overload
    def stream_ask(
        self,
        prompt: str | ChatMessage,
        *,
        messages: Sequence[ChatMessage] | None = None,
        output_type: type[T],
        run_id: str | None = None,
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
    ) -> AgentStream[T]: ...

    @overload
    def stream_ask(
        self,
        prompt: str | ChatMessage,
        *,
        messages: Sequence[ChatMessage] | None = None,
        output_type: None = None,
        run_id: str | None = None,
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
    ) -> AgentStream[Any]: ...

    def stream_ask(
        self,
        prompt: str | ChatMessage,
        *,
        messages: Sequence[ChatMessage] | None = None,
        output_type: type[Any] | None = None,
        run_id: str | None = None,
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
    ) -> AgentStream[Any]:
        """Return an async-iterable AgentStream yielding live TextEvent(delta=True) tokens."""
        return AgentStream(
            self,
            prompt,
            messages=messages,
            output_type=output_type,
            run_id=run_id,
            tools=tools,
            max_tool_rounds=max_tool_rounds,
            hitl_hook=hitl_hook,
        )

    async def _run_turn(
        self,
        prompt: str | ChatMessage,
        *,
        prior_history: Sequence[ChatMessage],
        output_type: type[Any] | None,
        run_id: str | None,
        include_system: bool,
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
        stream: bool = False,
        _delta_sink: Any = None,
    ) -> AgentReply[Any]:
        effective_run_id = run_id or uuid4().hex
        target_type = output_type if output_type is not None else self.default_output_type
        response_format = ResponseFormat.from_type(target_type) if target_type is not None else None

        active_tools = tuple(tools) if tools is not None else self.tools
        tool_map = index_tools(active_tools)
        tool_specs: tuple[ToolSpec, ...] = tuple(item.spec for item in active_tools)

        round_limit = max_tool_rounds if max_tool_rounds is not None else self.max_tool_rounds
        if round_limit < 1:
            raise ValueError("max_tool_rounds must be >= 1")

        active_hook = coerce_hitl_hook(hitl_hook) if hitl_hook is not None else self.hitl_hook
        base_context = Context(
            stream=self.stream,
            hitl_hook=active_hook,
            source=self.name,
            run_id=effective_run_id,
        )

        history: list[ChatMessage] = []
        if include_system and self.system_message is not None:
            has_system = any(m.role == "system" for m in prior_history)
            if not has_system:
                history.append(ChatMessage(role="system", content=self.system_message))

        history.extend(prior_history)
        user_msg = _coerce_user_message(prompt)
        history.append(user_msg)

        await self.stream.emit(
            MessageEvent(message=user_msg, source=self.name, run_id=effective_run_id)
        )

        total_usage = Usage()
        tool_rounds = 0
        model_response: ModelResponse | None = None
        assistant_msg: ChatMessage | None = None
        turn_context = TurnContext(
            prompt=user_msg,
            source=self.name,
            run_id=effective_run_id,
            stream=self.stream,
        )

        with tracer.start_as_current_span(
            "ag2.agent.ask",
            attributes={"ag2.agent.name": self.name, "ag2.run_id": effective_run_id},
        ):
            while True:
                request_messages = list(history)
                for mw in self.middleware:
                    request_messages = await mw.transform(request_messages, turn_context)
                try:
                    if stream:
                        streamed_resp: ModelResponse | None = None
                        async for chunk in stream_model_response(
                            self._client,
                            request_messages,
                            tools=tool_specs,
                            response_format=response_format,
                        ):
                            if chunk.text_delta:
                                delta_ev = TextEvent(
                                    text=chunk.text_delta,
                                    delta=True,
                                    role="assistant",
                                    source=self.name,
                                    run_id=effective_run_id,
                                )
                                await self.stream.emit(delta_ev)
                                if _delta_sink is not None:
                                    await _delta_sink.send(delta_ev)
                            if chunk.response is not None:
                                streamed_resp = chunk.response
                        if streamed_resp is None:
                            raise ProviderError(
                                "Streaming ModelClient ended without yielding final ModelResponse"
                            )
                        model_response = streamed_resp
                    else:
                        model_response = await self._client.complete(
                            request_messages,
                            tools=tool_specs,
                            response_format=response_format,
                        )
                except Exception as exc:
                    await self._emit_error(exc, effective_run_id)
                    raise

                total_usage = _add_usage(total_usage, model_response.usage)
                assistant_msg = model_response.message
                history.append(assistant_msg)
                await self.stream.emit(
                    MessageEvent(message=assistant_msg, source=self.name, run_id=effective_run_id)
                )

                if not assistant_msg.tool_calls:
                    break

                tool_rounds += 1
                if tool_rounds > round_limit:
                    err = MaxToolRoundsError(
                        f"Agent {self.name!r} exceeded max_tool_rounds={round_limit}"
                    )
                    await self._emit_error(err, effective_run_id)
                    raise err

                try:
                    tool_messages = await self._execute_tool_calls(
                        assistant_msg.tool_calls,
                        tool_map=tool_map,
                        context=base_context,
                        run_id=effective_run_id,
                    )
                except Exception as exc:
                    await self._emit_error(exc, effective_run_id)
                    raise

                for tool_msg in tool_messages:
                    history.append(tool_msg)
                    await self.stream.emit(
                        MessageEvent(message=tool_msg, source=self.name, run_id=effective_run_id)
                    )

        assert model_response is not None and assistant_msg is not None
        assistant_text = assistant_msg.content or ""
        if assistant_text:
            await self.stream.emit(
                TextEvent(
                    text=assistant_text,
                    role="assistant",
                    source=self.name,
                    run_id=effective_run_id,
                )
            )

        parsed_output: Any = assistant_text
        if target_type is not None:
            parsed_output = await self._parse_structured_output(
                assistant_text,
                target_type,
                effective_run_id,
            )

        logger.debug(
            "ag2.agent.turn_completed",
            agent=self.name,
            run_id=effective_run_id,
            model=model_response.model,
            tool_rounds=tool_rounds,
            total_tokens=total_usage.total_tokens,
        )

        return AgentReply(
            agent=self,
            text=assistant_text,
            output=cast(Any, parsed_output),
            message=assistant_msg,
            history=tuple(history),
            usage=total_usage,
            run_id=effective_run_id,
            model=model_response.model,
            finish_reason=model_response.finish_reason,
        )

    async def _execute_tool_calls(
        self,
        tool_calls: Sequence[ToolCall],
        *,
        tool_map: dict[str, Tool],
        context: Context,
        run_id: str,
    ) -> list[ChatMessage]:
        results: list[tuple[str, bool] | None] = [None] * len(tool_calls)

        for tc in tool_calls:
            await self.stream.emit(
                ToolCallEvent(
                    call_id=tc.id,
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    source=self.name,
                    run_id=run_id,
                )
            )

        async def _run_one(index: int, tc: ToolCall) -> None:
            tool_impl = tool_map.get(tc.name)
            if tool_impl is None:
                results[index] = (f"Error: unknown tool {tc.name!r}", True)
                return

            tool_context = context.with_tool(tool_name=tc.name, call_id=tc.id)
            try:
                raw_result = await tool_impl.invoke(tc.arguments, context=tool_context)
            except anyio.get_cancelled_exc_class():
                raise
            except HITLNotConfiguredError:
                raise
            except Exception as exc:
                logger.warning(
                    "ag2.tool.execution_failed",
                    agent=self.name,
                    tool=tc.name,
                    call_id=tc.id,
                    error=str(exc),
                )
                results[index] = (f"Error executing tool {tc.name!r}: {exc}", True)
                return

            results[index] = (serialize_tool_result(raw_result), False)

        try:
            async with anyio.create_task_group() as tg:
                for idx, tc in enumerate(tool_calls):
                    tg.start_soon(_run_one, idx, tc)
        except* HITLNotConfiguredError as eg:
            raise eg.exceptions[0] from eg

        tool_messages: list[ChatMessage] = []
        for idx, tc in enumerate(tool_calls):
            pair = results[idx]
            assert pair is not None
            content, is_error = pair
            await self.stream.emit(
                ToolResultEvent(
                    call_id=tc.id,
                    tool_name=tc.name,
                    content=content,
                    is_error=is_error,
                    source=self.name,
                    run_id=run_id,
                )
            )
            tool_messages.append(
                ChatMessage(
                    role="tool",
                    tool_call_id=tc.id,
                    name=tc.name,
                    content=content,
                )
            )
        return tool_messages

    async def _parse_structured_output(
        self,
        raw_text: str,
        target_type: type[Any],
        run_id: str,
    ) -> Any:
        try:
            return TypeAdapter(target_type).validate_json(raw_text)
        except ValidationError as exc:
            err = StructuredOutputError(
                f"Failed to parse structured output into {target_type!r}: {exc}"
            )
            await self._emit_error(err, run_id)
            raise err from exc

    async def _emit_error(self, exc: BaseException, run_id: str) -> None:
        if self.stream.closed:
            return
        with anyio.CancelScope(shield=True):
            await self.stream.emit(
                ErrorEvent(
                    error_type=type(exc).__name__,
                    message=str(exc),
                    source=self.name,
                    run_id=run_id,
                )
            )

    async def aclose(self) -> None:
        """Close owned ModelClient and MemoryStream resources."""
        with anyio.CancelScope(shield=True):
            if self._owns_client:
                await self._client.aclose()
            if self._owns_stream:
                await self.stream.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()
