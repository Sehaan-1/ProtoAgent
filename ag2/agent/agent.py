"""Core async Agent and AgentReply conversation and tool-calling loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
from ag2.providers.base import (
    ModelClient,
    ModelResponse,
    ResponseFormat,
    StructuredOutputError,
    ToolSpec,
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
    ) -> AgentReply[Any]: ...

    async def ask(
        self,
        prompt: str | ChatMessage,
        *,
        output_type: type[Any] | None = None,
        tools: Sequence[Tool] | None = None,
        max_tool_rounds: int | None = None,
        hitl_hook: HITLHook | HITLCallable | None = None,
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
        )


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

        with tracer.start_as_current_span(
            "ag2.agent.ask",
            attributes={"ag2.agent.name": self.name, "ag2.run_id": effective_run_id},
        ):
            while True:
                try:
                    model_response = await self._client.complete(
                        history,
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
