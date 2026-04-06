"""Core async Agent and AgentReply conversation continuation loop."""

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
from ag2.events.messages import ChatMessage, Usage
from ag2.events.models import ErrorEvent, MessageEvent, TextEvent
from ag2.events.stream import MemoryStream
from ag2.providers.base import ModelClient, ResponseFormat, StructuredOutputError, ToolSpec
from ag2.providers.factory import create_model_client

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.agent")

T = TypeVar("T")


def _coerce_user_message(prompt: str | ChatMessage) -> ChatMessage:
    if isinstance(prompt, ChatMessage):
        return prompt
    return ChatMessage(role="user", content=prompt)


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
    ) -> AgentReply[T]: ...

    @overload
    async def ask(
        self,
        prompt: str | ChatMessage,
        *,
        output_type: None = None,
    ) -> AgentReply[Any]: ...

    async def ask(
        self,
        prompt: str | ChatMessage,
        *,
        output_type: type[Any] | None = None,
    ) -> AgentReply[Any]:
        """Continue this exact conversation thread with a follow-up prompt."""
        return await self.agent._run_turn(
            prompt,
            prior_history=self.history,
            output_type=output_type,
            run_id=self.run_id,
            include_system=False,
        )


class Agent:
    """Async protocol-driven conversational agent."""

    def __init__(
        self,
        *,
        name: str = "agent",
        model: ModelConfig | ModelClient,
        system_message: str | None = None,
        stream: MemoryStream | None = None,
        output_type: type[Any] | None = None,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.default_output_type = output_type

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
    ) -> AgentReply[T]: ...

    @overload
    async def ask(
        self,
        prompt: str | ChatMessage,
        *,
        messages: Sequence[ChatMessage] | None = None,
        output_type: None = None,
        run_id: str | None = None,
    ) -> AgentReply[Any]: ...

    async def ask(
        self,
        prompt: str | ChatMessage,
        *,
        messages: Sequence[ChatMessage] | None = None,
        output_type: type[Any] | None = None,
        run_id: str | None = None,
    ) -> AgentReply[Any]:
        """Send a prompt to the agent and await its typed AgentReply."""
        return await self._run_turn(
            prompt,
            prior_history=messages or (),
            output_type=output_type,
            run_id=run_id,
            include_system=True,
        )

    async def _run_turn(
        self,
        prompt: str | ChatMessage,
        *,
        prior_history: Sequence[ChatMessage],
        output_type: type[Any] | None,
        run_id: str | None,
        include_system: bool,
        tools: Sequence[ToolSpec] = (),
    ) -> AgentReply[Any]:
        effective_run_id = run_id or uuid4().hex
        target_type = output_type if output_type is not None else self.default_output_type
        response_format = ResponseFormat.from_type(target_type) if target_type is not None else None

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

        with tracer.start_as_current_span(
            "ag2.agent.ask",
            attributes={"ag2.agent.name": self.name, "ag2.run_id": effective_run_id},
        ):
            try:
                model_response = await self._client.complete(
                    history,
                    tools=tools,
                    response_format=response_format,
                )
            except Exception as exc:
                await self._emit_error(exc, effective_run_id)
                raise

        assistant_msg = model_response.message
        history.append(assistant_msg)

        await self.stream.emit(
            MessageEvent(message=assistant_msg, source=self.name, run_id=effective_run_id)
        )
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
            total_tokens=model_response.usage.total_tokens,
        )

        return AgentReply(
            agent=self,
            text=assistant_text,
            output=cast(Any, parsed_output),
            message=assistant_msg,
            history=tuple(history),
            usage=model_response.usage,
            run_id=effective_run_id,
            model=model_response.model,
            finish_reason=model_response.finish_reason,
        )

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
