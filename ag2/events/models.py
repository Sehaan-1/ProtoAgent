"""Typed event models and event-type constants for AG2."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ag2.events.messages import ChatMessage

EV_TEXT: Literal["text"] = "text"
EV_MESSAGE: Literal["message"] = "message"
EV_TOOL_CALL: Literal["tool_call"] = "tool_call"
EV_TOOL_RESULT: Literal["tool_result"] = "tool_result"
EV_HUMAN_INPUT_REQUEST: Literal["human_input_request"] = "human_input_request"
EV_HUMAN_MESSAGE: Literal["human_message"] = "human_message"
EV_COMPACTION_STARTED: Literal["compaction_started"] = "compaction_started"
EV_COMPACTION_COMPLETED: Literal["compaction_completed"] = "compaction_completed"
EV_COMPACTION_FAILED: Literal["compaction_failed"] = "compaction_failed"
EV_ERROR: Literal["error"] = "error"


class BaseEvent(BaseModel):
    """Base immutable event emitted onto an AG2 event stream."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextEvent(BaseEvent):
    """Text output or streaming token delta produced by an agent or model."""

    event_type: Literal["text"] = EV_TEXT
    text: str
    delta: bool = False
    role: Literal["assistant", "system", "user"] = "assistant"


class MessageEvent(BaseEvent):
    """Structured chat message recorded on the event stream."""

    event_type: Literal["message"] = EV_MESSAGE
    message: ChatMessage


class ToolCallEvent(BaseEvent):
    """Emitted when an agent dispatches a tool call."""

    event_type: Literal["tool_call"] = EV_TOOL_CALL
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultEvent(BaseEvent):
    """Emitted when a tool invocation completes or errors."""

    event_type: Literal["tool_result"] = EV_TOOL_RESULT
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    content: str
    is_error: bool = False


class HumanInputRequest(BaseEvent):
    """Request emitted when execution pauses for human-in-the-loop input."""

    event_type: Literal["human_input_request"] = EV_HUMAN_INPUT_REQUEST
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    prompt: str
    timeout: float | None = Field(default=None, gt=0.0)
    choices: list[str] | None = None
    default: str | None = None


class HumanMessage(BaseEvent):
    """Human response delivered back to a paused tool or conversation."""

    event_type: Literal["human_message"] = EV_HUMAN_MESSAGE
    request_id: str | None = None
    content: str
    sender: str = "human"
    timed_out: bool = False


class CompactionStarted(BaseEvent):
    """Emitted when conversation history compaction begins."""

    event_type: Literal["compaction_started"] = EV_COMPACTION_STARTED
    strategy: str = Field(min_length=1)
    original_message_count: int = Field(ge=0)
    original_token_estimate: int | None = Field(default=None, ge=0)


class CompactionCompleted(BaseEvent):
    """Emitted when conversation history compaction succeeds."""

    event_type: Literal["compaction_completed"] = EV_COMPACTION_COMPLETED
    strategy: str = Field(min_length=1)
    original_message_count: int = Field(ge=0)
    compacted_message_count: int = Field(ge=0)
    summary: str | None = None


class CompactionFailed(BaseEvent):
    """Emitted when conversation history compaction fails."""

    event_type: Literal["compaction_failed"] = EV_COMPACTION_FAILED
    strategy: str = Field(min_length=1)
    error: str


class ErrorEvent(BaseEvent):
    """Structured error emitted on the event stream."""

    event_type: Literal["error"] = EV_ERROR
    error_type: str = Field(min_length=1)
    message: str
    recoverable: bool = False


AnyEvent = Annotated[
    TextEvent
    | MessageEvent
    | ToolCallEvent
    | ToolResultEvent
    | HumanInputRequest
    | HumanMessage
    | CompactionStarted
    | CompactionCompleted
    | CompactionFailed
    | ErrorEvent,
    Field(discriminator="event_type"),
]
