"""Canonical chat message, tool call, and usage schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MessageRole = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A model-initiated tool invocation request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """A single turn message exchanged with a model or agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_role_fields(self) -> ChatMessage:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages must define tool_call_id")
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages may contain tool_calls")
        if self.role in ("system", "user") and self.content is None:
            raise ValueError(f"{self.role} messages must define content")
        return self


class Usage(BaseModel):
    """Token accounting reported by a model call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
