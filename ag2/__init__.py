"""AG2 modern async multi-agent framework."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from ag2._version import __version__

if TYPE_CHECKING:
    from ag2.agent import Agent, AgentReply
    from ag2.config import (
        AnthropicConfig,
        AnyModelConfig,
        GeminiConfig,
        ModelConfig,
        OpenAICompletionsConfig,
        resolve_api_key,
    )
    from ag2.events import (
        EV_COMPACTION_COMPLETED,
        EV_COMPACTION_FAILED,
        EV_COMPACTION_STARTED,
        EV_ERROR,
        EV_HUMAN_INPUT_REQUEST,
        EV_HUMAN_MESSAGE,
        EV_MESSAGE,
        EV_TEXT,
        EV_TOOL_CALL,
        EV_TOOL_RESULT,
        AnyEvent,
        BaseEvent,
        ChatMessage,
        CompactionCompleted,
        CompactionFailed,
        CompactionStarted,
        ErrorEvent,
        HumanInputRequest,
        HumanMessage,
        MemoryStream,
        MessageEvent,
        StreamClosedError,
        TextEvent,
        ToolCall,
        ToolCallEvent,
        ToolResultEvent,
        Usage,
    )
    from ag2.providers import (
        ModelClient,
        ModelResponse,
        OpenAICompletionsClient,
        ProviderError,
        ResponseFormat,
        RetryableProviderError,
        StructuredOutputError,
        ToolSpec,
        create_model_client,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Agent": ("ag2.agent", "Agent"),
    "AgentReply": ("ag2.agent", "AgentReply"),
    "AnthropicConfig": ("ag2.config", "AnthropicConfig"),
    "AnyModelConfig": ("ag2.config", "AnyModelConfig"),
    "GeminiConfig": ("ag2.config", "GeminiConfig"),
    "ModelConfig": ("ag2.config", "ModelConfig"),
    "OpenAICompletionsConfig": ("ag2.config", "OpenAICompletionsConfig"),
    "resolve_api_key": ("ag2.config", "resolve_api_key"),
    "EV_COMPACTION_COMPLETED": ("ag2.events", "EV_COMPACTION_COMPLETED"),
    "EV_COMPACTION_FAILED": ("ag2.events", "EV_COMPACTION_FAILED"),
    "EV_COMPACTION_STARTED": ("ag2.events", "EV_COMPACTION_STARTED"),
    "EV_ERROR": ("ag2.events", "EV_ERROR"),
    "EV_HUMAN_INPUT_REQUEST": ("ag2.events", "EV_HUMAN_INPUT_REQUEST"),
    "EV_HUMAN_MESSAGE": ("ag2.events", "EV_HUMAN_MESSAGE"),
    "EV_MESSAGE": ("ag2.events", "EV_MESSAGE"),
    "EV_TEXT": ("ag2.events", "EV_TEXT"),
    "EV_TOOL_CALL": ("ag2.events", "EV_TOOL_CALL"),
    "EV_TOOL_RESULT": ("ag2.events", "EV_TOOL_RESULT"),
    "AnyEvent": ("ag2.events", "AnyEvent"),
    "BaseEvent": ("ag2.events", "BaseEvent"),
    "ChatMessage": ("ag2.events", "ChatMessage"),
    "CompactionCompleted": ("ag2.events", "CompactionCompleted"),
    "CompactionFailed": ("ag2.events", "CompactionFailed"),
    "CompactionStarted": ("ag2.events", "CompactionStarted"),
    "ErrorEvent": ("ag2.events", "ErrorEvent"),
    "HumanInputRequest": ("ag2.events", "HumanInputRequest"),
    "HumanMessage": ("ag2.events", "HumanMessage"),
    "MemoryStream": ("ag2.events", "MemoryStream"),
    "MessageEvent": ("ag2.events", "MessageEvent"),
    "StreamClosedError": ("ag2.events", "StreamClosedError"),
    "TextEvent": ("ag2.events", "TextEvent"),
    "ToolCall": ("ag2.events", "ToolCall"),
    "ToolCallEvent": ("ag2.events", "ToolCallEvent"),
    "ToolResultEvent": ("ag2.events", "ToolResultEvent"),
    "Usage": ("ag2.events", "Usage"),
    "ModelClient": ("ag2.providers", "ModelClient"),
    "ModelResponse": ("ag2.providers", "ModelResponse"),
    "OpenAICompletionsClient": ("ag2.providers", "OpenAICompletionsClient"),
    "ProviderError": ("ag2.providers", "ProviderError"),
    "ResponseFormat": ("ag2.providers", "ResponseFormat"),
    "RetryableProviderError": ("ag2.providers", "RetryableProviderError"),
    "StructuredOutputError": ("ag2.providers", "StructuredOutputError"),
    "ToolSpec": ("ag2.providers", "ToolSpec"),
    "create_model_client": ("ag2.providers", "create_model_client"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'ag2' has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "EV_COMPACTION_COMPLETED",
    "EV_COMPACTION_FAILED",
    "EV_COMPACTION_STARTED",
    "EV_ERROR",
    "EV_HUMAN_INPUT_REQUEST",
    "EV_HUMAN_MESSAGE",
    "EV_MESSAGE",
    "EV_TEXT",
    "EV_TOOL_CALL",
    "EV_TOOL_RESULT",
    "Agent",
    "AgentReply",
    "AnthropicConfig",
    "AnyEvent",
    "AnyModelConfig",
    "BaseEvent",
    "ChatMessage",
    "CompactionCompleted",
    "CompactionFailed",
    "CompactionStarted",
    "ErrorEvent",
    "GeminiConfig",
    "HumanInputRequest",
    "HumanMessage",
    "MemoryStream",
    "MessageEvent",
    "ModelClient",
    "ModelConfig",
    "ModelResponse",
    "OpenAICompletionsClient",
    "OpenAICompletionsConfig",
    "ProviderError",
    "ResponseFormat",
    "RetryableProviderError",
    "StreamClosedError",
    "StructuredOutputError",
    "TextEvent",
    "ToolCall",
    "ToolCallEvent",
    "ToolResultEvent",
    "ToolSpec",
    "Usage",
    "__version__",
    "create_model_client",
    "resolve_api_key",
]
