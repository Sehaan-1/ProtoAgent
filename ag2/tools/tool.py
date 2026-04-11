"""@tool decorator, Pydantic schema extraction, validation, and async invocation."""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable
from typing import Annotated, Any, get_args, get_origin, get_type_hints, overload

import anyio
import structlog
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic.fields import FieldInfo

from ag2.hitl.context import Context
from ag2.providers.base import ToolSpec, enforce_strict_json_schema

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.tools")


class ToolError(RuntimeError):
    """Base error for tool definition, validation, and invocation failures."""


class ToolValidationError(ToolError):
    """Raised when tool arguments fail Pydantic schema validation."""


class ToolExecutionError(ToolError):
    """Raised when a tool callable fails during execution."""


class MaxToolRoundsError(ToolError):
    """Raised when an agent exceeds its configured max_tool_rounds budget."""


def _is_context_annotation(annotation: Any) -> bool:
    if annotation is Context:
        return True
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        return bool(args and args[0] is Context)
    return False


def _build_params_model(
    fn: Callable[..., Any],
    tool_name: str,
) -> tuple[type[BaseModel], str | None]:
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    field_definitions: dict[str, Any] = {}
    context_param: str | None = None

    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ToolError(
                f"Tool {tool_name!r} cannot use *args or **kwargs parameter {param_name!r}"
            )
        fallback_annotation = (
            param.annotation if param.annotation is not inspect.Parameter.empty else Any
        )
        annotation = hints.get(param_name, fallback_annotation)
        if _is_context_annotation(annotation):
            if context_param is not None:
                raise ToolError(f"Tool {tool_name!r} may declare at most one Context parameter")
            context_param = param_name
            continue

        if param.default is inspect.Parameter.empty:
            field_definitions[param_name] = (annotation, ...)
        elif isinstance(param.default, FieldInfo):
            field_definitions[param_name] = (annotation, param.default)
        else:
            field_definitions[param_name] = (annotation, Field(default=param.default))

    model_cls = create_model(
        f"{tool_name}Params",
        __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True),
        **field_definitions,
    )
    return model_cls, context_param


def serialize_tool_result(result: Any) -> str:
    """Serialize a tool return value into a string suitable for a tool ChatMessage."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    if isinstance(result, (dict, list, int, float, bool)):
        return json.dumps(result, default=str)
    return str(result)


class Tool:
    """Executable tool wrapping a sync or async callable with Pydantic validation."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError("tool target must be callable")

        self._fn = fn
        raw_name = name if name is not None else getattr(fn, "__name__", "")
        self.name = str(raw_name).strip()
        if not self.name:
            raise ValueError("Tool name must not be empty")

        raw_doc = inspect.getdoc(fn) or ""
        self.description = description if description is not None else raw_doc
        is_async_fn = inspect.iscoroutinefunction(fn)
        is_async_obj = not inspect.isfunction(fn) and inspect.iscoroutinefunction(type(fn).__call__)
        self.is_async = is_async_fn or is_async_obj

        if parameters is None:
            params_model, context_param = _build_params_model(fn, self.name)
            self._params_model: type[BaseModel] | None = params_model
            self._context_param: str | None = context_param
            self._param_names: tuple[str, ...] = tuple(params_model.model_fields.keys())
            self._required_keys: tuple[str, ...] = ()
            schema = params_model.model_json_schema()
        else:
            self._params_model = None
            self._context_param = None
            self._param_names = ()
            raw_required = parameters.get("required", ())
            self._required_keys = (
                tuple(str(k) for k in raw_required) if isinstance(raw_required, list) else ()
            )
            schema = dict(parameters) if parameters else {"type": "object", "properties": {}}
        self.parameters: dict[str, Any] = enforce_strict_json_schema(schema)

    @classmethod
    def from_schema(
        cls,
        fn: Callable[..., Any],
        *,
        name: str,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> Tool:
        """Construct a Tool from an explicit JSON Schema (e.g. remote MCP tools)."""
        return cls(
            fn,
            name=name,
            description=description,
            parameters=parameters
            if parameters is not None
            else {"type": "object", "properties": {}},
        )

    @property
    def expects_context(self) -> bool:
        """Whether the wrapped tool declares an injected HITL Context parameter."""
        return self._context_param is not None

    @property
    def spec(self) -> ToolSpec:
        """Return the provider-agnostic ToolSpec descriptor for this tool."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate raw tool arguments and return typed keyword arguments."""
        if self._params_model is None:
            if not isinstance(arguments, dict):
                raise ToolValidationError(
                    f"Invalid arguments for tool {self.name!r}: expected JSON object"
                )
            missing = [key for key in self._required_keys if key not in arguments]
            if missing:
                raise ToolValidationError(
                    f"Invalid arguments for tool {self.name!r}: missing required keys {missing}"
                )
            return dict(arguments)
        try:
            validated = self._params_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolValidationError(f"Invalid arguments for tool {self.name!r}: {exc}") from exc
        return {field: getattr(validated, field) for field in self._param_names}

    async def invoke(
        self,
        arguments: dict[str, Any] | None = None,
        *,
        context: Context | None = None,
    ) -> Any:
        """Validate arguments and asynchronously invoke the wrapped function."""
        kwargs = self.validate_arguments(arguments if arguments is not None else {})
        if self._context_param is not None:
            kwargs[self._context_param] = context if context is not None else Context()

        with tracer.start_as_current_span(
            "ag2.tool.invoke",
            attributes={"ag2.tool.name": self.name, "ag2.tool.is_async": self.is_async},
        ):
            if self.is_async:
                return await self._fn(**kwargs)
            return await anyio.to_thread.run_sync(functools.partial(self._fn, **kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow direct invocation of the decorated function."""
        return self._fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, is_async={self.is_async})"


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...


@overload
def tool(
    fn: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorate a sync or async function as an AG2 Tool."""
    if fn is not None:
        return Tool(fn, name=name, description=description)

    def _decorator(target: Callable[..., Any]) -> Tool:
        return Tool(target, name=name, description=description)

    return _decorator


def index_tools(tools: tuple[Tool, ...] | list[Tool]) -> dict[str, Tool]:
    """Build a name-indexed tool mapping, rejecting duplicate tool names."""
    mapping: dict[str, Tool] = {}
    for item in tools:
        if item.name in mapping:
            raise ValueError(f"Duplicate tool name: {item.name!r}")
        mapping[item.name] = item
    return mapping
