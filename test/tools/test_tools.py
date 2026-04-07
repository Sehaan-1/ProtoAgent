"""Unit tests for ag2.tools decorator, schema generation, validation, and isolation."""

from __future__ import annotations

import ast
import threading
import time
from pathlib import Path
from typing import Annotated

import anyio
import pytest
from pydantic import BaseModel, Field

import ag2.tools
from ag2.tools import Tool, ToolError, ToolValidationError, index_tools, serialize_tool_result, tool


class _Range(BaseModel):
    low: int
    high: int


class _Report(BaseModel):
    total: int


async def test_sync_and_async_tool_decorator_and_schema() -> None:
    @tool
    def multiply(
        a: Annotated[int, Field(description="Left operand")],
        b: int = 3,
    ) -> int:
        """Multiply two integers."""
        return a * b

    @tool(name="sum_range", description="Sum values in a range.")
    async def sum_async(bounds: _Range) -> _Report:
        await anyio.lowlevel.checkpoint()
        return _Report(total=sum(range(bounds.low, bounds.high + 1)))

    assert isinstance(multiply, Tool)
    assert multiply.name == "multiply"
    assert multiply.description == "Multiply two integers."
    assert multiply.is_async is False
    assert multiply(4, 5) == 20
    assert multiply.parameters["additionalProperties"] is False
    assert multiply.parameters["properties"]["a"]["description"] == "Left operand"
    assert multiply.parameters["required"] == ["a"]

    assert sum_async.name == "sum_range"
    assert sum_async.is_async is True
    assert await multiply.invoke({"a": 6}) == 18

    report = await sum_async.invoke({"bounds": {"low": 1, "high": 4}})
    assert isinstance(report, _Report)
    assert report.total == 10
    assert serialize_tool_result(report) == '{"total":10}'


async def test_tool_validation_errors_and_sync_offloading() -> None:
    event_loop_thread_id = threading.get_ident()
    worker_thread_id: int | None = None

    @tool
    def blocking_work(seconds: float) -> str:
        nonlocal worker_thread_id
        worker_thread_id = threading.get_ident()
        time.sleep(seconds)
        return "done"

    with pytest.raises(ToolValidationError):
        await blocking_work.invoke({"seconds": "not-a-float"})

    with pytest.raises(ToolValidationError):
        await blocking_work.invoke({"seconds": 0.01, "unexpected": True})

    res = await blocking_work.invoke({"seconds": 0.01})
    assert res == "done"
    assert worker_thread_id is not None
    assert worker_thread_id != event_loop_thread_id


def test_tool_rejects_varargs_and_duplicates() -> None:
    with pytest.raises(ToolError, match=r"\*args"):

        @tool
        def bad_varargs(*args: int) -> int:
            return sum(args)

    @tool(name="dup")
    def first() -> str:
        return "a"

    @tool(name="dup")
    def second() -> str:
        return "b"

    with pytest.raises(ValueError, match="Duplicate tool name"):
        index_tools([first, second])


def test_serialize_tool_result_variants() -> None:
    assert serialize_tool_result(None) == ""
    assert serialize_tool_result("plain") == "plain"
    assert serialize_tool_result({"k": 1}) == '{"k": 1}'
    assert serialize_tool_result([1, 2]) == "[1, 2]"
    assert serialize_tool_result(42) == "42"


def test_tools_does_not_import_downstream_layers() -> None:
    forbidden = ("ag2.agent", "ag2.mcp", "ag2.network", "ag2.knowledge")
    tools_dir = Path(ag2.tools.__file__).parent
    for py_file in tools_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden), (
                        f"{py_file.name} must not import {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert not node.module.startswith(forbidden), (
                    f"{py_file.name} must not import {node.module}"
                )
