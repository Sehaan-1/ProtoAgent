"""Unit and real-server E2E tests for ag2.mcp integration."""

from __future__ import annotations

import ast
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from mcp.shared.memory import create_client_server_memory_streams
from pydantic import SecretStr

import ag2.mcp
from ag2.agent import Agent
from ag2.config import OpenAICompletionsConfig
from ag2.events import MemoryStream, ToolResultEvent
from ag2.mcp import (
    MCPConnectionError,
    MCPHttpConfig,
    MCPSession,
    MCPStdioConfig,
    MCPToolAdapter,
    load_mcp_tools,
)
from ag2.providers import OpenAICompletionsClient
from ag2.tools import ToolError, ToolValidationError
from mcp import ClientSession
from test.mcp.server_fixture import build_test_mcp_server


@asynccontextmanager
async def _in_memory_mcp_session(server: Any) -> Any:
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        lowlevel = server._lowlevel_server
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lowlevel.run,
                server_read,
                server_write,
                lowlevel.create_initialization_options(),
            )
            async with ClientSession(client_read, client_write) as client_session:
                await client_session.initialize()
                yield MCPSession(client_session)
                tg.cancel_scope.cancel()


async def test_mcp_tool_discovery_schema_invocation_errors_and_resources() -> None:
    server = build_test_mcp_server()
    async with _in_memory_mcp_session(server) as mcp_session:
        tools = await load_mcp_tools(mcp_session, prefix="remote_")

        by_name = {item.name: item for item in tools}
        assert {"remote_add_numbers", "remote_echo_note", "remote_trigger_error"} <= set(by_name)

        add_tool = by_name["remote_add_numbers"]
        assert add_tool.parameters["additionalProperties"] is False
        assert add_tool.parameters["required"] == ["a", "b"]
        assert await add_tool.invoke({"a": 14, "b": 28}) == "42"

        with pytest.raises(ToolValidationError, match="missing required keys"):
            await add_tool.invoke({"a": 14})

        err_tool = await MCPToolAdapter(mcp_session).get_tool("trigger_error", prefix="remote_")
        with pytest.raises(ToolError, match="boom: invalid token"):
            await err_tool.invoke({"reason": "invalid token"})

        with pytest.raises(KeyError, match="missing_tool"):
            await MCPToolAdapter(mcp_session).get_tool("missing_tool")

        resources = await mcp_session.list_resources()
        assert [str(r.uri) for r in resources] == ["memo://welcome"]
        assert (
            await mcp_session.read_resource("memo://welcome") == "Welcome to AG2 MCP integration."
        )


async def test_mcp_connection_errors_for_invalid_endpoints() -> None:
    bad_stdio = MCPStdioConfig(command="ag2-nonexistent-mcp-binary-xyz")
    with pytest.raises(MCPConnectionError, match="Failed to connect to stdio MCP server"):
        async with MCPSession.connect(bad_stdio):
            pass

    bad_http = MCPHttpConfig(url="http://127.0.0.1:1/unreachable-mcp")
    with pytest.raises(MCPConnectionError, match="Failed to connect to HTTP MCP server"):
        async with MCPSession.connect(bad_http):
            pass


@pytest.mark.e2e
async def test_e2e_agent_executes_tools_against_real_stdio_mcp_server() -> None:
    """Spawn a real stdio MCP server subprocess and drive it through Agent + OpenAIClient."""
    fixture_path = Path(__file__).parent / "server_fixture.py"
    cfg = MCPStdioConfig(command=sys.executable, args=[str(fixture_path)])

    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        payload = json.loads(request.content.decode("utf-8"))
        messages: list[dict[str, Any]] = payload["messages"]

        if turn == 1:
            names = {tool_def["function"]["name"] for tool_def in payload["tools"]}
            assert {"add_numbers", "echo_note", "trigger_error"} <= names
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-mcp-1",
                    "object": "chat.completion",
                    "created": 1700000101,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_add",
                                        "type": "function",
                                        "function": {
                                            "name": "add_numbers",
                                            "arguments": '{"a": 8, "b": 34}',
                                        },
                                    },
                                    {
                                        "id": "call_echo",
                                        "type": "function",
                                        "function": {
                                            "name": "echo_note",
                                            "arguments": '{"text": "mcp live", "upper": true}',
                                        },
                                    },
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )

        tool_contents = [m["content"] for m in messages if m["role"] == "tool"]
        assert tool_contents == ["42", "MCP LIVE"]
        return httpx.Response(
            200,
            json={
                "id": "cmpl-mcp-2",
                "object": "chat.completion",
                "created": 1700000102,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Computed 42 and echoed MCP LIVE via MCP.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    model_cfg = OpenAICompletionsConfig(model="gpt-4o-mini", api_key=SecretStr("sk-mcp"))
    async with MCPSession.connect(cfg) as mcp_session:
        mcp_tools = await mcp_session.as_tools()
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client,
            OpenAICompletionsClient(model_cfg, http_client=http_client) as model_client,
            MemoryStream(history_limit=40) as stream,
            Agent(
                name="mcp-agent",
                model=model_client,
                tools=mcp_tools,
                stream=stream,
            ) as agent,
        ):
            reply = await agent.ask("Add 8 and 34 and echo 'mcp live' in caps.")

        assert reply.text == "Computed 42 and echoed MCP LIVE via MCP."
        results = [ev for ev in stream.history if isinstance(ev, ToolResultEvent)]
        assert [ev.content for ev in results] == ["42", "MCP LIVE"]
        assert all(not ev.is_error for ev in results)


def test_mcp_does_not_import_downstream_or_sibling_layers() -> None:
    forbidden = ("ag2.agent", "ag2.hitl", "ag2.network", "ag2.knowledge")
    mcp_dir = Path(ag2.mcp.__file__).parent
    for py_file in mcp_dir.glob("*.py"):
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
