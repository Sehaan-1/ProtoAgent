"""Adapters converting discovered MCP server tools into native ``ag2.tools.Tool`` objects."""

from __future__ import annotations

import json
from typing import Any

import structlog
from opentelemetry import trace

from ag2.mcp.client import MCPSession
from ag2.providers.base import enforce_strict_json_schema
from ag2.tools.tool import Tool, ToolError
from mcp import ClientSession, types

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.mcp")


def _serialize_call_tool_result(result: types.CallToolResult, tool_name: str) -> str:
    """Serialize an MCP CallToolResult into a string, raising ToolError if is_error=True."""
    parts: list[str] = []
    for item in result.content:
        if isinstance(item, types.TextContent):
            parts.append(item.text)
        else:
            try:
                parts.append(json.dumps(item.model_dump(mode="json")))
            except Exception:
                parts.append(str(item))

    if not parts and result.structured_content is not None:
        parts.append(json.dumps(result.structured_content, default=str))

    text = "\n".join(parts)
    if result.is_error:
        raise ToolError(f"MCP tool {tool_name!r} failed: {text or 'unknown MCP error'}")
    return text


def mcp_tool_to_ag2_tool(
    mcp_tool: types.Tool,
    session: MCPSession | ClientSession,
    *,
    name_prefix: str | None = None,
) -> Tool:
    """Wrap a remote MCP tool descriptor into an executable ``ag2.tools.Tool``."""
    remote_name = mcp_tool.name
    exposed_name = f"{name_prefix}{remote_name}" if name_prefix else remote_name
    description = mcp_tool.description or ""
    raw_schema = (
        dict(mcp_tool.input_schema)
        if mcp_tool.input_schema
        else {
            "type": "object",
            "properties": {},
        }
    )
    parameters = enforce_strict_json_schema(raw_schema)

    async def _invoke(**kwargs: Any) -> str:
        with tracer.start_as_current_span(
            "ag2.mcp.tool.invoke",
            attributes={"ag2.mcp.tool": remote_name, "ag2.tool.name": exposed_name},
        ):
            logger.debug("ag2.mcp.tool.invoke", tool=remote_name, args=kwargs)
            if isinstance(session, MCPSession):
                result = await session.call_tool(remote_name, arguments=kwargs)
            else:
                result = await session.call_tool(remote_name, arguments=kwargs)
            return _serialize_call_tool_result(result, remote_name)

    return Tool.from_schema(
        _invoke,
        name=exposed_name,
        description=description,
        parameters=parameters,
    )


class MCPToolAdapter:
    """Discover tools on an MCP session and convert them to native ``Tool`` callables."""

    def __init__(self, session: MCPSession | ClientSession) -> None:
        self._session = session

    async def list_tools(self, *, prefix: str | None = None) -> list[Tool]:
        """Fetch all tools from the MCP server and return converted ``Tool`` instances."""
        if isinstance(self._session, MCPSession):
            descriptors = await self._session.list_tools()
        else:
            descriptors = []
            cursor: str | None = None
            while True:
                params = types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None
                page = await self._session.list_tools(params=params)
                descriptors.extend(page.tools)
                cursor = page.next_cursor
                if cursor is None:
                    break

        return [
            mcp_tool_to_ag2_tool(descriptor, self._session, name_prefix=prefix)
            for descriptor in descriptors
        ]

    async def get_tool(self, name: str, *, prefix: str | None = None) -> Tool:
        """Fetch and return a single converted tool matching ``name`` (or unprefixed name)."""
        tools = await self.list_tools(prefix=prefix)
        expected = f"{prefix}{name}" if prefix else name
        for item in tools:
            if item.name in (name, expected):
                return item
        raise KeyError(f"MCP tool {name!r} not found on server")


async def load_mcp_tools(
    session: MCPSession | ClientSession,
    *,
    prefix: str | None = None,
) -> list[Tool]:
    """Convenience helper converting all remote tools on ``session`` to AG2 tools."""
    return await MCPToolAdapter(session).list_tools(prefix=prefix)
