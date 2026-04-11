"""Async MCP client transports, typed configs, and session lifecycle management."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
import structlog
from mcp.client.stdio import StdioServerParameters, stdio_client
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

from ag2.tools.tool import Tool
from mcp import ClientSession, types

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.mcp")


class MCPError(RuntimeError):
    """Base error for MCP transport, discovery, and invocation failures."""


class MCPConnectionError(MCPError):
    """Raised when connecting to or initializing an MCP server fails."""


class MCPStdioConfig(BaseModel):
    """Configuration for spawning a local stdio MCP server subprocess."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    transport: Literal["stdio"] = "stdio"
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | Path | None = None


class MCPHttpConfig(BaseModel):
    """Configuration for connecting to a streamable-HTTP MCP server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    transport: Literal["http"] = "http"
    url: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)
    terminate_on_close: bool = True


AnyMCPConfig = Annotated[
    MCPStdioConfig | MCPHttpConfig,
    Field(discriminator="transport"),
]


def _extract_resource_text(contents: list[Any]) -> str:
    parts: list[str] = []
    for item in contents:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        blob = getattr(item, "blob", None)
        if isinstance(blob, str):
            parts.append(blob)
            continue
        parts.append(str(item))
    return "\n".join(parts)


class MCPSession:
    """Thin async context-manager wrapper over the official ``mcp.ClientSession``."""

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    @property
    def raw(self) -> ClientSession:
        """Return the underlying ``mcp.ClientSession``."""
        return self._session

    @classmethod
    @contextlib.asynccontextmanager
    async def connect_stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> AsyncIterator[MCPSession]:
        """Launch a stdio MCP server process and yield an initialized session."""
        params = StdioServerParameters(
            command=command,
            args=list(args) if args is not None else [],
            env=env,
            cwd=cwd,
        )
        try:
            async with (
                stdio_client(params) as (read_stream, write_stream),
                ClientSession(read_stream, write_stream) as session,
            ):
                await session.initialize()
                logger.info(
                    "ag2.mcp.stdio_connected",
                    command=command,
                    args=params.args,
                )
                yield cls(session)
        except anyio.get_cancelled_exc_class():
            raise
        except MCPError:
            raise
        except Exception as exc:
            raise MCPConnectionError(
                f"Failed to connect to stdio MCP server {command!r}: {exc}"
            ) from exc

    @classmethod
    @contextlib.asynccontextmanager
    async def connect_http(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        terminate_on_close: bool = True,
        http_client: Any = None,
    ) -> AsyncIterator[MCPSession]:
        """Connect to a streamable-HTTP MCP endpoint and yield an initialized session."""
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client

        managed_http_client = None
        client_to_use = http_client
        if client_to_use is None and headers:
            managed_http_client = create_mcp_http_client(headers=headers)
            client_to_use = managed_http_client

        try:
            async with contextlib.AsyncExitStack() as stack:
                if managed_http_client is not None:
                    await stack.enter_async_context(managed_http_client)
                read_stream, write_stream = await stack.enter_async_context(
                    streamable_http_client(
                        url,
                        http_client=client_to_use,
                        terminate_on_close=terminate_on_close,
                    )
                )
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
                logger.info("ag2.mcp.http_connected", url=url)
                yield cls(session)
        except anyio.get_cancelled_exc_class():
            raise
        except MCPError:
            raise
        except Exception as exc:
            raise MCPConnectionError(
                f"Failed to connect to HTTP MCP server {url!r}: {exc}"
            ) from exc

    @classmethod
    @contextlib.asynccontextmanager
    async def connect(cls, config: MCPStdioConfig | MCPHttpConfig) -> AsyncIterator[MCPSession]:
        """Open an ``MCPSession`` from an ``MCPStdioConfig`` or ``MCPHttpConfig``."""
        if isinstance(config, MCPStdioConfig):
            async with cls.connect_stdio(
                config.command,
                config.args,
                env=config.env,
                cwd=config.cwd,
            ) as session:
                yield session
            return
        async with cls.connect_http(
            config.url,
            headers=config.headers,
            terminate_on_close=config.terminate_on_close,
        ) as session:
            yield session

    async def list_tools(self) -> list[types.Tool]:
        """List all tools exposed by the remote server, following pagination cursors."""
        tools: list[types.Tool] = []
        cursor: str | None = None
        with tracer.start_as_current_span("ag2.mcp.list_tools"):
            while True:
                params = types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None
                page = await self._session.list_tools(params=params)
                tools.extend(page.tools)
                cursor = page.next_cursor
                if cursor is None:
                    break
        return tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> types.CallToolResult:
        """Invoke a remote MCP tool by name and return its ``CallToolResult``."""
        with tracer.start_as_current_span(
            "ag2.mcp.call_tool",
            attributes={"ag2.mcp.tool": name},
        ):
            result = await self._session.call_tool(name, arguments=arguments or {})
        return result

    async def list_resources(self) -> list[types.Resource]:
        """List all resources exposed by the remote server, following pagination."""
        resources: list[types.Resource] = []
        cursor: str | None = None
        with tracer.start_as_current_span("ag2.mcp.list_resources"):
            while True:
                params = types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None
                page = await self._session.list_resources(params=params)
                resources.extend(page.resources)
                cursor = page.next_cursor
                if cursor is None:
                    break
        return resources

    async def read_resource(self, uri: str) -> str:
        """Read a resource URI from the remote MCP server as text."""
        with tracer.start_as_current_span(
            "ag2.mcp.read_resource",
            attributes={"ag2.mcp.resource_uri": uri},
        ):
            result = await self._session.read_resource(uri)
        if not isinstance(result, types.ReadResourceResult):
            raise MCPError(
                f"MCP resource {uri!r} returned unsupported result type {type(result).__name__}"
            )
        return _extract_resource_text(list(result.contents))

    async def as_tools(self, *, prefix: str | None = None) -> list[Tool]:
        """Discover remote MCP tools and wrap each as a native ``ag2.tools.Tool``."""
        from ag2.mcp.adapter import load_mcp_tools

        return await load_mcp_tools(self, prefix=prefix)
