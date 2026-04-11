"""Minimal real stdio/HTTP MCP server fixture for Slice 6 integration tests."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent


def build_test_mcp_server() -> MCPServer:
    """Create a real MCPServer exposing tools and resources for tests."""
    server = MCPServer(name="ag2-test-mcp")

    @server.tool()
    def add_numbers(a: int, b: int) -> int:
        """Add two integers cleanly."""
        return a + b

    @server.tool()
    def echo_note(text: str, upper: bool = False) -> str:
        """Echo text back, optionally uppercased."""
        return text.upper() if upper else text

    @server.tool()
    def trigger_error(reason: str) -> CallToolResult:
        """Return an explicit MCP error result."""
        return CallToolResult(
            content=[TextContent(type="text", text=f"boom: {reason}")],
            is_error=True,
        )

    @server.resource("memo://welcome")
    def welcome_memo() -> str:
        """Sample resource hosted by the test MCP server."""
        return "Welcome to AG2 MCP integration."

    return server


if __name__ == "__main__":
    build_test_mcp_server().run(transport="stdio")
