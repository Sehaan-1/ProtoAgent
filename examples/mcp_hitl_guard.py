"""Risk-based HITL + MCP tool firewall demo over a real stdio MCP server.

Run without an API key:
    uv run python examples/mcp_hitl_guard.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from ag2 import (
    CallbackHITLHook,
    Context,
    HumanInputRequest,
    MCPSession,
    MCPStdioConfig,
    MemoryStream,
    SpanCapture,
    ToolPolicy,
    ToolPolicyRule,
    configure_telemetry,
    guard_tools,
    tool,
)

_MCP_FIXTURE = Path(__file__).resolve().parent.parent / "test" / "mcp" / "server_fixture.py"


@tool
async def delete_workspace_file(path: str) -> str:
    """Simulated destructive native tool protected by the firewall."""
    return f"deleted:{path}"


async def _reviewer(request: HumanInputRequest) -> str:
    """Approve non-production actions and reject destructive ones."""
    approve = "prod" not in request.prompt.lower()
    decision = "approve" if approve else "deny"
    print(f"  [HITL prompt] {request.prompt}\n  [HITL choice] -> {decision}")
    return decision


async def main() -> None:
    """Connect to a real stdio MCP server, guard native + MCP tools, and inspect spans."""
    configure_telemetry("protoagent-guard-demo", log_level="WARNING")

    cfg = MCPStdioConfig(command=sys.executable, args=[str(_MCP_FIXTURE)])

    policy = ToolPolicy(
        rules=[
            ToolPolicyRule(
                "deny",
                pattern="echo_note",
                predicate=lambda _n, args: "secret" in str(args.get("text", "")).lower(),
                reason="secret payload redacted by argument rule",
            ),
        ],
        allow=("add_numbers",),
        confirm=("echo_note", "delete_workspace_file"),
        default="deny",
    )

    async with MCPSession.connect(cfg) as mcp_session:
        mcp_tools = await mcp_session.as_tools()
        guarded = {t.name: t for t in guard_tools([*mcp_tools, delete_workspace_file], policy)}

        stream = MemoryStream()
        ctx = Context.create(stream=stream, hitl_hook=CallbackHITLHook(_reviewer), source="demo")

        with SpanCapture() as spans:
            print("1. Allow rule (MCP read tool):")
            print("  ->", await guarded["add_numbers"].invoke({"a": 19, "b": 23}, context=ctx))

            print("\n2. Confirm rule -> human approves (MCP echo tool):")
            print(
                "  ->",
                await guarded["echo_note"].invoke(
                    {"text": "deploy report", "upper": True},
                    context=ctx,
                ),
            )

            print("\n3. Confirm rule -> human denies (native destructive tool):")
            print(
                "  ->",
                await guarded["delete_workspace_file"].invoke(
                    {"path": "/var/prod/users.db"},
                    context=ctx,
                ),
            )

            print("\n4. Deny rule -> blocked before MCP execution (argument predicate):")
            print(
                "  ->",
                await guarded["echo_note"].invoke(
                    {"text": "leak secret key", "upper": False},
                    context=ctx,
                ),
            )

        guard_outcomes = [
            (
                dict(s.attributes or {}).get("ag2.tool.name"),
                dict(s.attributes or {}).get("ag2.tool.guard.outcome"),
            )
            for s in spans.by_name("ag2.tool.guard")
        ]
        print("\nCaptured guard spans:")
        for tool_name, outcome in guard_outcomes:
            print(f"  - {tool_name}: {outcome}")


if __name__ == "__main__":
    asyncio.run(main())
