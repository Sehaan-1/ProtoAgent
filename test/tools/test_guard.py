"""Unit and E2E tests for ag2.tools.guard risk-based HITL and MCP firewall."""

from __future__ import annotations

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

from ag2.agent import Agent
from ag2.config import OpenAICompletionsConfig
from ag2.events import (
    HumanInputRequest,
    HumanMessage,
    MemoryStream,
    ToolResultEvent,
)
from ag2.hitl import (
    CallbackHITLHook,
    Context,
    HITLNotConfiguredError,
    HITLTimeoutError,
    QueueHITLHook,
)
from ag2.mcp import MCPSession, MCPStdioConfig, load_mcp_tools
from ag2.providers import OpenAICompletionsClient
from ag2.telemetry import SpanCapture
from ag2.tools import (
    GuardedTool,
    Tool,
    ToolDeniedError,
    ToolPolicy,
    ToolPolicyRule,
    ToolValidationError,
    guard_tool,
    guard_tools,
    tool,
)
from mcp import ClientSession
from test.mcp.server_fixture import build_test_mcp_server

_MCP_FIXTURE = Path(__file__).resolve().parent.parent / "mcp" / "server_fixture.py"


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


def test_policy_rule_validation_and_precedence() -> None:
    with pytest.raises(ValueError, match="Invalid tool decision"):
        ToolPolicyRule(decision="block")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="at least one"):
        ToolPolicyRule(decision="deny")

    with pytest.raises(ValueError, match="Invalid default decision"):
        ToolPolicy(default="block")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid timeout_default"):
        ToolPolicy(timeout_default="skip")  # type: ignore[arg-type]

    policy = ToolPolicy(
        rules=[
            ToolPolicyRule(
                "deny",
                pattern="sql_*",
                predicate=lambda _name, args: "DROP" in str(args.get("query", "")).upper(),
                reason="destructive SQL blocked",
            ),
        ],
        allow=("read_*",),
        confirm=("sql_*", "write_*"),
        deny=("write_secret",),
        default="confirm",
    )

    assert policy.evaluate("sql_exec", {"query": "DROP TABLE users"}).decision == "deny"
    assert policy.evaluate("sql_exec", {"query": "SELECT 1"}).decision == "confirm"
    assert policy.evaluate("write_secret").decision == "deny"
    assert policy.evaluate("write_note").decision == "confirm"
    assert policy.evaluate("read_note").decision == "allow"
    assert policy.evaluate("unlisted_mcp_tool").decision == "confirm"


def test_guard_preserves_tool_spec_and_schema() -> None:
    @tool
    def transfer(amount: int, account: str = "checking") -> str:
        """Transfer funds between accounts."""
        return f"{amount}:{account}"

    guarded = guard_tool(transfer, ToolPolicy(confirm=("transfer",)))

    assert isinstance(guarded, Tool)
    assert isinstance(guarded, GuardedTool)
    assert guarded.spec == transfer.spec
    assert guarded.parameters == transfer.parameters
    assert guarded.expects_context is True
    assert guarded(25) == "25:checking"

    read_only = guard_tool(transfer, ToolPolicy(allow=("transfer",)))
    assert read_only.expects_context is False


async def test_guard_allow_confirm_deny_and_rejection_modes() -> None:
    executed: list[tuple[str, Any]] = []

    @tool
    async def read_metric(name: str) -> str:
        executed.append(("read_metric", name))
        return f"metric:{name}=99"

    @tool
    async def deploy_service(env: str) -> str:
        executed.append(("deploy_service", env))
        return f"deployed:{env}"

    @tool
    async def wipe_db(target: str) -> str:
        executed.append(("wipe_db", target))
        return f"wiped:{target}"

    policy = ToolPolicy(
        allow=("read_*",),
        confirm=("deploy_*",),
        deny=("wipe_*",),
        default="deny",
    )
    guarded = {t.name: t for t in guard_tools([read_metric, deploy_service, wipe_db], policy)}

    with pytest.raises(ToolValidationError):
        await guarded["read_metric"].invoke({})

    hook = QueueHITLHook()
    stream = MemoryStream(history_limit=20)
    ctx = Context.create(stream=stream, hitl_hook=hook, source="tester", run_id="run-guard")

    # Allow runs directly without HITL interaction.
    assert await guarded["read_metric"].invoke({"name": "cpu"}, context=ctx) == "metric:cpu=99"

    # Deny rejects without executing tool or prompting human.
    denied_msg = await guarded["wipe_db"].invoke({"target": "prod"}, context=ctx)
    assert "denied by policy" in denied_msg

    # Confirm approves and runs tool.
    async def _respond_sequence() -> None:
        req1 = await hook.next_request()
        assert isinstance(req1, HumanInputRequest)
        assert req1.choices == ["approve", "deny"]
        assert "deploy_service" in req1.prompt
        await hook.submit(req1.request_id, "approve")

        req2 = await hook.next_request()
        await hook.submit(req2.request_id, "deny")

    async with anyio.create_task_group() as tg:
        tg.start_soon(_respond_sequence)
        approved = await guarded["deploy_service"].invoke({"env": "staging"}, context=ctx)
        rejected = await guarded["deploy_service"].invoke({"env": "prod"}, context=ctx)

    assert approved == "deployed:staging"
    assert "denied by human reviewer" in rejected
    assert executed == [("read_metric", "cpu"), ("deploy_service", "staging")]

    # Audit events recorded for HITL approvals/rejections.
    human_msgs = [ev for ev in stream.history if isinstance(ev, HumanMessage)]
    assert [m.content for m in human_msgs] == ["approve", "deny"]

    # Strict raise_on_deny mode raises ToolDeniedError.
    strict_tool = guard_tool(wipe_db, ToolPolicy(deny=("wipe_*",), raise_on_deny=True))
    with pytest.raises(ToolDeniedError, match="wipe_db"):
        await strict_tool.invoke({"target": "prod"}, context=ctx)


async def test_guard_missing_hitl_hook_and_timeout_handling() -> None:
    @tool
    async def sensitive_op(x: int) -> int:
        return x * 2

    guarded = guard_tool(sensitive_op, ToolPolicy(confirm=("sensitive_op",)))
    with pytest.raises(HITLNotConfiguredError):
        await guarded.invoke({"x": 3})

    async def slow_hook(_request: HumanInputRequest) -> str:
        await anyio.sleep(0.2)
        return "approve"

    timeout_deny_tool = guard_tool(
        sensitive_op,
        ToolPolicy(confirm=("sensitive_op",), confirm_timeout=0.02, timeout_default="deny"),
    )
    ctx = Context.create(stream=MemoryStream(), hitl_hook=CallbackHITLHook(slow_hook))
    msg = await timeout_deny_tool.invoke({"x": 3}, context=ctx)
    assert "denied by human reviewer" in msg

    timeout_raise_tool = guard_tool(
        sensitive_op,
        ToolPolicy(
            confirm=("sensitive_op",),
            confirm_timeout=0.02,
            timeout_default=None,
            raise_on_timeout=True,
        ),
    )
    with pytest.raises(HITLTimeoutError):
        await timeout_raise_tool.invoke({"x": 3}, context=ctx)


async def test_guard_wraps_mcp_tools_transparently_with_telemetry() -> None:
    server = build_test_mcp_server()
    async with _in_memory_mcp_session(server) as mcp_session:
        raw_tools = await load_mcp_tools(mcp_session)
        raw_specs = {t.name: t.spec for t in raw_tools}

        policy = ToolPolicy(
            rules=[
                ToolPolicyRule(
                    "deny",
                    pattern="echo_note",
                    predicate=lambda _n, args: "forbidden" in str(args.get("text", "")).lower(),
                    reason="blocked keyword in MCP payload",
                )
            ],
            allow=("add_numbers",),
            confirm=("echo_note",),
            default="deny",
        )
        guarded_tools = guard_tools(raw_tools, policy)
        assert {t.name: t.spec for t in guarded_tools} == raw_specs
        by_name = {t.name: t for t in guarded_tools}

        prompts: list[str] = []

        async def approve_echo(req: HumanInputRequest) -> str:
            prompts.append(req.prompt)
            return "approve"

        ctx = Context.create(stream=MemoryStream(), hitl_hook=CallbackHITLHook(approve_echo))

        with SpanCapture() as spans:
            assert await by_name["add_numbers"].invoke({"a": 3, "b": 4}, context=ctx) == "7"
            assert (
                await by_name["echo_note"].invoke({"text": "ok", "upper": True}, context=ctx)
                == "OK"
            )
            blocked = await by_name["echo_note"].invoke({"text": "FORBIDDEN token"}, context=ctx)

        assert "denied by policy" in blocked
        assert len(prompts) == 1
        guard_spans = spans.by_name("ag2.tool.guard")
        outcomes = [dict(s.attributes or {}).get("ag2.tool.guard.outcome") for s in guard_spans]
        assert outcomes == ["allowed", "approved_by_human", "denied_by_policy"]


def test_top_level_lazy_exports() -> None:
    import ag2

    assert ag2.ToolPolicy is ToolPolicy
    assert ag2.ToolPolicyRule is ToolPolicyRule
    assert ag2.GuardedTool is GuardedTool
    assert ag2.ToolDeniedError is ToolDeniedError
    assert ag2.guard_tool is guard_tool
    assert ag2.guard_tools is guard_tools


@pytest.mark.e2e
async def test_e2e_agent_with_guarded_stdio_mcp_tools_and_hitl() -> None:
    """Verify Agent executes allow/confirm/deny tools against a real stdio MCP server."""
    cfg = MCPStdioConfig(command=sys.executable, args=[str(_MCP_FIXTURE)])

    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        payload = json.loads(request.content.decode("utf-8"))
        messages: list[dict[str, Any]] = payload["messages"]

        if turn == 1:
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-guard-1",
                    "object": "chat.completion",
                    "created": 1700000201,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_read",
                                        "type": "function",
                                        "function": {
                                            "name": "add_numbers",
                                            "arguments": '{"a": 10, "b": 5}',
                                        },
                                    },
                                    {
                                        "id": "call_confirm_approve",
                                        "type": "function",
                                        "function": {
                                            "name": "echo_note",
                                            "arguments": '{"text": "ship it", "upper": true}',
                                        },
                                    },
                                    {
                                        "id": "call_deny_arg",
                                        "type": "function",
                                        "function": {
                                            "name": "echo_note",
                                            "arguments": '{"text": "secret drop", "upper": false}',
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
        assert tool_contents[0] == "15"
        assert tool_contents[1] == "SHIP IT"
        assert "denied by policy" in tool_contents[2]
        return httpx.Response(
            200,
            json={
                "id": "cmpl-guard-2",
                "object": "chat.completion",
                "created": 1700000202,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Read 15, confirmed echo SHIP IT, blocked secret payload.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    policy = ToolPolicy(
        rules=[
            ToolPolicyRule(
                "deny",
                pattern="echo_note",
                predicate=lambda _n, args: "secret" in str(args.get("text", "")).lower(),
                reason="sensitive text blocked",
            ),
        ],
        allow=("add_numbers",),
        confirm=("echo_note",),
        default="deny",
    )

    async def reviewer(_req: HumanInputRequest) -> str:
        return "approve"

    model_cfg = OpenAICompletionsConfig(model="gpt-4o-mini", api_key=SecretStr("sk-guard"))
    async with MCPSession.connect(cfg) as mcp_session:
        mcp_tools = guard_tools(await mcp_session.as_tools(), policy)
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client,
            OpenAICompletionsClient(model_cfg, http_client=http_client) as model_client,
            MemoryStream(history_limit=40) as stream,
            Agent(
                name="guarded-mcp-agent",
                model=model_client,
                tools=mcp_tools,
                hitl_hook=CallbackHITLHook(reviewer),
                stream=stream,
            ) as agent,
        ):
            reply = await agent.ask("Run allowed read, confirmed echo, and blocked echo.")

        assert reply.text == "Read 15, confirmed echo SHIP IT, blocked secret payload."
        results = [ev for ev in stream.history if isinstance(ev, ToolResultEvent)]
        assert len(results) == 3
        assert results[0].content == "15"
        assert results[1].content == "SHIP IT"
        assert "denied by policy" in results[2].content
