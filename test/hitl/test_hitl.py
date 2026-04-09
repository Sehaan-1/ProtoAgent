"""Unit and E2E tests for ag2.hitl Context, hooks, timeouts, and Agent integration."""

from __future__ import annotations

import ast
import json
import threading
import time
from pathlib import Path
from typing import Annotated, Any

import anyio
import httpx
import pytest
from pydantic import SecretStr

import ag2.hitl
from ag2.agent import Agent
from ag2.config import OpenAICompletionsConfig
from ag2.events import (
    ErrorEvent,
    HumanInputRequest,
    HumanMessage,
    MemoryStream,
    ToolResultEvent,
)
from ag2.hitl import (
    CallbackHITLHook,
    CLIHITLHook,
    Context,
    HITLError,
    HITLNotConfiguredError,
    HITLTimeoutError,
    QueueHITLHook,
)
from ag2.providers import OpenAICompletionsClient
from ag2.tools import ToolError, tool


def test_tool_schema_omits_injected_context_and_rejects_multiple_contexts() -> None:
    @tool
    async def deploy(service: str, ctx: Context) -> str:
        """Deploy a service after approval."""
        answer = await ctx.input(f"Deploy {service}?")
        return f"{service}:{answer}"

    assert deploy.expects_context is True
    assert list(deploy.parameters["properties"].keys()) == ["service"]
    assert deploy.parameters["required"] == ["service"]

    @tool
    def annotated_ctx(x: int, ctx: Annotated[Context, "injected"]) -> int:
        del ctx
        return x + 1

    assert annotated_ctx.expects_context is True
    assert list(annotated_ctx.parameters["properties"].keys()) == ["x"]

    with pytest.raises(ToolError, match="at most one Context"):

        @tool
        def two_contexts(a: Context, b: Context) -> str:
            return f"{a}{b}"


async def test_callback_and_cli_hitl_hooks_and_sync_offloading() -> None:
    loop_thread = threading.get_ident()
    callback_thread: int | None = None

    def sync_cb(req: HumanInputRequest) -> str:
        nonlocal callback_thread
        callback_thread = threading.get_ident()
        return f"approved:{req.prompt}"

    async with MemoryStream(history_limit=10) as stream:
        ctx = Context.create(stream=stream, hitl_hook=sync_cb, source="cli", run_id="r1")
        result = await ctx.input("Proceed?")
        assert result == "approved:Proceed?"
        assert callback_thread is not None
        assert callback_thread != loop_thread

        prompts_seen: list[str] = []

        def fake_input(prompt_str: str) -> str:
            prompts_seen.append(prompt_str)
            return ""

        cli_ctx = Context.create(
            stream=stream,
            hitl_hook=CLIHITLHook(input_fn=fake_input),
            source="cli",
            run_id="r2",
        )
        confirmed = await cli_ctx.confirm("Confirm?", default=True)
        assert confirmed is True
        assert "choices=['yes', 'no']" in prompts_seen[0]
        assert "default='yes'" in prompts_seen[0]


async def test_context_input_timeout_default_error_and_choices() -> None:
    async def slow_hook(_req: HumanInputRequest) -> str:
        await anyio.sleep(5.0)
        return "too-late"

    async with MemoryStream(history_limit=20) as stream:
        ctx = Context.create(
            stream=stream,
            hitl_hook=CallbackHITLHook(slow_hook),
            source="tool-runner",
            run_id="r-timeout",
            tool_name="approve",
            call_id="call-9",
        )

        # Timeout with default returns default and emits timed_out HumanMessage.
        val = await ctx.input("Approve?", timeout=0.05, default="deny")
        assert val == "deny"

        human_msgs = [ev for ev in stream.history if isinstance(ev, HumanMessage)]
        assert len(human_msgs) == 1
        assert human_msgs[0].timed_out is True
        assert human_msgs[0].content == "deny"
        assert human_msgs[0].metadata["tool_name"] == "approve"
        assert human_msgs[0].metadata["call_id"] == "call-9"

        # Timeout without default raises HITLTimeoutError.
        with pytest.raises(HITLTimeoutError):
            await ctx.input("Approve strictly?", timeout=0.05)

        # Invalid choice raises HITLError.
        bad_ctx = Context.create(
            stream=stream,
            hitl_hook=lambda _req: "maybe",
        )
        with pytest.raises(HITLError, match="Invalid human choice"):
            await bad_ctx.input("Yes or no?", choices=["yes", "no"])


async def test_queue_hitl_hook_cancellation_cleans_up_pending() -> None:
    async with QueueHITLHook() as hook, MemoryStream() as stream:
        ctx = Context.create(stream=stream, hitl_hook=hook)
        with anyio.move_on_after(0.05) as scope:
            await ctx.input("Waiting for human")
        assert scope.cancelled_caught is True
        assert hook.pending_count == 0


async def test_cli_hitl_hook_timeout_abandons_blocked_input_thread() -> None:
    release = threading.Event()

    def blocking_input(_prompt_str: str) -> str:
        release.wait(timeout=2.0)
        return "late"

    try:
        async with MemoryStream(history_limit=10) as stream:
            meta = {"env": "prod"}
            ctx = Context(
                stream=stream,
                hitl_hook=CLIHITLHook(input_fn=blocking_input),
                metadata=meta,
            )
            meta["env"] = "mutated"
            assert ctx.metadata["env"] == "prod"

            started = time.monotonic()
            answer = await ctx.input("Proceed?", timeout=0.05, default="abort")
            elapsed = time.monotonic() - started
            assert answer == "abort"
            assert elapsed < 0.5
    finally:
        release.set()


async def test_async_callable_object_hitl_hook() -> None:
    class _AsyncApprover:
        async def __call__(self, req: HumanInputRequest) -> str:
            await anyio.lowlevel.checkpoint()
            return f"ok:{req.prompt}"

    async with MemoryStream() as stream:
        ctx = Context.create(stream=stream, hitl_hook=_AsyncApprover())
        assert await ctx.input("Deploy?") == "ok:Deploy?"


async def test_agent_raises_hitl_not_configured_when_hook_missing() -> None:
    @tool
    async def needs_human(ctx: Context) -> str:
        return await ctx.input("Secret code?")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "object": "chat.completion",
                "created": 1700000001,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "needs_human", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    cfg = OpenAICompletionsConfig(model="gpt-4o-mini", api_key=SecretStr("sk-test"))
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client,
        OpenAICompletionsClient(cfg, http_client=http_client) as client,
        MemoryStream(history_limit=20) as stream,
        Agent(name="ops", model=client, tools=[needs_human], stream=stream) as agent,
    ):
        with pytest.raises(HITLNotConfiguredError):
            await agent.ask("Run restricted tool")

        err_events = [ev for ev in stream.history if isinstance(ev, ErrorEvent)]
        assert len(err_events) == 1
        assert err_events[0].error_type == "HITLNotConfiguredError"


@pytest.mark.e2e
async def test_e2e_agent_pauses_in_tool_for_queue_hitl_and_resumes() -> None:
    """End-to-end test: Agent calls HITL tool, pauses via QueueHITLHook, resumes on submit."""

    @tool
    async def transfer_funds(amount: int, ctx: Context) -> str:
        """Transfer funds after human approval."""
        approved = await ctx.confirm(f"Approve transfer of ${amount}?")
        if not approved:
            return "cancelled"
        return f"transferred:${amount}"

    turn = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        payload = json.loads(request.content.decode("utf-8"))
        messages: list[dict[str, Any]] = payload["messages"]

        if turn == 1:
            # Verify context parameter was NOT sent in tool JSON Schema.
            tool_params = payload["tools"][0]["function"]["parameters"]
            assert list(tool_params["properties"].keys()) == ["amount"]
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-1",
                    "object": "chat.completion",
                    "created": 1700000001,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_tx",
                                        "type": "function",
                                        "function": {
                                            "name": "transfer_funds",
                                            "arguments": '{"amount": 250}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )

        assert messages[-1]["role"] == "tool"
        assert messages[-1]["content"] == "transferred:$250"
        return httpx.Response(
            200,
            json={
                "id": "cmpl-2",
                "object": "chat.completion",
                "created": 1700000002,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Transfer of $250 completed after human approval.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    cfg = OpenAICompletionsConfig(model="gpt-4o-mini", api_key=SecretStr("sk-hitl"))
    hook = QueueHITLHook()

    try:
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client,
            OpenAICompletionsClient(cfg, http_client=http_client) as model_client,
            MemoryStream(history_limit=50) as stream,
            Agent(
                name="finance",
                model=model_client,
                tools=[transfer_funds],
                hitl_hook=hook,
                stream=stream,
            ) as agent,
        ):
            reply_text: str | None = None

            async def _human_reviewer() -> None:
                req = await hook.next_request()
                assert req.prompt == "Approve transfer of $250?"
                assert req.choices == ["yes", "no"]
                assert req.metadata["tool_name"] == "transfer_funds"
                assert req.metadata["call_id"] == "call_tx"
                await hook.submit(req.request_id, "yes", sender="alice")

            async def _run_agent() -> None:
                nonlocal reply_text
                reply = await agent.ask("Send $250", run_id="run-hitl-1")
                reply_text = reply.text

            async with anyio.create_task_group() as tg:
                tg.start_soon(_human_reviewer)
                tg.start_soon(_run_agent)

            assert reply_text == "Transfer of $250 completed after human approval."

            req_events = [ev for ev in stream.history if isinstance(ev, HumanInputRequest)]
            msg_events = [ev for ev in stream.history if isinstance(ev, HumanMessage)]
            tool_results = [ev for ev in stream.history if isinstance(ev, ToolResultEvent)]
            assert len(req_events) == 1
            assert len(msg_events) == 1
            assert msg_events[0].sender == "alice"
            assert msg_events[0].request_id == req_events[0].request_id
            assert msg_events[0].run_id == "run-hitl-1"
            assert tool_results[0].content == "transferred:$250"
    finally:
        await hook.aclose()


def test_hitl_does_not_import_downstream_layers() -> None:
    forbidden = ("ag2.tools", "ag2.agent", "ag2.mcp", "ag2.network", "ag2.knowledge")
    hitl_dir = Path(ag2.hitl.__file__).parent
    for py_file in hitl_dir.glob("*.py"):
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
