"""Unit tests and dependency boundary checks for ag2.telemetry."""

from __future__ import annotations

import ast
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import structlog
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider

import ag2
import ag2.telemetry
from ag2.agent import Agent
from ag2.events import ChatMessage, Usage
from ag2.providers import ModelResponse, ResponseFormat, ToolSpec
from ag2.telemetry import SpanCapture, configure_telemetry, get_meter, get_tracer


class _FakeModelClient:
    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolSpec] = (),
        response_format: ResponseFormat | None = None,
    ) -> ModelResponse:
        del messages, tools, response_format
        return ModelResponse(
            message=ChatMessage(role="assistant", content="telemetry ok"),
            usage=Usage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
            model="fake-otel-model",
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        return None


def test_configure_telemetry_sets_up_provider(capsys: pytest.CaptureFixture[str]) -> None:
    configure_telemetry("ag2-test-service", log_level="INFO", console_export=False)

    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    assert isinstance(metrics.get_meter_provider(), MeterProvider)

    tracer = get_tracer("ag2.test")
    meter = get_meter("ag2.test")
    assert tracer is not None
    assert meter is not None

    log = structlog.get_logger("ag2.test.logger")
    log.info("telemetry.configured", component="unit_test")
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip().splitlines()[-1])
    assert payload["event"] == "telemetry.configured"
    assert payload["level"] == "info"
    assert payload["component"] == "unit_test"

    # Calling with an OTLP endpoint when the optional gRPC exporter isn't installed
    # should degrade cleanly rather than raising.
    configure_telemetry(
        "ag2-test-service",
        otlp_endpoint="http://localhost:4317",
        log_level="WARNING",
        console_export=False,
    )


async def test_span_capture_records_agent_spans() -> None:
    # Run configure_telemetry first to verify SpanCapture still records spans even
    # after a global TracerProvider has been configured.
    configure_telemetry("ag2-agent-test", log_level="WARNING", console_export=False)

    async with Agent(name="observed-agent", model=_FakeModelClient()) as agent:
        with SpanCapture() as cap:
            reply = await agent.ask("Hello telemetry", run_id="run-otel-1")

    assert reply.text == "telemetry ok"
    assert "ag2.agent.ask" in cap.span_names
    ask_spans = cap.by_name("ag2.agent.ask")
    assert len(ask_spans) == 1
    attrs = ask_spans[0].attributes or {}
    assert attrs["ag2.agent.name"] == "observed-agent"
    assert attrs["ag2.run_id"] == "run-otel-1"

    # Subsequent SpanCapture blocks start isolated and don't retain prior spans.
    with SpanCapture() as second_cap:
        tracer = get_tracer("ag2.custom")
        with tracer.start_as_current_span("custom.work"):
            pass

    assert second_cap.span_names == ["custom.work"]


def test_telemetry_does_not_import_downstream() -> None:
    forbidden = (
        "ag2.agent",
        "ag2.compact",
        "ag2.hitl",
        "ag2.knowledge",
        "ag2.mcp",
        "ag2.middleware",
        "ag2.network",
        "ag2.providers",
        "ag2.tools",
    )
    assert ag2.telemetry.__file__ is not None
    pkg_dir = Path(ag2.telemetry.__file__).parent
    for py_file in pkg_dir.glob("*.py"):
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


def test_top_level_telemetry_exports_and_examples_importable() -> None:
    assert ag2.configure_telemetry is configure_telemetry
    assert ag2.SpanCapture is SpanCapture
    assert ag2.get_tracer is get_tracer
    assert ag2.get_meter is get_meter

    examples_dir = Path(__file__).resolve().parents[2] / "examples"
    example_modules = [p.stem for p in sorted(examples_dir.glob("*.py")) if p.name != "__init__.py"]
    assert {
        "quickstart",
        "multi_provider",
        "streaming_agent",
        "multi_agent_workflow",
        "knowledge_agent",
        "telemetry_demo",
    }.issubset(set(example_modules))

    for mod_name in example_modules:
        mod: Any = importlib.import_module(f"examples.{mod_name}")
        assert callable(mod.main)


async def test_telemetry_demo_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    demo_mod: Any = importlib.import_module("examples.telemetry_demo")
    await demo_mod.main()
    out = capsys.readouterr().out
    assert "Captured 3 span(s)" in out
    assert "demo.process_item" in out
