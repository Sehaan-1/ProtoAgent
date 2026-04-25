"""Core telemetry setup: OpenTelemetry SDK wiring and structlog JSON renderer.

This module is safe to import and call even when optional ``opentelemetry-sdk``
or OTLP exporter packages are not installed: ``get_tracer`` and ``get_meter``
delegate directly to ``opentelemetry-api`` (which provides built-in no-op
implementations when no SDK provider is configured), and ``configure_telemetry``
/ ``SpanCapture`` degrade gracefully with warnings if ``opentelemetry-sdk`` is
absent.
"""

from __future__ import annotations

import importlib
import logging
import threading
import types
from typing import TYPE_CHECKING, Any, Self

import structlog
from opentelemetry import metrics, trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
else:
    ReadableSpan = Any
    Span = Any
    SpanProcessor = object
    TracerProvider = Any

__all__ = ["SpanCapture", "configure_telemetry", "get_meter", "get_tracer"]

_log = logging.getLogger(__name__)

_CAPTURE_LOCK = threading.Lock()
_ACTIVE_CAPTURES: list[SpanCapture] = []
_DISPATCH_INSTALLED = False


def _otel_sdk_available() -> bool:
    """Return True when ``opentelemetry-sdk`` is importable."""
    try:
        import opentelemetry.sdk.trace  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _ensure_tracer_provider(service_name: str) -> Any | None:
    """Return an active SDK ``TracerProvider``, installing one globally if needed."""
    if not _otel_sdk_available():
        return None

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

    current = trace.get_tracer_provider()
    if isinstance(current, SdkTracerProvider):
        return current

    provider = SdkTracerProvider(resource=Resource.create({"service.name": service_name}))
    trace.set_tracer_provider(provider)
    return provider


def _ensure_dispatch_processor(provider: Any) -> None:
    """Attach a single shared dispatch processor to ``provider`` for ``SpanCapture``."""
    global _DISPATCH_INSTALLED
    if _DISPATCH_INSTALLED:
        return

    from opentelemetry.sdk.trace import SpanProcessor as SdkSpanProcessor

    class _DispatchSpanProcessor(SdkSpanProcessor):
        def on_start(self, span: Span, parent_context: Any = None) -> None:
            del span, parent_context

        def on_end(self, span: ReadableSpan) -> None:
            with _CAPTURE_LOCK:
                targets = list(_ACTIVE_CAPTURES)
            for capture in targets:
                capture._record(span)

        def shutdown(self) -> None:
            return None

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            del timeout_millis
            return True

    provider.add_span_processor(_DispatchSpanProcessor())
    _DISPATCH_INSTALLED = True


def configure_telemetry(
    service_name: str,
    *,
    otlp_endpoint: str | None = None,
    log_level: str = "INFO",
    console_export: bool = False,
) -> None:
    """Configure OpenTelemetry tracing/metrics and structlog JSON logging.

    Parameters
    ----------
    service_name:
        Resource attribute ``service.name`` attached to emitted telemetry.
    otlp_endpoint:
        Optional OTLP gRPC endpoint (e.g. ``"http://localhost:4317"``). Requires
        ``opentelemetry-exporter-otlp-proto-grpc``; falls back cleanly if absent.
    log_level:
        Minimum logging level name (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
    console_export:
        When ``True`` and ``otlp_endpoint`` is ``None``, attach a
        ``ConsoleSpanExporter`` for local trace inspection.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    logging.basicConfig(level=numeric_level)

    provider = _ensure_tracer_provider(service_name)
    if provider is None:
        _log.warning(
            "opentelemetry-sdk not installed — tracing/metrics disabled. "
            "Install with: pip install 'ag2[telemetry]'"
        )
        return

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    if otlp_endpoint:
        try:
            otlp_mod = importlib.import_module(
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
            )
            otlp_exporter_cls = otlp_mod.OTLPSpanExporter
            provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter_cls(endpoint=otlp_endpoint))
            )
        except ModuleNotFoundError:
            _log.warning(
                "opentelemetry-exporter-otlp-proto-grpc not installed — "
                "falling back to ConsoleSpanExporter"
            )
            if console_export:
                provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    try:
        from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider

        current_meter_provider = metrics.get_meter_provider()
        if not isinstance(current_meter_provider, SdkMeterProvider):
            metrics.set_meter_provider(
                SdkMeterProvider(resource=Resource.create({"service.name": service_name}))
            )
    except Exception as exc:  # pragma: no cover — metrics setup is best-effort
        _log.warning("Could not configure metrics provider: %s", exc)


def get_tracer(name: str) -> trace.Tracer:
    """Return an OpenTelemetry :class:`~opentelemetry.trace.Tracer` for ``name``."""
    return trace.get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    """Return an OpenTelemetry :class:`~opentelemetry.metrics.Meter` for ``name``."""
    return metrics.get_meter(name)


class SpanCapture:
    """Context manager recording finished OpenTelemetry spans in memory for tests.

    Hooks into the active SDK ``TracerProvider`` (installing one if none is set
    yet), so module-level tracers such as ``ag2.agent`` emit spans into every
    active ``SpanCapture`` block without triggering OpenTelemetry override
    warnings across tests.
    """

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []
        self.span_names: list[str] = []
        self._active = False

    def _record(self, span: ReadableSpan) -> None:
        self.spans.append(span)
        self.span_names.append(span.name)

    def by_name(self, name: str) -> list[ReadableSpan]:
        """Return captured spans whose name matches ``name``."""
        return [span for span in self.spans if span.name == name]

    def __enter__(self) -> Self:
        provider = _ensure_tracer_provider("ag2-test")
        if provider is None:
            return self
        with _CAPTURE_LOCK:
            _ensure_dispatch_processor(provider)
            _ACTIVE_CAPTURES.append(self)
            self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        del exc_type, exc_val, exc_tb
        if not self._active:
            return
        with _CAPTURE_LOCK:
            if self in _ACTIVE_CAPTURES:
                _ACTIVE_CAPTURES.remove(self)
            self._active = False
