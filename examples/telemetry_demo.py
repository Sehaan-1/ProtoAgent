"""Telemetry demo — configure_telemetry, get_tracer, get_meter, and SpanCapture.

Run:
    uv run python examples/telemetry_demo.py
"""

from __future__ import annotations

import asyncio

from ag2 import SpanCapture, configure_telemetry, get_meter, get_tracer


async def process_item(item: str) -> str:
    """Simulate async work inside a named OpenTelemetry span."""
    tracer = get_tracer("ag2.demo")
    with tracer.start_as_current_span("demo.process_item") as span:
        span.set_attribute("item", item)
        await asyncio.sleep(0)
        return item.upper()


async def main() -> None:
    """Run the self-contained telemetry demonstration."""
    configure_telemetry("ag2-demo", log_level="WARNING")

    meter = get_meter("ag2.demo")
    counter = meter.create_counter(
        "demo.items_processed",
        description="Number of items processed",
    )

    items = ["apple", "banana", "cherry"]
    with SpanCapture() as cap:
        for item in items:
            result = await process_item(item)
            counter.add(1, {"item": item})
            print(f"  {item!r:10s} -> {result!r}")

    print(f"\nCaptured {len(cap.span_names)} span(s): {cap.span_names}")


if __name__ == "__main__":
    asyncio.run(main())
