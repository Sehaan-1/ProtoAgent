"""Smoke tests for Slice 1 repo and toolchain scaffold."""

from pathlib import Path

import anyio.lowlevel
import pydantic

import ag2


def test_package_import_and_version() -> None:
    """Verify the top-level ag2 package imports cleanly and exposes __version__."""
    assert isinstance(ag2.__version__, str)
    assert ag2.__version__ == "0.1.0"


def test_py_typed_marker_present() -> None:
    """Verify PEP 561 py.typed marker is bundled in the ag2 package."""
    marker = Path(ag2.__file__).parent / "py.typed"
    assert marker.is_file()


async def test_async_runtime_smoke() -> None:
    """Verify pytest-asyncio and anyio async primitives run cleanly."""
    assert int(pydantic.__version__.split(".")[0]) >= 2
    await anyio.lowlevel.checkpoint()
