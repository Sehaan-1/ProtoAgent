# AG2 (ProtoAgent)

Clean, async-first, protocol-driven multi-agent framework built slice-by-slice following the modern `import ag2` (v1.x) architecture.

## Development Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# Install dev dependencies
uv sync --locked --dev

# Run lint, format check, typecheck, and tests
uv run ruff check .
uv run ruff format --check .
uv run mypy ag2 test
uv run pytest -v

# Or via just
just check
```

## Optional Extras

Install provider or integration extras as needed:

```bash
uv sync --extra openai --extra mcp
# or everything
uv sync --all-extras
```
