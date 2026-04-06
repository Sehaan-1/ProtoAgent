set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Default recipe: run all quality gates
default: check

# Sync development environment from uv.lock
sync:
    uv sync --locked --dev

# Run ruff linter
lint:
    uv run ruff check .

# Check formatting without modifying files
format-check:
    uv run ruff format --check .

# Auto-format code and fix safe lint issues
format:
    uv run ruff check --fix .
    uv run ruff format .

# Run strict static type checking
typecheck:
    uv run mypy ag2 test

# Run the test suite
test *args:
    uv run pytest -v {{args}}

# Run all CI verification gates
check: lint format-check typecheck test
