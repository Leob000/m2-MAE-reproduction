set shell := ["bash", "-c"]

# Run all checks (format, lint, type-check) on the codebase
check path=".":
    @echo "Running checks on {{path}}..."
    uv run ruff format {{path}}
    uv run ruff check --fix {{path}}
    uv run ty check {{path}}

# Run the full suite including tests
full-check: (check) test

# Run code formatting
format path=".":
    uv run ruff format {{path}}

# Run linting with auto-fixes
lint path=".":
    uv run ruff check --fix {{path}}

# Run static type checking
type-check path=".":
    uv run ty check {{path}}

# Run tests
test:
    uv run pytest
