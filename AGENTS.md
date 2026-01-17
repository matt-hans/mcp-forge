# Repository Guidelines

## Project Structure & Module Organization
`src/mcp_forge/` holds the Python package, with submodules for `data/` (synthesis, QC) and `tools/` (MCP inspection). CLI entry points live in `src/mcp_forge/cli.py`. Tests are split into `tests/unit/` and `tests/integration/`, with fixtures in `tests/fixtures/`. Architecture notes live in `TECHNICAL_ARCHITECTURE_v1.1.md`. Pipeline state is persisted at runtime under `.mcp-forge/` (generated locally).

## Build, Test, and Development Commands
```bash
# Install for development
pip install -e ".[dev]"

# Lint and type check
ruff check .
mypy .

# Run tests
pytest
```
Use `mcp-forge doctor` to validate CUDA/GPU setup. See `README.md` for the full pipeline CLI flow.

## Coding Style & Naming Conventions
- Python, 4-space indentation, line length 100 (ruff).
- Use `snake_case` for functions/files, `CamelCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep module names aligned to domain areas: `data/` for synthesis/QC, `tools/` for MCP inspection.

## Testing Guidelines
Tests use `pytest` with `pytest-asyncio`. Place unit tests in `tests/unit/` and integration tests in `tests/integration/`. Name tests `test_*.py` and functions `test_*`. Coverage is enforced via `pytest-cov` with an 85% minimum; run:
```bash
pytest --cov
```
Markers: `slow` and `integration` (e.g., `pytest -m "not slow"`).

## Commit & Pull Request Guidelines
Commit messages follow Conventional Commits with optional scope, e.g., `feat(phase-4): add CLI threshold options`. Use `feat`, `fix`, `docs`, `test`, or `chore`. PRs should include a clear summary, the tests run (or why not), and any required environment notes (CUDA/VRAM, `OPENAI_API_KEY`).

## Configuration & Runtime Notes
Set `OPENAI_API_KEY` for seed generation. Local pipeline artifacts are written to `.mcp-forge/`, so avoid committing that directory.
