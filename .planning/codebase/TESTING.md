# Testing

## Test Framework

| Framework | Version | Purpose |
|-----------|---------|---------|
| pytest | >=7.0.0 | Test runner |
| pytest-asyncio | >=0.21.0 | Async test support |
| pytest-cov | >=4.0.0 | Coverage reporting |

## Configuration

From `pyproject.toml:79-81`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

## Current Test State

**No tests exist yet.** The `tests/` directory is referenced in configuration but does not exist in the codebase.

## Test Commands

From `CLAUDE.md`:

```bash
# Run all tests
pytest

# Run a single test
pytest tests/test_module.py::test_function -v

# Run with coverage
pytest --cov=mcp_forge
```

## Planned Test Structure

Based on architecture document:

```
tests/
├── test_cli.py           # CLI command tests
├── test_state.py         # State management tests
├── test_inspector.py     # MCP inspection tests
├── test_qc.py            # QC validation tests
├── test_integration.py   # End-to-end pipeline tests
└── fixtures/
    ├── sample_tools.json     # Test tool definitions
    └── sample_data.jsonl     # Test training data
```

## Testing Patterns Needed

| Area | Type | Priority |
|------|------|----------|
| QC validation | Unit | High |
| Schema validation | Unit | High |
| MCP inspection | Integration | High |
| CLI commands | Integration | Medium |
| Pipeline stages | Integration | Medium |
| State serialization | Unit | Medium |

## Quality Thresholds

From `CLAUDE.md`:

| Metric | Target |
|--------|--------|
| Tool-call parse rate | ≥98% |
| Schema conformance | ≥95% |
| Tool selection accuracy | ≥90% |
| No-tool correctness | ≥85% |
| Loop completion rate | ≥95% |

## Async Testing

The codebase uses asyncio for MCP communication. Tests should use:

```python
import pytest

@pytest.mark.asyncio
async def test_inspect_mcp_server():
    tools = await inspect_mcp_server("npx -y @mcp/server")
    assert len(tools) > 0
```

## CI/CD Considerations

- Deterministic stubs needed for MCP server mocking
- GPU tests may need separate CI pipeline
- Coverage threshold should be established (suggest 85%)
