# Conventions

## Code Style

| Aspect | Convention | Source |
|--------|------------|--------|
| Line Length | 100 characters | `pyproject.toml:67` |
| Python Version | 3.10+ | `pyproject.toml:11` |
| Quote Style | Double quotes | Observed in codebase |
| Indent | 4 spaces | Standard Python |

## Linting & Formatting

Configured in `pyproject.toml:66-76`:

```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]
```

| Rule Set | Purpose |
|----------|---------|
| E | pycodestyle errors |
| F | pyflakes |
| I | isort (import sorting) |
| UP | pyupgrade |
| B | flake8-bugbear |
| SIM | flake8-simplify |

## Type Checking

Configured in `pyproject.toml:74-77`:

```toml
[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true
```

## Naming Conventions

| Element | Convention | Examples |
|---------|------------|----------|
| Classes | PascalCase | `PipelineState`, `ToolDefinition` |
| Functions | snake_case | `validate_dataset`, `inspect_mcp_server` |
| Constants | UPPER_SNAKE | `MCP_TIMEOUT_SECONDS`, `STATE_DIR` |
| Private | Leading underscore | `_run_pipeline`, `_validate_sample` |
| Dataclasses | PascalCase + @dataclass | `QCReport`, `ValidationResult` |
| Enums | PascalCase with UPPER values | `PipelineStage.INSPECTING` |

## Type Annotations

Observed patterns:
- Full type hints on function signatures
- `from __future__ import annotations` for forward refs
- Union types with `|` syntax (3.10+)
- `list[T]` instead of `List[T]`
- `dict[K, V]` instead of `Dict[K, V]`

```python
# Example from inspector.py:41
async def inspect_mcp_server(
    command: str,
    timeout: float = MCP_TIMEOUT_SECONDS
) -> list[ToolDefinition]:
```

## Docstrings

Pattern: Module-level docstrings with version info, brief function docstrings.

```python
# Module level (cli.py:1-4)
"""MCP-Forge CLI entry point.

v1.1: Full command suite including qa, benchmark, pack, and verify-bundle.
"""

# Function level (state.py:70-71)
def schema_hash(self) -> str:
    """Generate hash of the tool schema for drift detection."""
```

## Error Handling

| Pattern | Location | Usage |
|---------|----------|-------|
| Custom exceptions | `inspector.py:21-23` | `MCPInspectorError` |
| Exception chaining | Throughout | `raise X from e` |
| Try/except with cleanup | `cli.py:132-138` | Pipeline error handling |

## Import Organization

Standard order (enforced by ruff/isort):
1. `from __future__ import annotations`
2. Standard library
3. Third-party packages
4. Local imports

## Dataclass Patterns

Consistent use of:
- `@dataclass` decorator
- `field(default_factory=...)` for mutable defaults
- `to_dict()` and `from_dict()` methods for serialization
- Optional fields with `| None` type

```python
@dataclass
class QCReport:
    total_samples: int
    issues: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

## CLI Patterns

Click conventions:
- `@click.group()` for command groups
- `@click.option()` with short flags (`-s`, `-m`)
- `@click.pass_context` for state propagation
- Rich console for styled output
