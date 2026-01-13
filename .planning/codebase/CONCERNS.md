# Concerns

## Critical

### Majority of Pipeline Not Implemented

**Severity:** Critical
**Impact:** Core functionality missing

The codebase has stub implementations for most pipeline stages. Only the framework is in place.

| Stage | Status | Location |
|-------|--------|----------|
| Inspecting | Implemented | `inspector.py` |
| Synthesizing | TODO | `cli.py:171` |
| QC Validating | Partial | `qc.py` exists but not wired |
| Training | TODO | `cli.py:193` |
| Validating | TODO | `cli.py:202` |
| Benchmarking | TODO | `cli.py:211` |
| Exporting | TODO | `cli.py:221` |
| Packaging | TODO | `cli.py:230` |

## High

### No Tests

**Severity:** High
**Impact:** Quality assurance risk

No test files exist despite pytest being configured. This means:
- No regression protection
- No validation of existing code
- CI/CD cannot verify changes

**Location:** `tests/` directory missing

### File-based Tool Loading Not Implemented

**Severity:** High
**Impact:** Feature gap

```python
# cli.py:155
raise NotImplementedError("File-based tool loading coming in v1.2")
```

Users cannot import tools from files, limiting adoption for those without MCP servers.

### QC Module Import Path Mismatch

**Severity:** High
**Impact:** Runtime error

The CLI imports QC from a non-existent path:
```python
# cli.py:409
from .data.qc import DataQualityController, QCConfig
```

But `qc.py` is at root level with different imports:
```python
# qc.py:20
from ..state import QCReport, Scenario, ToolDefinition
```

This will cause `ModuleNotFoundError` at runtime.

## Medium

### Empty Exception Classes

**Severity:** Medium
**Impact:** Missing error context

```python
# inspector.py:21-23
class MCPInspectorError(Exception):
    """Error during MCP server inspection."""
    pass
```

The `pass` statement means no additional context is captured.

### Empty Tools Group

**Severity:** Medium
**Impact:** CLI structure incomplete

```python
# cli.py:351-354
@cli.group()
def tools():
    """Tool management commands."""
    pass
```

The `pass` means no shared logic for tool commands.

### Hardcoded Thresholds

**Severity:** Medium
**Impact:** Inflexibility

Quality thresholds are hardcoded rather than configurable:
- Schema pass rate: 98% (`qc.py:29`)
- Min samples per tool: 10 (`qc.py:30`)

Should be in configuration file.

## Low

### Missing Input Validation

**Severity:** Low
**Impact:** Poor UX on bad input

No validation on CLI inputs before processing:
- `--samples` could be negative
- `--no-tool-ratio` + `--error-ratio` could exceed 1.0
- `--output` path not validated

### Incomplete Error Messages

**Severity:** Low
**Impact:** Debugging difficulty

Some error paths lack detailed context:
```python
# inspector.py:95-96
except Exception as e:
    raise MCPInspectorError(f"Failed to connect to MCP server: {e}")
```

Original exception type is lost.

### Documentation Drift

**Severity:** Low
**Impact:** Developer confusion

`CLAUDE.md` describes module structure that doesn't match actual codebase:
- Says "Flat, Pre-Modularization" but describes `src/mcp_forge/` structure
- `qc.py` location differs from description

## Technical Debt Summary

| Category | Count | Priority |
|----------|-------|----------|
| Not Implemented | 7 TODOs | v1.1 |
| Missing Tests | Entire suite | v1.1 |
| Import Errors | 1 | Immediate |
| Configuration | 2 items | v1.2 |
| Error Handling | 3 items | v1.2 |

## Risk Areas for Future Development

1. **Training Integration:** No Unsloth/LoRA code exists yet - largest implementation risk
2. **CUDA Dependencies:** Training requires GPU - testing will be challenging
3. **MCP Server Reliability:** External process management needs robust error handling
4. **Data Format Evolution:** JSONL schema changes could break existing data
