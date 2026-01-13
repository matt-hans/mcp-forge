# Phase 1: Foundation & Package Structure - Execution Plan

**Phase**: 1 of 9
**Milestone**: v1.0 - Full Pipeline Implementation
**Status**: Ready for Execution
**Created**: 2026-01-12

---

## Objective

Fix the structural issues so the package installs and imports work correctly. Migrate from flat file structure to proper `src/mcp_forge/` package layout.

---

## Execution Context

**Files to Read First:**
- `pyproject.toml` - Current build configuration
- `TECHNICAL_ARCHITECTURE_v1.1.md` - Target structure
- `.planning/codebase/STRUCTURE.md` - Structure documentation

**Key Constraints:**
- No functional changes to existing code logic
- Preserve all existing functionality
- Must pass `pip install -e .` after migration
- CLI entry points must work (`mcp-forge --help`)

---

## Context

### Current State (Flat Structure)
```
mcp-forge/
├── cli.py           # Uses relative imports: from . import __version__
├── state.py         # Standalone, no internal imports
├── inspector.py     # Uses: from .state import ToolDefinition
├── qc.py            # Uses: from ..state import QCReport, Scenario, ToolDefinition
├── pyproject.toml   # Already points to src/mcp_forge
```

### Target State (Package Structure)
```
mcp-forge/
├── src/
│   └── mcp_forge/
│       ├── __init__.py          # Package init with __version__
│       ├── cli.py               # CLI entry point
│       ├── state.py             # Core state management
│       ├── tools/
│       │   ├── __init__.py
│       │   └── inspector.py     # MCP server inspection
│       └── data/
│           ├── __init__.py
│           └── qc.py            # Quality control engine
├── tests/                        # Empty placeholder
├── pyproject.toml               # Already configured correctly
```

### Import Analysis
| File | Current Import | Target Import |
|------|----------------|---------------|
| `cli.py:17` | `from . import __version__` | `from mcp_forge import __version__` (or relative) |
| `cli.py:18-22` | `from .state import ...` | `from mcp_forge.state import ...` (or relative) |
| `cli.py:143` | `from .inspector import ...` | `from mcp_forge.tools.inspector import ...` |
| `cli.py:409` | `from .data.qc import ...` | Correct path (after move) |
| `inspector.py:13` | `from .state import ToolDefinition` | `from mcp_forge.state import ToolDefinition` |
| `qc.py:20` | `from ..state import ...` | `from mcp_forge.state import ...` |

---

## Tasks

### Task 1: Create Directory Structure
**Action**: Create the `src/mcp_forge/` directory tree

```bash
mkdir -p src/mcp_forge/tools
mkdir -p src/mcp_forge/data
mkdir -p tests
```

**Verification**: `ls -la src/mcp_forge/`

---

### Task 2: Create Package `__init__.py`
**Action**: Create `src/mcp_forge/__init__.py` with version

```python
"""MCP-Forge: Fine-tune local LLMs on MCP server tool schemas."""

__version__ = "0.1.0"
__all__ = ["__version__"]
```

**Verification**: `python -c "from mcp_forge import __version__; print(__version__)"`

---

### Task 3: Migrate `state.py`
**Action**: Move `state.py` to `src/mcp_forge/state.py`

- No import changes needed (standalone module)
- File contains: `PipelineStage`, `ToolDefinition`, `QCReport`, `ValidationResult`, `BenchmarkResult`, `SynthesisPlan`, `PipelineState`, `StateManager`

**Verification**: `python -c "from mcp_forge.state import PipelineState, StateManager"`

---

### Task 4: Migrate `inspector.py` to Tools Package
**Action**: Move `inspector.py` to `src/mcp_forge/tools/inspector.py`

**Create**: `src/mcp_forge/tools/__init__.py`
```python
"""Tool provider layer for MCP and file-based tool sources."""

from .inspector import (
    inspect_mcp_server,
    format_tool_for_display,
    validate_tool_call,
    MCPInspectorError,
)

__all__ = [
    "inspect_mcp_server",
    "format_tool_for_display",
    "validate_tool_call",
    "MCPInspectorError",
]
```

**Update import in `inspector.py`** (line 13):
```python
# FROM:
from .state import ToolDefinition
# TO:
from mcp_forge.state import ToolDefinition
```

**Verification**: `python -c "from mcp_forge.tools import inspect_mcp_server"`

---

### Task 5: Migrate `qc.py` to Data Package
**Action**: Move `qc.py` to `src/mcp_forge/data/qc.py`

**Create**: `src/mcp_forge/data/__init__.py`
```python
"""Data synthesis and quality control layer."""

from .qc import (
    DataQualityController,
    QCConfig,
    QCIssue,
    ValidatedSample,
)

__all__ = [
    "DataQualityController",
    "QCConfig",
    "QCIssue",
    "ValidatedSample",
]
```

**Update import in `qc.py`** (line 20):
```python
# FROM:
from ..state import QCReport, Scenario, ToolDefinition
# TO:
from mcp_forge.state import QCReport, Scenario, ToolDefinition
```

**Verification**: `python -c "from mcp_forge.data import DataQualityController"`

---

### Task 6: Migrate `cli.py`
**Action**: Move `cli.py` to `src/mcp_forge/cli.py`

**Update imports** (lines 17-22):
```python
# FROM:
from . import __version__
from .state import (
    PipelineStage,
    StateManager,
    SynthesisPlan,
)

# TO:
from mcp_forge import __version__
from mcp_forge.state import (
    PipelineStage,
    StateManager,
    SynthesisPlan,
)
```

**Update import in `_run_pipeline`** (line 143):
```python
# FROM:
from .inspector import inspect_mcp_server
# TO:
from mcp_forge.tools.inspector import inspect_mcp_server
```

**Update import in `qa` command** (lines 408-410):
```python
# FROM:
from .data.qc import DataQualityController, QCConfig
from .state import ToolDefinition
# TO:
from mcp_forge.data.qc import DataQualityController, QCConfig
from mcp_forge.state import ToolDefinition
```

**Verification**: `python -c "from mcp_forge.cli import cli"`

---

### Task 7: Clean Up Root Directory
**Action**: Remove migrated files from root

```bash
rm cli.py state.py inspector.py qc.py
```

**Keep**: `pyproject.toml`, `CLAUDE.md`, `README.md`, `TECHNICAL_ARCHITECTURE_v1.1.md`

---

### Task 8: Install Package in Editable Mode
**Action**: Install the package

```bash
pip install -e ".[dev]"
```

**Expected Output**: Successful installation with no errors

---

### Task 9: Verify CLI Entry Points
**Action**: Test all CLI commands work

```bash
mcp-forge --version
mcp-forge --help
mcp-forge doctor
mcp-forge status
```

**Expected**:
- `--version` shows `0.1.0`
- `--help` shows all commands
- `doctor` runs environment check
- `status` shows "No active session found."

---

### Task 10: Run Linting
**Action**: Verify code quality

```bash
ruff check src/
mypy src/mcp_forge/
```

**Expected**: No errors (or only pre-existing issues)

---

## Verification Checklist

After all tasks complete:

- [ ] `pip install -e .` succeeds without errors
- [ ] `mcp-forge --help` displays available commands
- [ ] `mcp-forge --version` shows `0.1.0`
- [ ] `mcp-forge doctor` runs and reports environment status
- [ ] `mcp-forge status` runs without import errors
- [ ] All Python files migrated to `src/mcp_forge/`
- [ ] Root directory only has config/doc files
- [ ] `ruff check src/` passes
- [ ] `mypy src/mcp_forge/` passes (or shows only pre-existing issues)

---

## Success Criteria

From ROADMAP.md Phase 1:

- [x] Define acceptance criteria (this plan)
- [ ] `pip install -e .` succeeds without errors
- [ ] `mcp-forge --help` displays available commands
- [ ] `mcp-forge doctor` runs and reports environment status
- [ ] All existing code migrated with no functional changes
- [ ] ruff and mypy pass on migrated code

---

## Output

**Artifacts Created:**
- `src/mcp_forge/__init__.py`
- `src/mcp_forge/cli.py`
- `src/mcp_forge/state.py`
- `src/mcp_forge/tools/__init__.py`
- `src/mcp_forge/tools/inspector.py`
- `src/mcp_forge/data/__init__.py`
- `src/mcp_forge/data/qc.py`
- `tests/` (empty directory placeholder)

**Files Removed:**
- `cli.py` (root)
- `state.py` (root)
- `inspector.py` (root)
- `qc.py` (root)

---

## Rollback Plan

If migration fails:
1. `git checkout -- .` to restore original files
2. Identify specific import that failed
3. Fix import path and retry

---

## Notes

- The `pyproject.toml` is already correctly configured for `src/mcp_forge/` layout
- No changes needed to `pyproject.toml`
- `jsonschema` dependency is used in `qc.py` but not listed in dependencies - may need to add
- The `qc.py` file in root uses `from ..state` which suggests it was already written expecting to be in a subpackage - this is a pre-existing bug

---

*Plan created: 2026-01-12*
