# Phase 2: Test Infrastructure - Execution Plan

**Phase**: 2 of 9
**Milestone**: v1.0 - Full Pipeline Implementation
**Status**: Ready for Execution
**Created**: 2026-01-13

---

## Objective

Establish a comprehensive test foundation before adding new features. Create test directory structure, configure pytest with async support, and write tests for all existing modules (state, inspector, qc).

---

## Execution Context

**Files to Read First:**
- `pyproject.toml` - Test configuration (pytest.ini_options)
- `src/mcp_forge/state.py` - Core state management (dataclasses, StateManager)
- `src/mcp_forge/tools/inspector.py` - MCP inspection (async functions)
- `src/mcp_forge/data/qc.py` - QC validation (DataQualityController)

**Key Constraints:**
- Use pytest-asyncio for async tests (already in dev dependencies)
- Target 85%+ coverage for existing code
- No external MCP server calls in unit tests - use mocks
- Tests must be deterministic and reproducible
- Follow existing code conventions (typing, docstrings)

**Dependencies:**
- pytest>=7.0.0 (dev dependency)
- pytest-asyncio>=0.21.0 (dev dependency)
- pytest-cov>=4.0.0 (dev dependency)

---

## Context

### Modules to Test

| Module | File | Key Components | Test Priority |
|--------|------|----------------|---------------|
| State | `state.py` | 7 dataclasses, StateManager, serialization | High |
| Inspector | `tools/inspector.py` | `inspect_mcp_server`, `validate_tool_call`, parsing | High |
| QC | `data/qc.py` | `DataQualityController`, schema validation, dedup | High |
| CLI | `cli.py` | Commands, integration | Medium |

### State Module Components (512 lines)

| Class | Lines | Test Focus |
|-------|-------|------------|
| `PipelineStage` | 22-36 | Enum values, string conversion |
| `Scenario` | 39-46 | Enum values |
| `ToolDefinition` | 49-73 | `to_dict`, `from_dict`, `schema_hash` |
| `QCReport` | 76-103 | `passes_threshold` logic |
| `ValidationResult` | 106-137 | `meets_release_criteria` |
| `BenchmarkResult` | 140-159 | Serialization |
| `SynthesisPlan` | 162-192 | `get_samples_per_scenario` calculation |
| `PipelineState` | 195-334 | Full serialization roundtrip, `compute_toolset_hash` |
| `StateManager` | 337-511 | File operations, atomic save, report paths |

### Inspector Module Components (196 lines)

| Function | Lines | Test Focus |
|----------|-------|------------|
| `parse_command` | 26-38 | Command parsing, edge cases |
| `inspect_mcp_server` | 41-96 | Mocked MCP client, timeout handling, error cases |
| `format_tool_for_display` | 99-124 | Output formatting |
| `generate_tool_use_prompt` | 127-151 | Prompt generation |
| `validate_tool_call` | 154-195 | Validation logic, required params |

### QC Module Components (438 lines)

| Class/Function | Lines | Test Focus |
|----------------|-------|------------|
| `QCConfig` | 25-42 | Default values |
| `QCIssue` | 45-64 | Serialization |
| `ValidatedSample` | 67-86 | Serialization |
| `DataQualityController.validate_dataset` | 106-192 | Full validation flow |
| `_validate_sample` | 194-291 | Sample validation logic |
| `_extract_tool_call` | 293-324 | Tool call parsing from messages |
| `_validate_schema` | 326-334 | JSON schema validation |
| `_check_coverage` | 349-379 | Coverage requirements |

---

## Tasks

### Task 1: Create Test Directory Structure

**Action**: Create the `tests/` directory tree with proper organization

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_state.py     # State dataclasses and StateManager
│   ├── test_inspector.py # Inspector functions (mocked)
│   └── test_qc.py        # QC validation logic
├── integration/
│   ├── __init__.py
│   └── test_cli.py       # CLI command tests
└── fixtures/
    ├── sample_tools.json     # Test tool definitions
    └── sample_data.jsonl     # Test training data samples
```

**Commands**:
```bash
mkdir -p tests/unit tests/integration tests/fixtures
touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py
```

**Verification**: `ls -la tests/`

---

### Task 2: Create Test Fixtures

**Action**: Create `tests/fixtures/sample_tools.json` with test tool definitions

```json
[
  {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "input_schema": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "City name"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "default": "celsius"
        }
      },
      "required": ["location"]
    }
  },
  {
    "name": "search_files",
    "description": "Search for files matching a pattern",
    "input_schema": {
      "type": "object",
      "properties": {
        "pattern": {
          "type": "string",
          "description": "Glob pattern to match"
        },
        "directory": {
          "type": "string",
          "description": "Directory to search in"
        }
      },
      "required": ["pattern"]
    }
  },
  {
    "name": "calculate",
    "description": "Perform a mathematical calculation",
    "input_schema": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "Mathematical expression to evaluate"
        }
      },
      "required": ["expression"]
    }
  }
]
```

**Action**: Create `tests/fixtures/sample_data.jsonl` with test training samples

```jsonl
{"id": "sample_001", "source": "seed", "scenario": "standard", "tool_name": "get_weather", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What's the weather in Paris?"}, {"role": "assistant", "content": "```json\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n```"}]}
{"id": "sample_002", "source": "seed", "scenario": "no_tool", "tool_name": null, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is 2 + 2?"}, {"role": "assistant", "content": "2 + 2 equals 4."}]}
{"id": "sample_003", "source": "augmented", "scenario": "standard", "tool_name": "search_files", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Find all Python files"}, {"role": "assistant", "content": "```json\n{\"name\": \"search_files\", \"arguments\": {\"pattern\": \"*.py\"}}\n```"}]}
{"id": "sample_004", "source": "seed", "scenario": "error", "tool_name": "get_weather", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Weather in InvalidCity123"}, {"role": "assistant", "content": "```json\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"InvalidCity123\"}}\n```"}]}
{"id": "sample_005", "source": "augmented", "scenario": "standard", "tool_name": "calculate", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Calculate the square root of 144"}, {"role": "assistant", "content": "```json\n{\"name\": \"calculate\", \"arguments\": {\"expression\": \"sqrt(144)\"}}\n```"}]}
```

**Verification**: `cat tests/fixtures/sample_tools.json | python -m json.tool`

---

### Task 3: Create Shared Test Configuration (conftest.py)

**Action**: Create `tests/conftest.py` with shared fixtures and pytest configuration

```python
"""Shared test fixtures and configuration for mcp-forge tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_forge.state import (
    PipelineStage,
    PipelineState,
    QCReport,
    Scenario,
    StateManager,
    SynthesisPlan,
    ToolDefinition,
    ValidationResult,
)


# =============================================================================
# Path Fixtures
# =============================================================================

@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_tools_path(fixtures_dir: Path) -> Path:
    """Return path to sample tools JSON file."""
    return fixtures_dir / "sample_tools.json"


@pytest.fixture
def sample_data_path(fixtures_dir: Path) -> Path:
    """Return path to sample training data JSONL file."""
    return fixtures_dir / "sample_data.jsonl"


# =============================================================================
# Tool Definition Fixtures
# =============================================================================

@pytest.fixture
def sample_tools(sample_tools_path: Path) -> list[ToolDefinition]:
    """Load sample tool definitions from fixture file."""
    with open(sample_tools_path) as f:
        data = json.load(f)
    return [ToolDefinition.from_dict(t) for t in data]


@pytest.fixture
def weather_tool() -> ToolDefinition:
    """Single weather tool for focused tests."""
    return ToolDefinition(
        name="get_weather",
        description="Get current weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    )


@pytest.fixture
def tool_with_no_required() -> ToolDefinition:
    """Tool with no required parameters."""
    return ToolDefinition(
        name="optional_tool",
        description="Tool with all optional params",
        input_schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "integer"},
            },
        },
    )


# =============================================================================
# State Fixtures
# =============================================================================

@pytest.fixture
def temp_state_dir(tmp_path: Path) -> Path:
    """Temporary directory for state tests."""
    return tmp_path


@pytest.fixture
def state_manager(temp_state_dir: Path) -> StateManager:
    """StateManager with temporary directory."""
    return StateManager(base_path=temp_state_dir)


@pytest.fixture
def sample_state(state_manager: StateManager) -> PipelineState:
    """Create a sample pipeline state for testing."""
    return state_manager.create_session(
        mcp_command="npx -y @test/server",
        system_prompt="You are a helpful assistant.",
        model_family="deepseek-r1",
        output_path="./test-output",
        quantization="q8_0",
        profile="balanced",
    )


@pytest.fixture
def synthesis_plan() -> SynthesisPlan:
    """Sample synthesis plan."""
    return SynthesisPlan(
        total_samples=100,
        seed_samples=20,
        augmented_samples=80,
        scenario_weights={
            "standard": 0.60,
            "no_tool": 0.15,
            "error": 0.10,
            "ambiguous": 0.10,
            "edge": 0.05,
        },
    )


@pytest.fixture
def qc_report() -> QCReport:
    """Sample QC report for testing."""
    return QCReport(
        total_samples=100,
        valid_samples=95,
        dropped_samples=5,
        schema_pass_rate=0.98,
        dedup_rate=0.02,
        tool_coverage={"get_weather": 40, "search_files": 35, "calculate": 20},
        scenario_coverage={
            "standard": 60,
            "no_tool": 15,
            "error": 10,
            "ambiguous": 10,
            "edge": 5,
        },
        issues=[],
    )


@pytest.fixture
def validation_result() -> ValidationResult:
    """Sample validation result for testing."""
    return ValidationResult(
        passed=True,
        samples_tested=20,
        samples_passed=19,
        tool_call_parse_rate=0.99,
        schema_conformance_rate=0.98,
        tool_selection_accuracy=0.95,
        loop_completion_rate=0.97,
        error_handling_rate=0.90,
    )


# =============================================================================
# MCP Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_mcp_session() -> MagicMock:
    """Mock MCP ClientSession for inspector tests."""
    session = MagicMock()
    session.initialize = AsyncMock()

    # Create mock tool objects
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    # Mock tools response
    tools_response = MagicMock()
    tools_response.tools = [mock_tool]
    session.list_tools = AsyncMock(return_value=tools_response)

    return session


@pytest.fixture
def mock_mcp_tools() -> list[dict[str, Any]]:
    """Raw MCP tool data for mocking responses."""
    return [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        },
        {
            "name": "search_files",
            "description": "Search for files",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    ]


# =============================================================================
# Data Fixtures
# =============================================================================

@pytest.fixture
def valid_training_sample() -> dict[str, Any]:
    """A valid training sample dictionary."""
    return {
        "id": "test_001",
        "source": "seed",
        "scenario": "standard",
        "tool_name": "get_weather",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": '```json\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n```',
            },
        ],
    }


@pytest.fixture
def invalid_training_sample() -> dict[str, Any]:
    """A training sample with validation errors."""
    return {
        "id": "test_bad",
        "source": "seed",
        "scenario": "standard",
        "tool_name": "get_weather",
        "messages": [
            {"role": "user", "content": "Weather please"},
            # Missing tool call in assistant response
            {"role": "assistant", "content": "Let me check the weather for you."},
        ],
    }


@pytest.fixture
def temp_jsonl_file(tmp_path: Path, valid_training_sample: dict[str, Any]) -> Path:
    """Create a temporary JSONL file with sample data."""
    path = tmp_path / "test_data.jsonl"
    with open(path, "w") as f:
        # Write multiple samples
        for i in range(10):
            sample = valid_training_sample.copy()
            sample["id"] = f"sample_{i:03d}"
            f.write(json.dumps(sample) + "\n")
    return path
```

**Verification**: `python -c "import tests.conftest"`

---

### Task 4: Write State Module Tests

**Action**: Create `tests/unit/test_state.py`

```python
"""Tests for mcp_forge.state module."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from mcp_forge.state import (
    BenchmarkResult,
    PipelineStage,
    PipelineState,
    QCReport,
    Scenario,
    StateManager,
    SynthesisPlan,
    ToolDefinition,
    ValidationResult,
)


class TestPipelineStage:
    """Tests for PipelineStage enum."""

    def test_all_stages_defined(self) -> None:
        """Verify all expected pipeline stages exist."""
        expected = [
            "idle", "inspecting", "synthesizing", "qc_validating",
            "formatting", "training", "validating", "benchmarking",
            "exporting", "packaging", "complete", "failed",
        ]
        actual = [s.value for s in PipelineStage]
        assert set(actual) == set(expected)

    def test_stage_string_conversion(self) -> None:
        """Test stage value is accessible as string."""
        assert PipelineStage.INSPECTING.value == "inspecting"
        assert PipelineStage.COMPLETE.value == "complete"


class TestScenario:
    """Tests for Scenario enum."""

    def test_all_scenarios_defined(self) -> None:
        """Verify all scenario types exist."""
        expected = ["standard", "no_tool", "error", "ambiguous", "edge"]
        actual = [s.value for s in Scenario]
        assert set(actual) == set(expected)


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_to_dict(self, weather_tool: ToolDefinition) -> None:
        """Test serialization to dictionary."""
        result = weather_tool.to_dict()
        assert result["name"] == "get_weather"
        assert result["description"] == "Get current weather for a location"
        assert "input_schema" in result
        assert result["source"] == "mcp"

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "name": "test_tool",
            "description": "Test description",
            "input_schema": {"type": "object"},
            "source": "file",
        }
        tool = ToolDefinition.from_dict(data)
        assert tool.name == "test_tool"
        assert tool.source == "file"

    def test_from_dict_default_source(self) -> None:
        """Test default source is 'mcp' when not specified."""
        data = {
            "name": "test_tool",
            "description": "Test",
            "input_schema": {},
        }
        tool = ToolDefinition.from_dict(data)
        assert tool.source == "mcp"

    def test_schema_hash_deterministic(self, weather_tool: ToolDefinition) -> None:
        """Test schema hash is deterministic."""
        hash1 = weather_tool.schema_hash()
        hash2 = weather_tool.schema_hash()
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

    def test_schema_hash_different_for_different_schemas(self) -> None:
        """Test different schemas produce different hashes."""
        tool1 = ToolDefinition(
            name="tool1",
            description="",
            input_schema={"type": "object", "properties": {"a": {"type": "string"}}},
        )
        tool2 = ToolDefinition(
            name="tool2",
            description="",
            input_schema={"type": "object", "properties": {"b": {"type": "integer"}}},
        )
        assert tool1.schema_hash() != tool2.schema_hash()


class TestQCReport:
    """Tests for QCReport dataclass."""

    def test_to_dict(self, qc_report: QCReport) -> None:
        """Test serialization."""
        result = qc_report.to_dict()
        assert result["total_samples"] == 100
        assert result["schema_pass_rate"] == 0.98

    def test_from_dict_roundtrip(self, qc_report: QCReport) -> None:
        """Test serialization roundtrip."""
        data = qc_report.to_dict()
        restored = QCReport.from_dict(data)
        assert restored.total_samples == qc_report.total_samples
        assert restored.schema_pass_rate == qc_report.schema_pass_rate

    def test_passes_threshold_success(self) -> None:
        """Test passing quality thresholds."""
        report = QCReport(
            total_samples=100,
            valid_samples=98,
            dropped_samples=2,
            schema_pass_rate=0.99,
            dedup_rate=0.01,
            tool_coverage={"tool1": 50, "tool2": 48},
            scenario_coverage={},
        )
        assert report.passes_threshold(min_schema_rate=0.98, min_per_tool=10)

    def test_passes_threshold_schema_fail(self) -> None:
        """Test failing due to low schema pass rate."""
        report = QCReport(
            total_samples=100,
            valid_samples=90,
            dropped_samples=10,
            schema_pass_rate=0.90,  # Below 0.98 threshold
            dedup_rate=0.0,
            tool_coverage={"tool1": 50, "tool2": 50},
            scenario_coverage={},
        )
        assert not report.passes_threshold(min_schema_rate=0.98, min_per_tool=10)

    def test_passes_threshold_coverage_fail(self) -> None:
        """Test failing due to low tool coverage."""
        report = QCReport(
            total_samples=100,
            valid_samples=98,
            dropped_samples=2,
            schema_pass_rate=0.99,
            dedup_rate=0.0,
            tool_coverage={"tool1": 50, "tool2": 5},  # tool2 below min
            scenario_coverage={},
        )
        assert not report.passes_threshold(min_schema_rate=0.98, min_per_tool=10)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_meets_release_criteria_success(self) -> None:
        """Test meeting all release criteria."""
        result = ValidationResult(
            passed=True,
            samples_tested=100,
            samples_passed=98,
            tool_call_parse_rate=0.99,
            schema_conformance_rate=0.97,
            tool_selection_accuracy=0.92,
            loop_completion_rate=0.96,
        )
        assert result.meets_release_criteria()

    def test_meets_release_criteria_fail_parse_rate(self) -> None:
        """Test failing parse rate threshold."""
        result = ValidationResult(
            passed=False,
            samples_tested=100,
            samples_passed=95,
            tool_call_parse_rate=0.95,  # Below 0.98
            schema_conformance_rate=0.97,
            tool_selection_accuracy=0.92,
            loop_completion_rate=0.96,
        )
        assert not result.meets_release_criteria()

    def test_meets_release_criteria_fail_accuracy(self) -> None:
        """Test failing accuracy threshold."""
        result = ValidationResult(
            passed=False,
            samples_tested=100,
            samples_passed=85,
            tool_call_parse_rate=0.99,
            schema_conformance_rate=0.97,
            tool_selection_accuracy=0.85,  # Below 0.90
            loop_completion_rate=0.96,
        )
        assert not result.meets_release_criteria()


class TestSynthesisPlan:
    """Tests for SynthesisPlan dataclass."""

    def test_get_samples_per_scenario(self, synthesis_plan: SynthesisPlan) -> None:
        """Test scenario sample calculation."""
        samples = synthesis_plan.get_samples_per_scenario()
        assert samples["standard"] == 60  # 100 * 0.60
        assert samples["no_tool"] == 15   # 100 * 0.15
        assert samples["error"] == 10     # 100 * 0.10
        assert samples["ambiguous"] == 10 # 100 * 0.10
        assert samples["edge"] == 5       # 100 * 0.05

    def test_default_scenario_weights(self) -> None:
        """Test default scenario weights are set."""
        plan = SynthesisPlan(
            total_samples=100,
            seed_samples=20,
            augmented_samples=80,
        )
        assert plan.scenario_weights["standard"] == 0.60
        assert plan.scenario_weights["no_tool"] == 0.15


class TestPipelineState:
    """Tests for PipelineState dataclass."""

    def test_to_dict_from_dict_roundtrip(
        self,
        sample_state: PipelineState,
        weather_tool: ToolDefinition,
        qc_report: QCReport,
    ) -> None:
        """Test full serialization roundtrip."""
        sample_state.tools = [weather_tool]
        sample_state.qc_report = qc_report

        data = sample_state.to_dict()
        restored = PipelineState.from_dict(data)

        assert restored.session_id == sample_state.session_id
        assert restored.stage == sample_state.stage
        assert restored.model_family == sample_state.model_family
        assert len(restored.tools) == 1
        assert restored.tools[0].name == "get_weather"
        assert restored.qc_report is not None
        assert restored.qc_report.total_samples == 100

    def test_update_stage(self, sample_state: PipelineState) -> None:
        """Test stage update also updates timestamp."""
        old_updated = sample_state.updated_at
        sample_state.update_stage(PipelineStage.INSPECTING)

        assert sample_state.stage == PipelineStage.INSPECTING
        assert sample_state.updated_at != old_updated

    def test_set_error(self, sample_state: PipelineState) -> None:
        """Test error setting changes stage to FAILED."""
        sample_state.set_error("Test error message")

        assert sample_state.error == "Test error message"
        assert sample_state.stage == PipelineStage.FAILED

    def test_compute_toolset_hash_empty(self, sample_state: PipelineState) -> None:
        """Test toolset hash returns empty string when no tools."""
        sample_state.tools = []
        assert sample_state.compute_toolset_hash() == ""

    def test_compute_toolset_hash_deterministic(
        self,
        sample_state: PipelineState,
        sample_tools: list[ToolDefinition],
    ) -> None:
        """Test toolset hash is deterministic."""
        sample_state.tools = sample_tools
        hash1 = sample_state.compute_toolset_hash()
        hash2 = sample_state.compute_toolset_hash()
        assert hash1 == hash2
        assert len(hash1) == 16


class TestStateManager:
    """Tests for StateManager class."""

    def test_ensure_dirs_creates_structure(self, state_manager: StateManager) -> None:
        """Test directory creation."""
        state_manager.ensure_dirs()

        assert state_manager.state_dir.exists()
        assert state_manager.data_dir.exists()
        assert state_manager.logs_dir.exists()
        assert state_manager.reports_dir.exists()

    def test_create_session(self, state_manager: StateManager) -> None:
        """Test session creation."""
        state = state_manager.create_session(
            mcp_command="test command",
            system_prompt="test prompt",
            model_family="deepseek-r1",
            output_path="./output",
        )

        assert state.session_id is not None
        assert len(state.session_id) == 8
        assert state.stage == PipelineStage.IDLE
        assert state.model_family == "deepseek-r1"

    def test_save_and_load_state(self, state_manager: StateManager) -> None:
        """Test state persistence roundtrip."""
        state = state_manager.create_session(
            mcp_command="test",
            system_prompt="prompt",
            model_family="qwen-2.5",
            output_path="./out",
        )
        state.update_stage(PipelineStage.TRAINING)
        state_manager.save_state(state)

        loaded = state_manager.load_state()

        assert loaded is not None
        assert loaded.session_id == state.session_id
        assert loaded.stage == PipelineStage.TRAINING

    def test_load_state_no_file(self, state_manager: StateManager) -> None:
        """Test loading when no state file exists."""
        state_manager.ensure_dirs()
        result = state_manager.load_state()
        assert result is None

    def test_clear_state(self, state_manager: StateManager) -> None:
        """Test state clearing."""
        state_manager.create_session(
            mcp_command="test",
            system_prompt="prompt",
            model_family="deepseek-r1",
            output_path="./out",
        )

        state_manager.clear_state()

        assert not state_manager.state_file.exists()
        assert state_manager.load_state() is None

    def test_can_resume_no_state(self, state_manager: StateManager) -> None:
        """Test can_resume returns False when no state."""
        state_manager.ensure_dirs()
        assert not state_manager.can_resume()

    def test_can_resume_idle_state(self, state_manager: StateManager) -> None:
        """Test can_resume returns False for IDLE state."""
        state_manager.create_session(
            mcp_command="test",
            system_prompt="prompt",
            model_family="deepseek-r1",
            output_path="./out",
        )
        assert not state_manager.can_resume()

    def test_can_resume_active_state(self, state_manager: StateManager) -> None:
        """Test can_resume returns True for in-progress state."""
        state = state_manager.create_session(
            mcp_command="test",
            system_prompt="prompt",
            model_family="deepseek-r1",
            output_path="./out",
        )
        state.update_stage(PipelineStage.TRAINING)
        state_manager.save_state(state)

        assert state_manager.can_resume()

    def test_get_data_path(self, state_manager: StateManager) -> None:
        """Test data path generation."""
        path = state_manager.get_data_path("train.jsonl")
        assert path.name == "train.jsonl"
        assert "data" in str(path)

    def test_get_report_path(self, state_manager: StateManager) -> None:
        """Test report path generation."""
        path = state_manager.get_report_path("qc_report.json")
        assert path.name == "qc_report.json"
        assert "reports" in str(path)

    def test_save_qc_report(
        self,
        state_manager: StateManager,
        qc_report: QCReport,
    ) -> None:
        """Test QC report saving."""
        path = state_manager.save_qc_report(qc_report)

        assert path.exists()
        assert "qc_" in path.name

        with open(path) as f:
            data = json.load(f)
        assert data["total_samples"] == 100

    def test_save_benchmark_result(self, state_manager: StateManager) -> None:
        """Test benchmark result saving creates both JSON and Markdown."""
        result = BenchmarkResult(
            model_name="test-model",
            timestamp=datetime.utcnow().isoformat(),
            overall_score=0.92,
            per_tool_results={"tool1": {"accuracy": 0.95, "schema": 0.98, "latency": 1.2}},
            per_scenario_results={"standard": {"pass_rate": 0.94}},
        )

        json_path = state_manager.save_benchmark_result(result)
        md_path = json_path.with_suffix(".md")

        assert json_path.exists()
        assert md_path.exists()

        # Verify markdown content
        md_content = md_path.read_text()
        assert "test-model" in md_content
        assert "92" in md_content or "0.92" in md_content
```

**Verification**: `pytest tests/unit/test_state.py -v`

---

### Task 5: Write Inspector Module Tests

**Action**: Create `tests/unit/test_inspector.py`

```python
"""Tests for mcp_forge.tools.inspector module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_forge.state import ToolDefinition
from mcp_forge.tools.inspector import (
    MCPInspectorError,
    format_tool_for_display,
    generate_tool_use_prompt,
    parse_command,
    validate_tool_call,
)


class TestParseCommand:
    """Tests for parse_command function."""

    def test_simple_command(self) -> None:
        """Test parsing a simple command."""
        exe, args = parse_command("python script.py")
        assert exe == "python"
        assert args == ["script.py"]

    def test_command_with_multiple_args(self) -> None:
        """Test parsing command with multiple arguments."""
        exe, args = parse_command("npx -y @mcp/server-weather")
        assert exe == "npx"
        assert args == ["-y", "@mcp/server-weather"]

    def test_command_no_args(self) -> None:
        """Test parsing command with no arguments."""
        exe, args = parse_command("node")
        assert exe == "node"
        assert args == []

    def test_command_with_quotes(self) -> None:
        """Test parsing command with quoted arguments."""
        exe, args = parse_command('python -c "print(1)"')
        assert exe == "python"
        assert args == ["-c", "print(1)"]

    def test_empty_command_raises(self) -> None:
        """Test empty command raises MCPInspectorError."""
        with pytest.raises(MCPInspectorError, match="Empty command"):
            parse_command("")

    def test_uv_run_command(self) -> None:
        """Test parsing uv run command."""
        exe, args = parse_command("uv run my_server --port 8080")
        assert exe == "uv"
        assert args == ["run", "my_server", "--port", "8080"]


class TestFormatToolForDisplay:
    """Tests for format_tool_for_display function."""

    def test_basic_formatting(self, weather_tool: ToolDefinition) -> None:
        """Test basic tool formatting."""
        result = format_tool_for_display(weather_tool)
        assert "get_weather" in result
        assert "location" in result

    def test_truncates_long_description(self) -> None:
        """Test long descriptions are truncated."""
        tool = ToolDefinition(
            name="test",
            description="A" * 150,  # Very long description
            input_schema={},
        )
        result = format_tool_for_display(tool)
        assert "..." in result
        assert len(result) < 200

    def test_shows_required_marker(self, weather_tool: ToolDefinition) -> None:
        """Test required parameters are marked."""
        result = format_tool_for_display(weather_tool)
        assert "location*" in result  # Required param has asterisk

    def test_empty_schema(self) -> None:
        """Test formatting tool with empty schema."""
        tool = ToolDefinition(name="empty", description="Empty tool", input_schema={})
        result = format_tool_for_display(tool)
        assert "empty" in result


class TestGenerateToolUsePrompt:
    """Tests for generate_tool_use_prompt function."""

    def test_generates_prompt_with_tools(self, sample_tools: list[ToolDefinition]) -> None:
        """Test prompt generation with multiple tools."""
        prompt = generate_tool_use_prompt(sample_tools)

        assert "get_weather" in prompt
        assert "search_files" in prompt
        assert "calculate" in prompt
        assert "input schema" in prompt.lower()

    def test_empty_tools_message(self) -> None:
        """Test message when no tools available."""
        prompt = generate_tool_use_prompt([])
        assert "No tools are available" in prompt

    def test_includes_tool_descriptions(self, sample_tools: list[ToolDefinition]) -> None:
        """Test tool descriptions are included."""
        prompt = generate_tool_use_prompt(sample_tools)
        assert "current weather" in prompt.lower()
        assert "Search for files" in prompt


class TestValidateToolCall:
    """Tests for validate_tool_call function."""

    def test_valid_tool_call(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation of correct tool call."""
        tool_call = {
            "name": "get_weather",
            "arguments": {"location": "Paris"},
        }
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert is_valid
        assert error is None

    def test_missing_name_field(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation fails when name is missing."""
        tool_call = {"arguments": {"location": "Paris"}}
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert not is_valid
        assert "missing 'name'" in error.lower()

    def test_unknown_tool_name(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation fails for unknown tool."""
        tool_call = {
            "name": "unknown_tool",
            "arguments": {},
        }
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert not is_valid
        assert "Unknown tool" in error

    def test_missing_required_parameter(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation fails when required param missing."""
        tool_call = {
            "name": "get_weather",
            "arguments": {},  # Missing required 'location'
        }
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert not is_valid
        assert "Missing required parameter" in error
        assert "location" in error

    def test_optional_parameter_not_required(self, sample_tools: list[ToolDefinition]) -> None:
        """Test optional parameters don't cause validation failure."""
        tool_call = {
            "name": "search_files",
            "arguments": {"pattern": "*.py"},  # 'directory' is optional
        }
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert is_valid

    def test_non_dict_tool_call(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation fails for non-dict input."""
        is_valid, error = validate_tool_call("not a dict", sample_tools)
        assert not is_valid
        assert "must be a dictionary" in error

    def test_non_dict_arguments(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation fails for non-dict arguments."""
        tool_call = {
            "name": "get_weather",
            "arguments": "not a dict",
        }
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert not is_valid
        assert "arguments" in error.lower()

    def test_tool_with_no_required_params(
        self,
        tool_with_no_required: ToolDefinition,
    ) -> None:
        """Test tool with no required params validates with empty args."""
        tool_call = {
            "name": "optional_tool",
            "arguments": {},
        }
        is_valid, error = validate_tool_call(tool_call, [tool_with_no_required])
        assert is_valid


class TestInspectMCPServer:
    """Tests for inspect_mcp_server async function."""

    @pytest.mark.asyncio
    async def test_inspect_returns_tools(self, mock_mcp_tools: list[dict[str, Any]]) -> None:
        """Test successful inspection returns ToolDefinitions."""
        from mcp_forge.tools.inspector import inspect_mcp_server

        # Create mock tool objects
        mock_tools = []
        for tool_data in mock_mcp_tools:
            mock_tool = MagicMock()
            mock_tool.name = tool_data["name"]
            mock_tool.description = tool_data["description"]
            mock_tool.inputSchema = tool_data["inputSchema"]
            mock_tools.append(mock_tool)

        # Mock the tools response
        mock_response = MagicMock()
        mock_response.tools = mock_tools

        # Mock session
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        # Mock stdio_client context manager
        mock_stdio = MagicMock()
        mock_stdio.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.__aexit__ = AsyncMock()

        with patch("mcp_forge.tools.inspector.stdio_client", return_value=mock_stdio):
            with patch("mcp_forge.tools.inspector.ClientSession", return_value=mock_session):
                tools = await inspect_mcp_server("test_command")

        assert len(tools) == 2
        assert tools[0].name == "get_weather"
        assert tools[1].name == "search_files"
        assert all(isinstance(t, ToolDefinition) for t in tools)

    @pytest.mark.asyncio
    async def test_inspect_timeout_error(self) -> None:
        """Test timeout raises MCPInspectorError."""
        import asyncio

        from mcp_forge.tools.inspector import inspect_mcp_server

        async def slow_operation(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(10)

        with patch("mcp_forge.tools.inspector.stdio_client") as mock_stdio:
            mock_stdio.return_value.__aenter__ = slow_operation

            with pytest.raises(MCPInspectorError, match="Timeout"):
                await inspect_mcp_server("slow_command", timeout=0.01)

    @pytest.mark.asyncio
    async def test_inspect_command_not_found(self) -> None:
        """Test FileNotFoundError is converted to MCPInspectorError."""
        from mcp_forge.tools.inspector import inspect_mcp_server

        with patch("mcp_forge.tools.inspector.stdio_client") as mock_stdio:
            mock_stdio.return_value.__aenter__ = AsyncMock(
                side_effect=FileNotFoundError("Command not found")
            )

            with pytest.raises(MCPInspectorError, match="Command not found"):
                await inspect_mcp_server("nonexistent_command")
```

**Verification**: `pytest tests/unit/test_inspector.py -v`

---

### Task 6: Write QC Module Tests

**Action**: Create `tests/unit/test_qc.py`

```python
"""Tests for mcp_forge.data.qc module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcp_forge.data.qc import (
    DataQualityController,
    QCConfig,
    QCIssue,
    ValidatedSample,
)
from mcp_forge.state import ToolDefinition


class TestQCConfig:
    """Tests for QCConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = QCConfig()
        assert config.schema_pass_threshold == 0.98
        assert config.min_samples_per_tool == 10
        assert config.dedup_enabled is True
        assert config.auto_repair is True

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = QCConfig(
            schema_pass_threshold=0.95,
            min_samples_per_tool=5,
            dedup_enabled=False,
        )
        assert config.schema_pass_threshold == 0.95
        assert config.min_samples_per_tool == 5
        assert config.dedup_enabled is False

    def test_default_scenario_targets(self) -> None:
        """Test default scenario target weights."""
        config = QCConfig()
        assert config.scenario_targets["standard"] == 0.60
        assert config.scenario_targets["no_tool"] == 0.15
        assert sum(config.scenario_targets.values()) == pytest.approx(1.0)


class TestQCIssue:
    """Tests for QCIssue dataclass."""

    def test_to_dict(self) -> None:
        """Test issue serialization."""
        issue = QCIssue(
            sample_id="test_001",
            issue_type="schema_error",
            message="Invalid schema",
            severity="error",
            repairable=True,
            repair_action="Fix schema",
        )
        result = issue.to_dict()

        assert result["sample_id"] == "test_001"
        assert result["issue_type"] == "schema_error"
        assert result["repairable"] is True


class TestValidatedSample:
    """Tests for ValidatedSample dataclass."""

    def test_to_dict(self) -> None:
        """Test sample serialization."""
        sample = ValidatedSample(
            id="sample_001",
            source="seed",
            scenario="standard",
            tool_name="get_weather",
            messages=[{"role": "user", "content": "test"}],
            content_hash="abc123",
        )
        result = sample.to_dict()

        assert result["id"] == "sample_001"
        assert result["qc_passed"] is True
        assert "content_hash" not in result  # Hash not in output


class TestDataQualityController:
    """Tests for DataQualityController class."""

    @pytest.fixture
    def qc_controller(self, sample_tools: list[ToolDefinition]) -> DataQualityController:
        """Create QC controller with sample tools."""
        return DataQualityController(sample_tools)

    def test_init_with_tools(
        self,
        qc_controller: DataQualityController,
        sample_tools: list[ToolDefinition],
    ) -> None:
        """Test controller initialization."""
        assert len(qc_controller.tool_names) == len(sample_tools)
        assert "get_weather" in qc_controller.tool_names
        assert "search_files" in qc_controller.tool_names

    def test_validate_dataset_success(
        self,
        qc_controller: DataQualityController,
        sample_data_path: Path,
    ) -> None:
        """Test successful dataset validation."""
        report, validated = qc_controller.validate_dataset(sample_data_path)

        assert report.total_samples == 5
        assert report.valid_samples > 0
        assert len(validated) > 0

    def test_validate_dataset_with_output(
        self,
        qc_controller: DataQualityController,
        sample_data_path: Path,
        tmp_path: Path,
    ) -> None:
        """Test validation writes cleaned output."""
        output_path = tmp_path / "cleaned.jsonl"
        report, validated = qc_controller.validate_dataset(
            sample_data_path,
            output_path=output_path,
        )

        assert output_path.exists()
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == len(validated)

    def test_validate_sample_valid(
        self,
        qc_controller: DataQualityController,
        valid_training_sample: dict[str, Any],
    ) -> None:
        """Test validation of valid sample."""
        result = qc_controller._validate_sample(valid_training_sample)

        assert result is not None
        assert isinstance(result, ValidatedSample)
        assert result.id == "test_001"
        assert result.tool_name == "get_weather"

    def test_validate_sample_missing_field(
        self,
        qc_controller: DataQualityController,
    ) -> None:
        """Test validation fails for missing required field."""
        sample = {"id": "bad", "source": "seed"}  # Missing 'messages'
        result = qc_controller._validate_sample(sample)

        assert result is None
        assert any(i.issue_type == "missing_field" for i in qc_controller.issues)

    def test_validate_sample_invalid_scenario(
        self,
        qc_controller: DataQualityController,
        valid_training_sample: dict[str, Any],
    ) -> None:
        """Test invalid scenario is repaired when auto_repair is on."""
        sample = valid_training_sample.copy()
        sample["scenario"] = "unknown_scenario"

        result = qc_controller._validate_sample(sample)

        # Should be repaired to 'standard'
        assert result is not None
        assert result.scenario == "standard"

    def test_validate_sample_invalid_scenario_no_repair(
        self,
        sample_tools: list[ToolDefinition],
        valid_training_sample: dict[str, Any],
    ) -> None:
        """Test invalid scenario rejected when auto_repair is off."""
        config = QCConfig(auto_repair=False)
        controller = DataQualityController(sample_tools, config)

        sample = valid_training_sample.copy()
        sample["scenario"] = "unknown_scenario"

        result = controller._validate_sample(sample)
        assert result is None

    def test_validate_sample_missing_tool_call(
        self,
        qc_controller: DataQualityController,
        invalid_training_sample: dict[str, Any],
    ) -> None:
        """Test validation fails when tool call is missing."""
        result = qc_controller._validate_sample(invalid_training_sample)

        assert result is None
        assert any(i.issue_type == "missing_tool_call" for i in qc_controller.issues)

    def test_validate_sample_unknown_tool(
        self,
        qc_controller: DataQualityController,
        valid_training_sample: dict[str, Any],
    ) -> None:
        """Test validation fails for unknown tool name."""
        sample = valid_training_sample.copy()
        sample["messages"][-1]["content"] = '```json\n{"name": "unknown_tool", "arguments": {}}\n```'

        result = qc_controller._validate_sample(sample)

        assert result is None
        assert any(i.issue_type == "invalid_tool" for i in qc_controller.issues)

    def test_validate_sample_schema_error(
        self,
        qc_controller: DataQualityController,
        valid_training_sample: dict[str, Any],
    ) -> None:
        """Test validation fails for schema errors."""
        sample = valid_training_sample.copy()
        # Missing required 'location' parameter
        sample["messages"][-1]["content"] = '```json\n{"name": "get_weather", "arguments": {}}\n```'

        result = qc_controller._validate_sample(sample)

        assert result is None
        assert any(i.issue_type == "schema_error" for i in qc_controller.issues)

    def test_duplicate_detection(
        self,
        qc_controller: DataQualityController,
        valid_training_sample: dict[str, Any],
    ) -> None:
        """Test duplicate samples are detected."""
        # Validate same sample twice
        qc_controller._validate_sample(valid_training_sample)
        result = qc_controller._validate_sample(valid_training_sample)

        assert result is None  # Duplicate dropped
        assert any(i.issue_type == "duplicate" for i in qc_controller.issues)

    def test_duplicate_detection_disabled(
        self,
        sample_tools: list[ToolDefinition],
        valid_training_sample: dict[str, Any],
    ) -> None:
        """Test duplicates allowed when dedup disabled."""
        config = QCConfig(dedup_enabled=False)
        controller = DataQualityController(sample_tools, config)

        controller._validate_sample(valid_training_sample)
        result = controller._validate_sample(valid_training_sample)

        # Duplicate detected but not dropped
        assert any(i.issue_type == "duplicate" for i in controller.issues)
        # Still returns result since dedup is disabled - check warning was added
        assert len([i for i in controller.issues if i.issue_type == "duplicate"]) == 1

    def test_no_tool_scenario_skips_tool_validation(
        self,
        qc_controller: DataQualityController,
    ) -> None:
        """Test no_tool scenario doesn't require tool call."""
        sample = {
            "id": "no_tool_001",
            "source": "seed",
            "scenario": "no_tool",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
        }

        result = qc_controller._validate_sample(sample)

        assert result is not None
        assert result.scenario == "no_tool"
        assert result.tool_name is None

    def test_extract_tool_call_json_block(
        self,
        qc_controller: DataQualityController,
    ) -> None:
        """Test tool call extraction from JSON block."""
        messages = [
            {
                "role": "assistant",
                "content": 'Here is the result:\n```json\n{"name": "test", "arguments": {"a": 1}}\n```',
            }
        ]

        result = qc_controller._extract_tool_call(messages)

        assert result is not None
        assert result["name"] == "test"
        assert result["arguments"]["a"] == 1

    def test_extract_tool_call_inline_json(
        self,
        qc_controller: DataQualityController,
    ) -> None:
        """Test tool call extraction from inline JSON."""
        messages = [
            {
                "role": "assistant",
                "content": 'I will call {"name": "test", "arguments": {}} now.',
            }
        ]

        result = qc_controller._extract_tool_call(messages)

        assert result is not None
        assert result["name"] == "test"

    def test_extract_tool_call_with_parameters_key(
        self,
        qc_controller: DataQualityController,
    ) -> None:
        """Test tool call extraction handles 'parameters' as alias for 'arguments'."""
        messages = [
            {
                "role": "assistant",
                "content": '```json\n{"name": "test", "parameters": {"x": 1}}\n```',
            }
        ]

        result = qc_controller._extract_tool_call(messages)

        assert result is not None
        assert result["arguments"]["x"] == 1

    def test_extract_tool_call_no_assistant_message(
        self,
        qc_controller: DataQualityController,
    ) -> None:
        """Test extraction returns None when no assistant message."""
        messages = [{"role": "user", "content": "Hello"}]

        result = qc_controller._extract_tool_call(messages)
        assert result is None

    def test_validate_schema_valid(
        self,
        qc_controller: DataQualityController,
    ) -> None:
        """Test schema validation passes for valid args."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        valid, error = qc_controller._validate_schema({"name": "test"}, schema)

        assert valid is True
        assert error is None

    def test_validate_schema_invalid(
        self,
        qc_controller: DataQualityController,
    ) -> None:
        """Test schema validation fails for invalid args."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }

        valid, error = qc_controller._validate_schema({"count": "not_int"}, schema)

        assert valid is False
        assert error is not None

    def test_compute_hash_deterministic(
        self,
        qc_controller: DataQualityController,
        valid_training_sample: dict[str, Any],
    ) -> None:
        """Test content hash is deterministic."""
        hash1 = qc_controller._compute_hash(valid_training_sample)
        hash2 = qc_controller._compute_hash(valid_training_sample)

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_compute_hash_different_for_different_content(
        self,
        qc_controller: DataQualityController,
        valid_training_sample: dict[str, Any],
    ) -> None:
        """Test different content produces different hash."""
        sample2 = valid_training_sample.copy()
        sample2["messages"] = [{"role": "user", "content": "Different content"}]

        hash1 = qc_controller._compute_hash(valid_training_sample)
        hash2 = qc_controller._compute_hash(sample2)

        assert hash1 != hash2

    def test_check_coverage_low_tool_coverage(
        self,
        qc_controller: DataQualityController,
    ) -> None:
        """Test coverage check adds warning for low tool coverage."""
        tool_coverage = {"get_weather": 5, "search_files": 20, "calculate": 20}
        scenario_coverage = {"standard": 40}

        qc_controller._check_coverage(tool_coverage, scenario_coverage, 45)

        assert any(
            i.issue_type == "low_coverage" and "get_weather" in i.message
            for i in qc_controller.issues
        )


class TestDataQualityControllerIntegration:
    """Integration tests for full validation flow."""

    def test_full_validation_flow(
        self,
        sample_tools: list[ToolDefinition],
        tmp_path: Path,
    ) -> None:
        """Test complete validation with various sample types."""
        # Create test data with various scenarios
        samples = [
            # Valid standard sample
            {
                "id": "valid_001",
                "source": "seed",
                "scenario": "standard",
                "messages": [
                    {"role": "user", "content": "Weather in Paris"},
                    {"role": "assistant", "content": '```json\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n```'},
                ],
            },
            # Valid no_tool sample
            {
                "id": "valid_002",
                "source": "seed",
                "scenario": "no_tool",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
            },
            # Invalid - missing field
            {"id": "invalid_001", "source": "seed"},
            # Invalid - bad JSON
            "not valid json",
        ]

        # Write to temp file
        data_path = tmp_path / "test.jsonl"
        with open(data_path, "w") as f:
            for sample in samples:
                if isinstance(sample, dict):
                    f.write(json.dumps(sample) + "\n")
                else:
                    f.write(sample + "\n")

        # Run validation
        controller = DataQualityController(sample_tools)
        report, validated = controller.validate_dataset(data_path)

        assert report.total_samples == 4
        assert report.valid_samples == 2  # Only 2 valid samples
        assert report.dropped_samples == 2
        assert len(validated) == 2
```

**Verification**: `pytest tests/unit/test_qc.py -v`

---

### Task 7: Add pytest Configuration Enhancements

**Action**: Update `pyproject.toml` with enhanced pytest configuration

Add markers and coverage configuration:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]
addopts = "--strict-markers"

[tool.coverage.run]
source = ["src/mcp_forge"]
branch = true
omit = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
fail_under = 85
show_missing = true
```

**Verification**: `pytest --collect-only`

---

### Task 8: Create Integration Test Stub

**Action**: Create `tests/integration/test_cli.py` with CLI smoke tests

```python
"""Integration tests for CLI commands."""

from __future__ import annotations

from click.testing import CliRunner
import pytest

from mcp_forge.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


class TestCLICommands:
    """Tests for CLI command invocations."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test --help displays usage."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Fine-tune" in result.output

    def test_cli_version(self, runner: CliRunner) -> None:
        """Test --version displays version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_status_no_session(self, runner: CliRunner) -> None:
        """Test status command with no active session."""
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "No active session" in result.output

    def test_tools_inspect_missing_server(self, runner: CliRunner) -> None:
        """Test tools inspect requires --server flag."""
        result = runner.invoke(cli, ["tools", "inspect"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()

    def test_qa_missing_data(self, runner: CliRunner) -> None:
        """Test qa command requires --data flag."""
        result = runner.invoke(cli, ["qa"])
        assert result.exit_code != 0

    def test_run_requires_server_or_tools(self, runner: CliRunner) -> None:
        """Test run command requires --server or --tools-file."""
        result = runner.invoke(cli, ["run", "--output", "./test"])
        assert result.exit_code != 0
        assert "required" in result.output.lower() or "Error" in result.output


class TestDoctorCommand:
    """Tests for doctor command."""

    @pytest.mark.slow
    def test_doctor_runs(self, runner: CliRunner) -> None:
        """Test doctor command executes without error."""
        result = runner.invoke(cli, ["doctor"])
        # Doctor should run even if some checks fail
        assert result.exit_code == 0
        assert "Environment Check" in result.output
        assert "Python" in result.output
```

**Verification**: `pytest tests/integration/test_cli.py -v`

---

### Task 9: Run Full Test Suite and Coverage

**Action**: Execute complete test suite with coverage reporting

```bash
# Run all tests with coverage
pytest --cov=mcp_forge --cov-report=term-missing --cov-report=html

# Run only unit tests
pytest tests/unit/ -v

# Run with verbose output
pytest -v --tb=short
```

**Expected Output**:
- All tests pass
- Coverage report generated
- Target: ≥85% coverage for existing modules

---

### Task 10: Fix Any Failing Tests and Update Coverage

**Action**: Address any test failures discovered during execution

This is an iterative task:
1. Run test suite
2. Identify failures
3. Fix test code or identify bugs in source
4. Re-run until all pass

**Verification**: `pytest -v && pytest --cov=mcp_forge --cov-fail-under=85`

---

## Verification Checklist

After all tasks complete:

- [ ] `tests/` directory structure created with unit/integration/fixtures
- [ ] `tests/conftest.py` has shared fixtures
- [ ] `tests/fixtures/sample_tools.json` exists with valid tool definitions
- [ ] `tests/fixtures/sample_data.jsonl` exists with sample training data
- [ ] `tests/unit/test_state.py` tests all state dataclasses and StateManager
- [ ] `tests/unit/test_inspector.py` tests all inspector functions
- [ ] `tests/unit/test_qc.py` tests DataQualityController
- [ ] `tests/integration/test_cli.py` has CLI smoke tests
- [ ] `pytest` discovers and runs all tests
- [ ] All tests pass (`pytest -v`)
- [ ] Coverage meets threshold (`pytest --cov=mcp_forge --cov-fail-under=85`)

---

## Success Criteria

From ROADMAP.md Phase 2:

- [ ] `pytest` discovers and runs tests
- [ ] Coverage reporting functional (target: 85%+ for existing code)
- [ ] State management fully tested
- [ ] Inspector module tested with mocked MCP server
- [ ] QC engine tested with sample datasets

---

## Output

**Artifacts Created:**
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/unit/__init__.py`
- `tests/unit/test_state.py`
- `tests/unit/test_inspector.py`
- `tests/unit/test_qc.py`
- `tests/integration/__init__.py`
- `tests/integration/test_cli.py`
- `tests/fixtures/sample_tools.json`
- `tests/fixtures/sample_data.jsonl`

**Configuration Updated:**
- `pyproject.toml` (pytest markers, coverage settings)

---

## Rollback Plan

If tests reveal bugs in source code:
1. Document bugs as issues in `.planning/phases/02-test-infrastructure/ISSUES.md`
2. Create fix tasks for Phase 3 or immediate hotfix
3. Mark affected tests as `xfail` temporarily if blocking

---

## Notes

- The inspector tests use mocks to avoid requiring real MCP server
- QC tests use fixture files to ensure reproducibility
- CLI tests use Click's CliRunner for isolated testing
- Coverage excludes `__init__.py` files and type-checking blocks
- Some tests may need adjustment based on actual behavior discovered during implementation

---

*Plan created: 2026-01-13*
