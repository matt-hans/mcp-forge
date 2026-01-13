"""Shared test fixtures and configuration for mcp-forge tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_forge.state import (
    PipelineState,
    QCReport,
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
