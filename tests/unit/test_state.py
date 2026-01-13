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
        assert samples["ambiguous"] == 10  # 100 * 0.10
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
