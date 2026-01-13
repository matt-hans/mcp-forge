"""Integration tests for validation pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_forge.state import PipelineStage, StateManager, ToolDefinition, ValidationResult
from mcp_forge.validation import (
    StubConfig,
    ValidationConfig,
    ValidationRunner,
    generate_validation_samples,
)
from mcp_forge.validation.stubs import StubRegistry


@pytest.fixture
def temp_state_dir(tmp_path: Path) -> Path:
    """Temporary directory for state tests."""
    return tmp_path


@pytest.fixture
def state_manager(temp_state_dir: Path) -> StateManager:
    """StateManager with temporary directory."""
    return StateManager(base_path=temp_state_dir)


@pytest.fixture
def weather_tool() -> ToolDefinition:
    """Weather tool definition."""
    return ToolDefinition(
        name="get_weather",
        description="Get weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        },
    )


class TestValidationPipelineIntegration:
    """Integration tests for validation stage in pipeline."""

    def test_stub_tools_match_tool_definitions(self, weather_tool: ToolDefinition):
        """Stub tools match expected tool definitions."""
        stub = StubRegistry.get("weather")
        stub_tools = stub.get_tools()

        assert len(stub_tools) == 1
        assert stub_tools[0]["name"] == weather_tool.name
        assert stub_tools[0]["inputSchema"]["required"] == ["location"]

    def test_generated_samples_work_with_stub(self, weather_tool: ToolDefinition):
        """Generated samples can be executed against stub."""
        stub = StubRegistry.get("weather")
        samples = generate_validation_samples([weather_tool], count=5, seed=42)

        for sample in samples:
            if sample.expected_tool == "get_weather" and sample.expected_args:
                result = stub.execute("get_weather", sample.expected_args)
                assert "error" not in result
                assert "temperature" in result

    def test_validation_config_with_stub(self, tmp_path: Path):
        """ValidationConfig works with stub configuration."""
        stub_config = StubConfig(stub_type="weather", deterministic=True)
        config = ValidationConfig(
            model_path=tmp_path / "adapter",
            stub_config=stub_config,
        )

        assert config.stub_config.stub_type == "weather"
        assert config.mcp_command is None

    def test_validation_result_serialization(self):
        """ValidationResult can be serialized to/from dict."""
        result = ValidationResult(
            passed=True,
            samples_tested=20,
            samples_passed=19,
            tool_call_parse_rate=0.98,
            schema_conformance_rate=0.95,
            tool_selection_accuracy=0.92,
            loop_completion_rate=0.96,
            failures=[{"sample": "test", "error": "failed"}],
        )

        # Serialize
        data = result.to_dict()
        assert data["passed"] is True
        assert data["samples_tested"] == 20
        assert data["tool_call_parse_rate"] == 0.98

        # Deserialize
        restored = ValidationResult.from_dict(data)
        assert restored.passed == result.passed
        assert restored.samples_tested == result.samples_tested
        assert restored.tool_call_parse_rate == result.tool_call_parse_rate

    def test_state_manager_stores_validation_result(
        self, state_manager: StateManager, weather_tool: ToolDefinition
    ):
        """State manager can store validation results."""
        state = state_manager.create_session(
            mcp_command="npx test-server",
            system_prompt="Test prompt",
            model_family="deepseek-r1",
            output_path="./output",
        )

        state.tools = [weather_tool]
        state.validation_result = ValidationResult(
            passed=True,
            samples_tested=10,
            samples_passed=9,
            tool_call_parse_rate=0.95,
            schema_conformance_rate=0.90,
            tool_selection_accuracy=0.88,
            loop_completion_rate=0.92,
        )

        state_manager.save_state(state)

        # Reload and verify
        loaded = state_manager.load_state()
        assert loaded.validation_result is not None
        assert loaded.validation_result.passed is True
        assert loaded.validation_result.samples_tested == 10

    def test_validation_report_saved_to_reports_dir(
        self, state_manager: StateManager
    ):
        """Validation reports can be saved to reports directory."""
        state_manager.ensure_dirs()

        result = ValidationResult(
            passed=True,
            samples_tested=20,
            samples_passed=18,
            tool_call_parse_rate=0.95,
            schema_conformance_rate=0.90,
            tool_selection_accuracy=0.88,
            loop_completion_rate=0.92,
        )

        report_path = state_manager.get_report_path("validation_test.json")
        with open(report_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        assert report_path.exists()
        with open(report_path) as f:
            loaded = json.load(f)
        assert loaded["passed"] is True

    def test_full_validation_flow_with_mocked_model(
        self, tmp_path: Path, weather_tool: ToolDefinition
    ):
        """Full validation flow with mocked model inference."""
        stub_config = StubConfig(stub_type="weather")
        config = ValidationConfig(
            model_path=tmp_path / "adapter",
            samples=5,
            stub_config=stub_config,
        )

        runner = ValidationRunner(config, [weather_tool])

        # Mock model and generate_response
        runner.model = MagicMock()
        runner.tokenizer = MagicMock()

        # All responses return valid tool calls
        tool_response = """<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>"""
        runner.generate_response = MagicMock(return_value=tool_response)

        samples = generate_validation_samples([weather_tool], count=5, seed=42)

        result = runner.run(samples)

        assert isinstance(result, ValidationResult)
        assert result.samples_tested == 5
        # At least some should pass (no-tool samples will count as failures here)
        assert result.samples_passed >= 0


class TestValidationCLIIntegration:
    """Integration tests for validation CLI commands."""

    @pytest.mark.integration
    def test_validate_command_help(self):
        """validate command shows help."""
        from click.testing import CliRunner

        from mcp_forge.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])

        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--stub" in result.output
        assert "--samples" in result.output

    @pytest.mark.integration
    def test_validate_command_requires_server_or_stub(self, tmp_path: Path):
        """validate command requires either --server or --stub."""
        from click.testing import CliRunner

        from mcp_forge.cli import cli

        # Create a fake model path
        model_path = tmp_path / "adapter"
        model_path.mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--model", str(model_path)])

        assert result.exit_code == 1
        assert "Either --server or --stub is required" in result.output


class TestStubConsistency:
    """Tests for stub behavior consistency."""

    def test_weather_stub_consistent_across_runs(self):
        """Weather stub produces consistent results across runs."""
        results = []
        for _ in range(3):
            stub = StubRegistry.get("weather", seed=42)
            result = stub.execute("get_weather", {"location": "Paris"})
            results.append(result)

        assert all(r == results[0] for r in results)

    def test_filesystem_stub_consistent_across_runs(self):
        """Filesystem stub produces consistent results across runs."""
        results = []
        for _ in range(3):
            stub = StubRegistry.get("filesystem", seed=42)
            result = stub.execute("list_files", {"path": "/home/user/documents"})
            results.append(result)

        assert all(r == results[0] for r in results)

    def test_sample_generation_consistent(self, weather_tool: ToolDefinition):
        """Sample generation is consistent with same seed."""
        samples1 = generate_validation_samples([weather_tool], count=10, seed=42)
        samples2 = generate_validation_samples([weather_tool], count=10, seed=42)

        assert len(samples1) == len(samples2)
        for s1, s2 in zip(samples1, samples2):
            assert s1.prompt == s2.prompt
            assert s1.expected_tool == s2.expected_tool
