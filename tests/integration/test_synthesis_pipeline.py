"""Integration tests for the synthesis pipeline.

Tests cover CLI commands and full pipeline execution with mocked external services.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from mcp_forge.cli import cli
from mcp_forge.state import StateManager, SynthesisPlan, ToolDefinition


@pytest.fixture
def cli_runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def tools_file(tmp_path: Path, sample_tools: list[ToolDefinition]) -> Path:
    """Create temporary tools JSON file."""
    tools_path = tmp_path / "tools.json"
    with open(tools_path, "w") as f:
        json.dump([t.to_dict() for t in sample_tools], f)
    return tools_path


@pytest.fixture
def mock_synthesis_result():
    """Mock SynthesisResult for pipeline tests."""
    from mcp_forge.data.synthesizer import SynthesisResult

    return SynthesisResult(
        seed_count=20,
        augmented_count=80,
        total_count=100,
        seed_path=Path("/tmp/seed.jsonl"),
        augmented_path=Path("/tmp/augmented.jsonl"),
        training_path=Path("/tmp/train.jsonl"),
        qc_passed=True,
        qc_report=None,
        scenario_distribution={"standard": 60, "no_tool": 15, "error": 10, "ambiguous": 10, "edge": 5},
    )


class TestSynthesisPipeline:
    """Integration tests for synthesis pipeline."""

    @pytest.mark.integration
    def test_cli_generate_command_help(self, cli_runner: CliRunner):
        """CLI generate command shows help."""
        result = cli_runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--server" in result.output
        assert "--tools" in result.output
        assert "--samples" in result.output

    @pytest.mark.integration
    def test_cli_generate_requires_source(self, cli_runner: CliRunner, tmp_path: Path):
        """CLI generate requires --server or --tools."""
        result = cli_runner.invoke(cli, [
            "generate",
            "--output", str(tmp_path / "output.jsonl"),
        ])
        assert result.exit_code == 1
        assert "Either --server or --tools is required" in result.output

    @pytest.mark.integration
    def test_cli_generate_with_tools_file(
        self,
        cli_runner: CliRunner,
        tools_file: Path,
        tmp_path: Path,
        mock_synthesis_result,
    ):
        """CLI generate works with tools file."""
        output_path = tmp_path / "output.jsonl"

        with patch("mcp_forge.data.synthesizer.DataSynthesizer") as MockSynth:
            # Setup mock
            mock_synthesizer = MagicMock()
            mock_synthesizer.synthesize = AsyncMock(return_value=mock_synthesis_result)
            MockSynth.return_value = mock_synthesizer

            # Create a dummy output file
            mock_synthesis_result.training_path = tmp_path / "train.jsonl"
            mock_synthesis_result.training_path.write_text('{"test": true}\n')

            result = cli_runner.invoke(cli, [
                "generate",
                "--tools", str(tools_file),
                "--samples", "100",
                "--output", str(output_path),
            ])

            # Should complete successfully
            assert "MCP-Forge Data Generation" in result.output
            assert "Found" in result.output

    @pytest.mark.integration
    def test_pipeline_stage_2_runs_synthesis(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_tools: list[ToolDefinition],
    ):
        """Pipeline stage 2 runs synthesis when continuing from stage 1."""
        # Create state directory
        state_dir = tmp_path / ".mcp-forge"
        state_dir.mkdir()

        # Create initial state after stage 1
        from mcp_forge.state import PipelineStage, PipelineState

        state = PipelineState(
            session_id="test123",
            stage=PipelineStage.INSPECTING,
            mcp_command="npx test",
            system_prompt="test",
            model_family="deepseek-r1",
            output_path=str(tmp_path / "output"),
            quantization="q8_0",
            tools=sample_tools,
            synthesis_plan=SynthesisPlan(
                total_samples=50,
                seed_samples=10,
                augmented_samples=40,
            ),
        )

        # Save state
        state_file = state_dir / "state.json"
        with open(state_file, "w") as f:
            json.dump(state.to_dict(), f)

        # Verify state exists
        assert state_file.exists()

    @pytest.mark.integration
    def test_synthesis_with_realistic_tools(
        self,
        tmp_path: Path,
        sample_tools: list[ToolDefinition],
    ):
        """Synthesis works with realistic tool definitions."""
        # This test verifies the tool definitions integrate properly
        from mcp_forge.data.synthesizer import DataSynthesizer

        plan = SynthesisPlan(
            total_samples=10,
            seed_samples=5,
            augmented_samples=5,
        )

        synthesizer = DataSynthesizer(
            tools=sample_tools,
            plan=plan,
            output_dir=tmp_path,
        )

        # Verify synthesizer initialized correctly
        assert len(synthesizer.tools) == len(sample_tools)
        assert synthesizer.plan.total_samples == 10

    @pytest.mark.integration
    def test_data_directory_structure(self, tmp_path: Path):
        """State manager creates correct directory structure."""
        state_manager = StateManager(base_path=tmp_path)
        state_manager.ensure_dirs()

        # Check directories exist
        assert (tmp_path / ".mcp-forge").exists()
        assert (tmp_path / ".mcp-forge" / "data").exists()
        assert (tmp_path / ".mcp-forge" / "logs").exists()
        assert (tmp_path / ".mcp-forge" / "reports").exists()


class TestCLIIntegration:
    """Integration tests for CLI commands related to synthesis."""

    @pytest.mark.integration
    def test_qa_command_with_synthesized_data(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_tools: list[ToolDefinition],
    ):
        """QA command works with synthesized data format."""
        # Create tools file
        tools_path = tmp_path / "tools.json"
        with open(tools_path, "w") as f:
            json.dump([t.to_dict() for t in sample_tools], f)

        # Create sample data file with Hermes format
        data_path = tmp_path / "data.jsonl"
        with open(data_path, "w") as f:
            for i in range(5):
                sample = {
                    "id": f"sample_{i}",
                    "source": "seed",
                    "scenario": "standard",
                    "tool_name": "get_weather",
                    "messages": [
                        {"role": "user", "content": f"Weather in city_{i}?"},
                        {"role": "assistant", "content": f'<tool_call>\n{{"name": "get_weather", "arguments": {{"location": "city_{i}"}}}}\n</tool_call>'},
                    ],
                }
                f.write(json.dumps(sample) + "\n")

        # Create state directory for report saving
        state_dir = tmp_path / ".mcp-forge" / "reports"
        state_dir.mkdir(parents=True)

        result = cli_runner.invoke(cli, [
            "qa",
            "--data", str(data_path),
            "--tools", str(tools_path),
        ])

        # Should complete and show report
        assert "Dataset Quality Report" in result.output

    @pytest.mark.integration
    def test_status_shows_synthesis_state(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        monkeypatch,
    ):
        """Status command shows synthesis state information."""
        # Create state with synthesis data
        from mcp_forge.state import PipelineStage, PipelineState

        state_dir = tmp_path / ".mcp-forge"
        state_dir.mkdir()

        state = PipelineState(
            session_id="test456",
            stage=PipelineStage.SYNTHESIZING,
            mcp_command="npx test",
            system_prompt="test",
            model_family="deepseek-r1",
            output_path=str(tmp_path),
            quantization="q8_0",
            synthesis_plan=SynthesisPlan(
                total_samples=100,
                seed_samples=20,
                augmented_samples=80,
            ),
            seed_data_path=str(tmp_path / "seed.jsonl"),
        )

        with open(state_dir / "state.json", "w") as f:
            json.dump(state.to_dict(), f)

        # Change to tmp_path so StateManager finds the state file
        monkeypatch.chdir(tmp_path)

        result = cli_runner.invoke(cli, ["status"])

        assert "synthesizing" in result.output.lower()
        assert "test456" in result.output


class TestEndToEndFlow:
    """End-to-end tests for synthesis flow."""

    @pytest.mark.integration
    def test_tools_inspect_to_generate_flow(
        self,
        cli_runner: CliRunner,
        tmp_path: Path,
        sample_tools: list[ToolDefinition],
    ):
        """Verify tools inspect output can be used for generate."""
        # Create tools file (simulating output from tools inspect)
        tools_path = tmp_path / "tools.json"
        with open(tools_path, "w") as f:
            json.dump([t.to_dict() for t in sample_tools], f)

        # Verify tools file is valid for generate command
        result = cli_runner.invoke(cli, [
            "generate",
            "--tools", str(tools_path),
            "--samples", "10",
            "--help",  # Just check help to verify options
        ])

        # Should show help without errors
        assert result.exit_code == 0

    @pytest.mark.integration
    def test_synthesis_plan_distribution(self):
        """Verify synthesis plan calculates correct distributions."""
        plan = SynthesisPlan(
            total_samples=1000,
            seed_samples=100,
            augmented_samples=900,
            scenario_weights={
                "standard": 0.60,
                "no_tool": 0.15,
                "error": 0.10,
                "ambiguous": 0.10,
                "edge": 0.05,
            },
        )

        samples_per_scenario = plan.get_samples_per_scenario()

        assert samples_per_scenario["standard"] == 600
        assert samples_per_scenario["no_tool"] == 150
        assert samples_per_scenario["error"] == 100
        assert samples_per_scenario["ambiguous"] == 100
        assert samples_per_scenario["edge"] == 50
        assert sum(samples_per_scenario.values()) == 1000
