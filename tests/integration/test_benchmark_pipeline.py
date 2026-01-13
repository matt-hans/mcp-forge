"""Integration tests for benchmark pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcp_forge.state import (
    BenchmarkResult,
    PipelineStage,
    StateManager,
    ToolDefinition,
)
from mcp_forge.validation import (
    BenchmarkConfig,
    BenchmarkRunner,
    LatencyStats,
)
from mcp_forge.validation.config import StubConfig
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


class TestBenchmarkPipelineIntegration:
    """Integration tests for benchmark stage in pipeline."""

    def test_benchmark_config_with_stub(self, tmp_path: Path):
        """BenchmarkConfig works with stub configuration."""
        stub_config = StubConfig(stub_type="weather", deterministic=True)
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test-model",
            stub_config=stub_config,
        )

        assert config.stub_config.stub_type == "weather"
        assert config.mcp_command is None

    def test_benchmark_result_serialization(self):
        """BenchmarkResult can be serialized to/from dict."""
        result = BenchmarkResult(
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            overall_score=0.85,
            per_tool_results={
                "get_weather": {
                    "samples": 20,
                    "accuracy": 0.90,
                    "schema": 0.95,
                    "latency_mean_ms": 100.5,
                }
            },
            per_scenario_results={
                "no_tool": {"pass_rate": 0.80},
            },
            baseline_comparison={
                "baseline_model": "old-model",
                "overall_delta": 0.05,
            },
        )

        # Serialize
        data = result.to_dict()
        assert data["model_name"] == "test-model"
        assert data["overall_score"] == 0.85
        assert "get_weather" in data["per_tool_results"]

        # Deserialize
        restored = BenchmarkResult.from_dict(data)
        assert restored.model_name == result.model_name
        assert restored.overall_score == result.overall_score
        assert restored.per_tool_results == result.per_tool_results

    def test_state_manager_stores_benchmark_result(
        self, state_manager: StateManager, weather_tool: ToolDefinition
    ):
        """State manager can store benchmark results."""
        state = state_manager.create_session(
            mcp_command="npx test-server",
            system_prompt="Test prompt",
            model_family="deepseek-r1",
            output_path="./output",
        )

        state.tools = [weather_tool]
        state.benchmark_result = BenchmarkResult(
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            overall_score=0.87,
            per_tool_results={"get_weather": {"accuracy": 0.92}},
            per_scenario_results={"no_tool": {"pass_rate": 0.85}},
        )

        state_manager.save_state(state)

        # Reload and verify
        loaded = state_manager.load_state()
        assert loaded.benchmark_result is not None
        assert loaded.benchmark_result.overall_score == 0.87
        assert loaded.benchmark_result.model_name == "test-model"

    def test_benchmark_report_saved_to_reports_dir(
        self, state_manager: StateManager
    ):
        """Benchmark reports saved to reports directory."""
        state_manager.ensure_dirs()

        result = BenchmarkResult(
            model_name="test-model",
            timestamp="2024-01-01T00:00:00Z",
            overall_score=0.85,
            per_tool_results={"get_weather": {"accuracy": 0.90}},
            per_scenario_results={"no_tool": {"pass_rate": 0.80}},
        )

        report_path = state_manager.save_benchmark_result(result)

        # Check JSON report exists
        assert report_path.exists()
        with open(report_path) as f:
            loaded = json.load(f)
        assert loaded["model_name"] == "test-model"

        # Check Markdown report exists
        md_path = report_path.with_suffix(".md")
        assert md_path.exists()

    @pytest.mark.integration
    def test_full_benchmark_flow_with_mocked_model(
        self, tmp_path: Path, weather_tool: ToolDefinition
    ):
        """Full benchmark flow with mocked model inference."""
        stub_config = StubConfig(stub_type="weather", deterministic=True)
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test-model",
            samples_per_tool=3,
            samples_per_scenario=3,
            stub_config=stub_config,
            warmup_samples=0,
        )

        runner = BenchmarkRunner(config, [weather_tool])

        # Mock validation runner
        mock_val_runner = MagicMock()
        mock_val_runner.model = MagicMock()
        mock_val_runner.tokenizer = MagicMock()
        mock_val_runner.validate_single.return_value = {
            "prompt": "test",
            "parsed": True,
            "schema_valid": True,
            "tool_correct": True,
            "loop_complete": True,
            "error": None,
        }
        runner._validation_runner = mock_val_runner

        result = runner.run()

        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "test-model"
        assert "get_weather" in result.per_tool_results


class TestBenchmarkCLIIntegration:
    """Integration tests for benchmark CLI commands."""

    @pytest.mark.integration
    def test_benchmark_command_help(self):
        """benchmark command shows help."""
        from click.testing import CliRunner

        from mcp_forge.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])

        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--tools" in result.output
        assert "--stub" in result.output
        assert "--samples-per-tool" in result.output

    @pytest.mark.integration
    def test_benchmark_command_requires_model_and_tools(self, tmp_path: Path):
        """benchmark command requires --model and --tools."""
        from click.testing import CliRunner

        from mcp_forge.cli import cli

        runner = CliRunner()

        # Missing both
        result = runner.invoke(cli, ["benchmark"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()


class TestLatencyStatsIntegration:
    """Integration tests for latency statistics."""

    def test_latency_stats_from_realistic_data(self):
        """LatencyStats works with realistic latency data."""
        # Simulate realistic latencies (in ms)
        latencies = [
            50.2, 48.1, 55.3, 52.0, 51.8,
            49.5, 53.2, 200.5,  # One outlier
            51.0, 50.8, 52.3, 49.9, 51.5,
            50.1, 52.8, 51.2, 50.5, 53.0,
            49.8, 51.9,
        ]

        stats = LatencyStats.from_samples(latencies)

        # P50 should be around 51ms
        assert 49 < stats.p50_ms < 55

        # Mean should be pulled up by outlier
        assert stats.mean_ms > 50

        # P95 should capture outlier effect
        assert stats.p95_ms > 100  # The outlier is 200ms


class TestStubConsistency:
    """Tests for stub behavior consistency in benchmarks."""

    def test_weather_stub_consistent_across_benchmark_runs(self):
        """Weather stub produces consistent results for benchmarks."""
        results = []
        for _ in range(3):
            stub = StubRegistry.get("weather", seed=42)
            result = stub.execute("get_weather", {"location": "Paris"})
            results.append(result)

        assert all(r == results[0] for r in results)

    def test_benchmark_samples_deterministic(self, tmp_path: Path):
        """Benchmark sample generation is deterministic."""
        stub_config = StubConfig(stub_type="weather")
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test",
            samples_per_tool=10,
            stub_config=stub_config,
        )

        tool = ToolDefinition(
            name="get_weather",
            description="Get weather",
            input_schema={"type": "object", "properties": {}},
        )

        runner1 = BenchmarkRunner(config, [tool])
        runner2 = BenchmarkRunner(config, [tool])

        samples1 = runner1._generate_benchmark_samples()
        samples2 = runner2._generate_benchmark_samples()

        for key in samples1:
            prompts1 = [s.prompt for s in samples1[key]]
            prompts2 = [s.prompt for s in samples2[key]]
            assert prompts1 == prompts2
