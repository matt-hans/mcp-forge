"""Unit tests for BenchmarkRunner class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_forge.state import BenchmarkResult, Scenario, ToolDefinition
from mcp_forge.validation.benchmark import BenchmarkConfig, BenchmarkRunner, LatencyStats
from mcp_forge.validation.config import StubConfig
from mcp_forge.validation.runner import ValidationSample


@pytest.fixture
def weather_tool() -> ToolDefinition:
    """Weather tool definition for tests."""
    return ToolDefinition(
        name="get_weather",
        description="Get weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    )


@pytest.fixture
def filesystem_tool() -> ToolDefinition:
    """Filesystem tool definition for tests."""
    return ToolDefinition(
        name="list_files",
        description="List files in a directory",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"},
            },
            "required": ["path"],
        },
    )


@pytest.fixture
def stub_config() -> StubConfig:
    """Standard stub config for tests."""
    return StubConfig(stub_type="weather", deterministic=True, seed=42)


@pytest.fixture
def benchmark_config(tmp_path: Path, stub_config: StubConfig) -> BenchmarkConfig:
    """Standard benchmark config for tests."""
    return BenchmarkConfig(
        model_path=tmp_path / "adapter",
        model_name="test-model",
        samples_per_tool=5,
        samples_per_scenario=5,
        stub_config=stub_config,
        warmup_samples=0,  # Disable warmup for faster tests
    )


class TestBenchmarkRunnerInit:
    """Tests for BenchmarkRunner initialization."""

    def test_initialization(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """Runner initializes with config and tools."""
        runner = BenchmarkRunner(benchmark_config, [weather_tool])
        assert runner.config == benchmark_config
        assert runner.tools == [weather_tool]
        assert runner._validation_runner is None
        assert runner._latencies == []

    def test_get_validation_runner_creates_instance(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """_get_validation_runner creates ValidationRunner."""
        runner = BenchmarkRunner(benchmark_config, [weather_tool])
        val_runner = runner._get_validation_runner()

        assert val_runner is not None
        assert val_runner.config.model_path == benchmark_config.model_path
        assert val_runner.config.stub_config == benchmark_config.stub_config

    def test_get_validation_runner_caches(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """_get_validation_runner returns cached instance."""
        runner = BenchmarkRunner(benchmark_config, [weather_tool])
        val_runner1 = runner._get_validation_runner()
        val_runner2 = runner._get_validation_runner()
        assert val_runner1 is val_runner2


class TestBenchmarkSampleGeneration:
    """Tests for benchmark sample generation."""

    def test_generate_tool_samples(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """Sample generation creates diverse prompts for weather tool."""
        import random

        runner = BenchmarkRunner(benchmark_config, [weather_tool])
        samples = runner._generate_tool_samples(weather_tool, 10, random.Random(42))

        assert len(samples) == 10
        for sample in samples:
            assert sample.expected_tool == "get_weather"
            assert "location" in sample.expected_args

    def test_generate_tool_samples_filesystem(
        self, benchmark_config: BenchmarkConfig, filesystem_tool: ToolDefinition
    ):
        """Sample generation creates filesystem prompts."""
        import random

        runner = BenchmarkRunner(benchmark_config, [filesystem_tool])
        samples = runner._generate_tool_samples(filesystem_tool, 10, random.Random(42))

        assert len(samples) == 10
        for sample in samples:
            assert sample.expected_tool == "list_files"
            assert "path" in sample.expected_args

    def test_generate_tool_samples_unknown_tool(
        self, benchmark_config: BenchmarkConfig
    ):
        """Sample generation creates generic prompts for unknown tools."""
        import random

        unknown_tool = ToolDefinition(
            name="custom_tool",
            description="A custom tool",
            input_schema={"type": "object", "properties": {}},
        )
        runner = BenchmarkRunner(benchmark_config, [unknown_tool])
        samples = runner._generate_tool_samples(unknown_tool, 5, random.Random(42))

        assert len(samples) == 5
        for sample in samples:
            assert sample.expected_tool == "custom_tool"
            assert "custom_tool" in sample.prompt

    def test_generate_scenario_samples_no_tool(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """Sample generation creates no-tool samples."""
        import random

        runner = BenchmarkRunner(benchmark_config, [weather_tool])
        samples = runner._generate_scenario_samples(Scenario.NO_TOOL, 5, random.Random(42))

        assert len(samples) <= 5
        for sample in samples:
            assert sample.expected_tool is None
            assert sample.expected_args is None

    def test_generate_scenario_samples_error(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """Sample generation creates error samples."""
        import random

        runner = BenchmarkRunner(benchmark_config, [weather_tool])
        samples = runner._generate_scenario_samples(Scenario.ERROR, 2, random.Random(42))

        assert len(samples) <= 2
        for sample in samples:
            assert sample.expected_tool is not None

    def test_generate_benchmark_samples_structure(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """_generate_benchmark_samples creates proper structure."""
        runner = BenchmarkRunner(benchmark_config, [weather_tool])
        samples = runner._generate_benchmark_samples()

        # Should have tool samples
        assert "tool:get_weather" in samples

        # Should have scenario samples
        assert "scenario:standard" in samples
        assert "scenario:no_tool" in samples
        assert "scenario:error" in samples

    def test_generate_benchmark_samples_deterministic(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """Sample generation is deterministic."""
        runner = BenchmarkRunner(benchmark_config, [weather_tool])

        samples1 = runner._generate_benchmark_samples()
        samples2 = runner._generate_benchmark_samples()

        # Same keys
        assert samples1.keys() == samples2.keys()

        # Same prompts
        for key in samples1:
            prompts1 = [s.prompt for s in samples1[key]]
            prompts2 = [s.prompt for s in samples2[key]]
            assert prompts1 == prompts2


class TestBenchmarkRunnerExecution:
    """Tests for BenchmarkRunner execution with mocked model."""

    @pytest.fixture
    def mock_runner(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """Create runner with mocked validation runner."""
        runner = BenchmarkRunner(benchmark_config, [weather_tool])

        # Mock the validation runner
        mock_val_runner = MagicMock()
        mock_val_runner.model = MagicMock()
        mock_val_runner.tokenizer = MagicMock()

        # Mock validate_single to return success
        mock_val_runner.validate_single.return_value = {
            "prompt": "test",
            "parsed": True,
            "schema_valid": True,
            "tool_correct": True,
            "loop_complete": True,
            "error": None,
        }

        runner._validation_runner = mock_val_runner
        return runner

    def test_run_aggregates_per_tool(self, mock_runner: BenchmarkRunner):
        """Run correctly aggregates per-tool metrics."""
        result = mock_runner.run()

        assert isinstance(result, BenchmarkResult)
        assert "get_weather" in result.per_tool_results

        weather_metrics = result.per_tool_results["get_weather"]
        assert "accuracy" in weather_metrics
        assert "schema" in weather_metrics
        assert "latency_mean_ms" in weather_metrics

    def test_run_aggregates_per_scenario(self, mock_runner: BenchmarkRunner):
        """Run correctly aggregates per-scenario metrics."""
        result = mock_runner.run()

        assert "no_tool" in result.per_scenario_results
        assert "standard" in result.per_scenario_results
        assert "pass_rate" in result.per_scenario_results["no_tool"]

    def test_run_tracks_latency(self, mock_runner: BenchmarkRunner):
        """Run tracks and aggregates latency metrics."""
        result = mock_runner.run()

        # Check latency is tracked
        weather_metrics = result.per_tool_results.get("get_weather", {})
        assert weather_metrics.get("latency_mean_ms", 0) >= 0

    def test_run_calculates_overall_score(self, mock_runner: BenchmarkRunner):
        """Run calculates overall score correctly."""
        result = mock_runner.run()

        # Overall score should be between 0 and 1
        assert 0 <= result.overall_score <= 1

    def test_run_calls_progress_callback(self, mock_runner: BenchmarkRunner):
        """Run calls progress callback with correct values."""
        progress_calls = []

        def track_progress(category: str, current: int, total: int, pct: float):
            progress_calls.append((category, current, total, pct))

        mock_runner.run(progress_callback=track_progress)

        assert len(progress_calls) > 0
        # Last call should be at 100%
        assert progress_calls[-1][3] == 1.0

    def test_run_handles_failed_samples(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """Run handles failed validation samples gracefully."""
        runner = BenchmarkRunner(benchmark_config, [weather_tool])

        mock_val_runner = MagicMock()
        mock_val_runner.model = MagicMock()
        mock_val_runner.tokenizer = MagicMock()

        # Alternate between success and failure
        call_count = [0]

        def mock_validate(sample):
            call_count[0] += 1
            return {
                "prompt": sample.prompt,
                "parsed": call_count[0] % 2 == 0,
                "schema_valid": call_count[0] % 2 == 0,
                "tool_correct": call_count[0] % 2 == 0,
                "loop_complete": call_count[0] % 2 == 0,
                "error": None if call_count[0] % 2 == 0 else "test error",
            }

        mock_val_runner.validate_single.side_effect = mock_validate
        runner._validation_runner = mock_val_runner

        result = runner.run()

        # Should still produce a result
        assert isinstance(result, BenchmarkResult)


class TestBenchmarkBaselineComparison:
    """Tests for baseline comparison functionality."""

    def test_compare_to_baseline(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """Baseline comparison calculates correct deltas."""
        runner = BenchmarkRunner(benchmark_config, [weather_tool])

        current = BenchmarkResult(
            model_name="new-model",
            timestamp="2024-01-02T00:00:00Z",
            overall_score=0.90,
            per_tool_results={
                "get_weather": {"accuracy": 0.95, "latency_mean_ms": 100.0},
            },
            per_scenario_results={"no_tool": {"pass_rate": 0.85}},
        )

        baseline = BenchmarkResult(
            model_name="old-model",
            timestamp="2024-01-01T00:00:00Z",
            overall_score=0.80,
            per_tool_results={
                "get_weather": {"accuracy": 0.85, "latency_mean_ms": 150.0},
            },
            per_scenario_results={"no_tool": {"pass_rate": 0.75}},
        )

        comparison = runner.compare_to_baseline(current, baseline)

        assert comparison["baseline_model"] == "old-model"
        assert comparison["overall_delta"] == pytest.approx(0.10)
        assert comparison["per_tool_deltas"]["get_weather"]["accuracy_delta"] == pytest.approx(0.10)
        assert comparison["per_tool_deltas"]["get_weather"]["latency_delta_ms"] == pytest.approx(-50.0)

    def test_compare_to_baseline_missing_tool(
        self, benchmark_config: BenchmarkConfig, weather_tool: ToolDefinition
    ):
        """Baseline comparison handles missing tools."""
        runner = BenchmarkRunner(benchmark_config, [weather_tool])

        current = BenchmarkResult(
            model_name="new-model",
            timestamp="2024-01-02T00:00:00Z",
            overall_score=0.90,
            per_tool_results={
                "get_weather": {"accuracy": 0.95},
                "list_files": {"accuracy": 0.80},
            },
        )

        baseline = BenchmarkResult(
            model_name="old-model",
            timestamp="2024-01-01T00:00:00Z",
            overall_score=0.80,
            per_tool_results={
                "get_weather": {"accuracy": 0.85},
                # No list_files in baseline
            },
        )

        comparison = runner.compare_to_baseline(current, baseline)

        # Should only have delta for get_weather
        assert "get_weather" in comparison["per_tool_deltas"]
        assert "list_files" not in comparison["per_tool_deltas"]
