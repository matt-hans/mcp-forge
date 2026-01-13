"""Unit tests for BenchmarkConfig class."""

from pathlib import Path

import pytest

from mcp_forge.validation.benchmark import BenchmarkConfig
from mcp_forge.validation.config import InferenceConfig, StubConfig


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_config_requires_server_or_stub(self, tmp_path: Path):
        """Config raises if neither server nor stub provided."""
        with pytest.raises(ValueError, match="Either stub_config or mcp_command must be provided"):
            BenchmarkConfig(
                model_path=tmp_path / "adapter",
                model_name="test-model",
            )

    def test_config_accepts_stub_config(self, tmp_path: Path):
        """Config accepts valid stub configuration."""
        stub_config = StubConfig(stub_type="weather", deterministic=True)
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test-model",
            stub_config=stub_config,
        )
        assert config.stub_config == stub_config
        assert config.mcp_command is None

    def test_config_accepts_mcp_command(self, tmp_path: Path):
        """Config accepts MCP command."""
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test-model",
            mcp_command="npx -y @mcp/server-weather",
        )
        assert config.mcp_command == "npx -y @mcp/server-weather"
        assert config.stub_config is None

    def test_config_defaults(self, tmp_path: Path):
        """Config has sensible defaults for samples and thresholds."""
        stub_config = StubConfig(stub_type="weather")
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test-model",
            stub_config=stub_config,
        )

        # Default sample counts
        assert config.samples_per_tool == 20
        assert config.samples_per_scenario == 20

        # Default thresholds (from CLAUDE.md)
        assert config.accuracy_threshold == 0.90
        assert config.no_tool_threshold == 0.85
        assert config.loop_threshold == 0.95

        # Default latency settings
        assert config.measure_latency is True
        assert config.warmup_samples == 3

    def test_config_custom_samples(self, tmp_path: Path):
        """Config accepts custom sample counts."""
        stub_config = StubConfig(stub_type="weather")
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test-model",
            stub_config=stub_config,
            samples_per_tool=50,
            samples_per_scenario=30,
        )
        assert config.samples_per_tool == 50
        assert config.samples_per_scenario == 30

    def test_config_custom_thresholds(self, tmp_path: Path):
        """Config accepts custom threshold values."""
        stub_config = StubConfig(stub_type="weather")
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test-model",
            stub_config=stub_config,
            accuracy_threshold=0.95,
            no_tool_threshold=0.90,
            loop_threshold=0.98,
        )
        assert config.accuracy_threshold == 0.95
        assert config.no_tool_threshold == 0.90
        assert config.loop_threshold == 0.98

    def test_config_baseline_path(self, tmp_path: Path):
        """Config accepts baseline path for comparison."""
        stub_config = StubConfig(stub_type="weather")
        baseline_path = tmp_path / "baseline.json"
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test-model",
            stub_config=stub_config,
            baseline_path=baseline_path,
        )
        assert config.baseline_path == baseline_path

    def test_config_inference_settings(self, tmp_path: Path):
        """Config accepts custom inference settings."""
        stub_config = StubConfig(stub_type="weather")
        inference = InferenceConfig(
            max_new_tokens=256,
            temperature=0.2,
        )
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test-model",
            stub_config=stub_config,
            inference=inference,
        )
        assert config.inference.max_new_tokens == 256
        assert config.inference.temperature == 0.2

    def test_config_disable_latency(self, tmp_path: Path):
        """Config can disable latency tracking."""
        stub_config = StubConfig(stub_type="weather")
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test-model",
            stub_config=stub_config,
            measure_latency=False,
        )
        assert config.measure_latency is False

    def test_config_custom_warmup(self, tmp_path: Path):
        """Config accepts custom warmup sample count."""
        stub_config = StubConfig(stub_type="weather")
        config = BenchmarkConfig(
            model_path=tmp_path / "adapter",
            model_name="test-model",
            stub_config=stub_config,
            warmup_samples=5,
        )
        assert config.warmup_samples == 5
