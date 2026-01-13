"""Unit tests for validation configuration dataclasses."""

from pathlib import Path

import pytest

from mcp_forge.validation.config import (
    InferenceConfig,
    StubConfig,
    ValidationConfig,
)


class TestInferenceConfig:
    """Tests for InferenceConfig dataclass."""

    def test_default_values(self):
        """InferenceConfig has sensible defaults."""
        config = InferenceConfig()
        assert config.max_new_tokens == 512
        assert config.temperature == 0.1
        assert config.top_p == 0.95
        assert config.do_sample is True

    def test_custom_values(self):
        """InferenceConfig accepts custom values."""
        config = InferenceConfig(
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.9,
            do_sample=False,
        )
        assert config.max_new_tokens == 256
        assert config.temperature == 0.5
        assert config.top_p == 0.9
        assert config.do_sample is False


class TestStubConfig:
    """Tests for StubConfig dataclass."""

    def test_required_stub_type(self):
        """StubConfig requires stub_type."""
        config = StubConfig(stub_type="weather")
        assert config.stub_type == "weather"

    def test_default_values(self):
        """StubConfig has sensible defaults."""
        config = StubConfig(stub_type="filesystem")
        assert config.deterministic is True
        assert config.seed == 42
        assert config.response_delay == 0.0

    def test_custom_values(self):
        """StubConfig accepts custom values."""
        config = StubConfig(
            stub_type="weather",
            deterministic=False,
            seed=123,
            response_delay=0.5,
        )
        assert config.stub_type == "weather"
        assert config.deterministic is False
        assert config.seed == 123
        assert config.response_delay == 0.5


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_requires_server_or_stub(self):
        """ValidationConfig raises if neither server nor stub provided."""
        with pytest.raises(ValueError, match="Either stub_config or mcp_command"):
            ValidationConfig(model_path=Path("/test/model"))

    def test_accepts_stub_config(self):
        """ValidationConfig accepts valid stub configuration."""
        stub_config = StubConfig(stub_type="weather")
        config = ValidationConfig(
            model_path=Path("/test/model"),
            stub_config=stub_config,
        )
        assert config.stub_config == stub_config
        assert config.mcp_command is None

    def test_accepts_mcp_command(self):
        """ValidationConfig accepts MCP server command."""
        config = ValidationConfig(
            model_path=Path("/test/model"),
            mcp_command="npx -y @mcp/server-weather",
        )
        assert config.mcp_command == "npx -y @mcp/server-weather"
        assert config.stub_config is None

    def test_accepts_both(self):
        """ValidationConfig accepts both stub and server (stub preferred)."""
        stub_config = StubConfig(stub_type="weather")
        config = ValidationConfig(
            model_path=Path("/test/model"),
            stub_config=stub_config,
            mcp_command="npx server",
        )
        assert config.stub_config == stub_config
        assert config.mcp_command == "npx server"

    def test_default_thresholds(self):
        """ValidationConfig has correct default thresholds."""
        config = ValidationConfig(
            model_path=Path("/test"),
            stub_config=StubConfig(stub_type="weather"),
        )
        assert config.parse_threshold == 0.98
        assert config.schema_threshold == 0.95
        assert config.accuracy_threshold == 0.90
        assert config.loop_threshold == 0.95

    def test_custom_thresholds(self):
        """ValidationConfig accepts custom thresholds."""
        config = ValidationConfig(
            model_path=Path("/test"),
            stub_config=StubConfig(stub_type="weather"),
            parse_threshold=0.99,
            schema_threshold=0.97,
            accuracy_threshold=0.95,
            loop_threshold=0.98,
        )
        assert config.parse_threshold == 0.99
        assert config.schema_threshold == 0.97
        assert config.accuracy_threshold == 0.95
        assert config.loop_threshold == 0.98

    def test_inference_config_default(self):
        """ValidationConfig creates default InferenceConfig."""
        config = ValidationConfig(
            model_path=Path("/test"),
            stub_config=StubConfig(stub_type="weather"),
        )
        assert isinstance(config.inference, InferenceConfig)
        assert config.inference.max_new_tokens == 512

    def test_custom_inference_config(self):
        """ValidationConfig accepts custom InferenceConfig."""
        inference = InferenceConfig(max_new_tokens=1024)
        config = ValidationConfig(
            model_path=Path("/test"),
            stub_config=StubConfig(stub_type="weather"),
            inference=inference,
        )
        assert config.inference.max_new_tokens == 1024

    def test_other_defaults(self):
        """ValidationConfig has other sensible defaults."""
        config = ValidationConfig(
            model_path=Path("/test"),
            stub_config=StubConfig(stub_type="weather"),
        )
        assert config.samples == 20
        assert config.timeout == 30.0
        assert config.retry_count == 3
        assert config.retry_delay == 1.0
