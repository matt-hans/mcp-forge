"""Unit tests for ValidationRunner class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_forge.state import ToolDefinition, ValidationResult
from mcp_forge.validation.config import StubConfig, ValidationConfig
from mcp_forge.validation.runner import (
    ValidationRunner,
    ValidationSample,
    generate_validation_samples,
)


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
def validation_config(tmp_path: Path, stub_config: StubConfig) -> ValidationConfig:
    """Standard validation config for tests."""
    return ValidationConfig(
        model_path=tmp_path / "adapter",
        samples=10,
        stub_config=stub_config,
    )


class TestValidationSample:
    """Tests for ValidationSample dataclass."""

    def test_sample_with_tool(self):
        """Sample with expected tool."""
        sample = ValidationSample(
            prompt="What's the weather in Paris?",
            expected_tool="get_weather",
            expected_args={"location": "Paris"},
        )
        assert sample.prompt == "What's the weather in Paris?"
        assert sample.expected_tool == "get_weather"
        assert sample.expected_args == {"location": "Paris"}

    def test_sample_without_tool(self):
        """Sample expecting no tool call."""
        sample = ValidationSample(
            prompt="What is the capital of France?",
            expected_tool=None,
            expected_args=None,
        )
        assert sample.expected_tool is None
        assert sample.expected_args is None


class TestValidationRunner:
    """Tests for ValidationRunner class."""

    def test_initialization(
        self, validation_config: ValidationConfig, weather_tool: ToolDefinition
    ):
        """Runner initializes with config and tools."""
        runner = ValidationRunner(validation_config, [weather_tool])
        assert runner.config == validation_config
        assert runner.tools == [weather_tool]
        assert runner.model is None
        assert runner.tokenizer is None

    def test_generate_response_requires_model(
        self, validation_config: ValidationConfig, weather_tool: ToolDefinition
    ):
        """generate_response raises without loaded model."""
        runner = ValidationRunner(validation_config, [weather_tool])
        with pytest.raises(RuntimeError, match="Model must be loaded"):
            runner.generate_response("test prompt")

    def test_get_stub_returns_correct_type(
        self, validation_config: ValidationConfig, weather_tool: ToolDefinition
    ):
        """_get_stub returns correct stub type."""
        runner = ValidationRunner(validation_config, [weather_tool])
        stub = runner._get_stub()
        assert stub.__class__.__name__ == "WeatherStub"

    def test_get_stub_caches(
        self, validation_config: ValidationConfig, weather_tool: ToolDefinition
    ):
        """_get_stub returns same instance on repeated calls."""
        runner = ValidationRunner(validation_config, [weather_tool])
        stub1 = runner._get_stub()
        stub2 = runner._get_stub()
        assert stub1 is stub2

    def test_get_stub_raises_without_config(
        self, weather_tool: ToolDefinition, tmp_path: Path
    ):
        """_get_stub raises when no stub_config provided."""
        config = ValidationConfig(
            model_path=tmp_path / "adapter",
            mcp_command="npx server",
        )
        runner = ValidationRunner(config, [weather_tool])
        with pytest.raises(ValueError, match="No stub configuration"):
            runner._get_stub()


class TestGenerateValidationSamples:
    """Tests for generate_validation_samples function."""

    def test_generates_correct_count(self, weather_tool: ToolDefinition):
        """Generates requested number of samples."""
        samples = generate_validation_samples([weather_tool], count=10)
        assert len(samples) == 10

    def test_includes_weather_samples(self, weather_tool: ToolDefinition):
        """Generates weather-specific prompts."""
        samples = generate_validation_samples([weather_tool], count=20, seed=42)
        weather_samples = [s for s in samples if s.expected_tool == "get_weather"]
        assert len(weather_samples) >= 5

    def test_includes_no_tool_samples(self, weather_tool: ToolDefinition):
        """Includes no-tool test cases by default."""
        samples = generate_validation_samples([weather_tool], count=20, seed=42)
        no_tool_samples = [s for s in samples if s.expected_tool is None]
        assert len(no_tool_samples) >= 1

    def test_excludes_no_tool_when_disabled(self, weather_tool: ToolDefinition):
        """Excludes no-tool samples when include_no_tool=False."""
        samples = generate_validation_samples(
            [weather_tool], count=10, include_no_tool=False
        )
        no_tool_samples = [s for s in samples if s.expected_tool is None]
        assert len(no_tool_samples) == 0

    def test_deterministic_with_seed(self, weather_tool: ToolDefinition):
        """Same seed produces identical samples."""
        samples1 = generate_validation_samples([weather_tool], count=10, seed=42)
        samples2 = generate_validation_samples([weather_tool], count=10, seed=42)
        prompts1 = [s.prompt for s in samples1]
        prompts2 = [s.prompt for s in samples2]
        assert prompts1 == prompts2

    def test_different_seed_different_samples(self, weather_tool: ToolDefinition):
        """Different seeds produce different sample order."""
        samples1 = generate_validation_samples([weather_tool], count=10, seed=42)
        samples2 = generate_validation_samples([weather_tool], count=10, seed=123)
        prompts1 = [s.prompt for s in samples1]
        prompts2 = [s.prompt for s in samples2]
        # Order should differ (prompts themselves may overlap)
        assert prompts1 != prompts2

    def test_covers_multiple_tools(
        self, weather_tool: ToolDefinition, filesystem_tool: ToolDefinition
    ):
        """Generates samples for multiple tools."""
        samples = generate_validation_samples(
            [weather_tool, filesystem_tool], count=20, include_no_tool=False
        )
        weather_samples = [s for s in samples if s.expected_tool == "get_weather"]
        fs_samples = [s for s in samples if s.expected_tool == "list_files"]
        assert len(weather_samples) >= 5
        assert len(fs_samples) >= 5

    def test_generic_prompts_for_unknown_tools(self):
        """Generates generic prompts for unknown tool types."""
        unknown_tool = ToolDefinition(
            name="custom_tool",
            description="A custom tool",
            input_schema={"type": "object", "properties": {}},
        )
        samples = generate_validation_samples(
            [unknown_tool], count=10, include_no_tool=False
        )
        for sample in samples:
            assert sample.expected_tool == "custom_tool"
            assert "custom_tool" in sample.prompt

    def test_empty_tools_list(self):
        """Handles empty tools list gracefully."""
        samples = generate_validation_samples([], count=5)
        # Should only have no-tool samples
        for sample in samples:
            assert sample.expected_tool is None


class TestValidationRunnerWithMockedModel:
    """Tests for ValidationRunner with mocked model inference."""

    @pytest.fixture
    def mock_unsloth(self):
        """Mock Unsloth FastLanguageModel."""
        with patch("mcp_forge.validation.runner.FastLanguageModel") as mock:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock.from_pretrained.return_value = (mock_model, mock_tokenizer)
            yield mock, mock_model, mock_tokenizer

    def test_validate_single_no_tool_correct(
        self,
        validation_config: ValidationConfig,
        weather_tool: ToolDefinition,
    ):
        """validate_single handles correct no-tool response."""
        runner = ValidationRunner(validation_config, [weather_tool])

        # Mock generate_response to return no tool call
        runner.generate_response = MagicMock(return_value="The capital is Paris.")

        sample = ValidationSample(
            prompt="What is the capital of France?",
            expected_tool=None,
            expected_args=None,
        )

        result = runner.validate_single(sample)
        assert result["parsed"] is True
        assert result["schema_valid"] is True
        assert result["tool_correct"] is True
        assert result["loop_complete"] is True
        assert result["error"] is None

    def test_validate_single_tool_call_parsed(
        self,
        validation_config: ValidationConfig,
        weather_tool: ToolDefinition,
    ):
        """validate_single parses valid tool call."""
        runner = ValidationRunner(validation_config, [weather_tool])

        # Mock generate_response to return valid tool call
        tool_response = """<tool_call>
{"name": "get_weather", "arguments": {"location": "Paris"}}
</tool_call>"""
        runner.generate_response = MagicMock(return_value=tool_response)

        sample = ValidationSample(
            prompt="What's the weather in Paris?",
            expected_tool="get_weather",
            expected_args={"location": "Paris"},
        )

        result = runner.validate_single(sample)
        assert result["parsed"] is True
        assert result["tool_call"]["name"] == "get_weather"
        assert result["schema_valid"] is True
        assert result["tool_correct"] is True
        assert result["loop_complete"] is True

    def test_validate_single_missing_tool_call(
        self,
        validation_config: ValidationConfig,
        weather_tool: ToolDefinition,
    ):
        """validate_single reports error when tool expected but not found."""
        runner = ValidationRunner(validation_config, [weather_tool])

        # Mock generate_response to return no tool call when one was expected
        runner.generate_response = MagicMock(return_value="I'll check the weather for you.")

        sample = ValidationSample(
            prompt="What's the weather in Paris?",
            expected_tool="get_weather",
            expected_args={"location": "Paris"},
        )

        result = runner.validate_single(sample)
        assert result["parsed"] is False
        assert result["error"] == "No tool call found in response"

    def test_validate_single_wrong_tool(
        self,
        validation_config: ValidationConfig,
        weather_tool: ToolDefinition,
        filesystem_tool: ToolDefinition,
    ):
        """validate_single detects wrong tool selection."""
        # Use weather stub config but have both tools
        runner = ValidationRunner(validation_config, [weather_tool, filesystem_tool])

        # Mock generate_response to return wrong tool
        tool_response = """<tool_call>
{"name": "list_files", "arguments": {"path": "/home"}}
</tool_call>"""
        runner.generate_response = MagicMock(return_value=tool_response)

        sample = ValidationSample(
            prompt="What's the weather in Paris?",
            expected_tool="get_weather",
            expected_args={"location": "Paris"},
        )

        result = runner.validate_single(sample)
        assert result["parsed"] is True
        assert result["tool_correct"] is False

    def test_run_aggregates_metrics(
        self,
        validation_config: ValidationConfig,
        weather_tool: ToolDefinition,
    ):
        """run() aggregates metrics across all samples."""
        runner = ValidationRunner(validation_config, [weather_tool])

        # Mock generate_response with alternating responses
        responses = [
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>',
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n</tool_call>',
            "I don't have weather data for that.",
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "London"}}\n</tool_call>',
        ]
        runner.generate_response = MagicMock(side_effect=responses)

        # Skip model loading
        runner.model = MagicMock()
        runner.tokenizer = MagicMock()

        samples = [
            ValidationSample("Weather Paris?", "get_weather", {"location": "Paris"}),
            ValidationSample("Weather Tokyo?", "get_weather", {"location": "Tokyo"}),
            ValidationSample("What's up?", None, None),  # No tool expected
            ValidationSample("Weather London?", "get_weather", {"location": "London"}),
        ]

        result = runner.run(samples)

        assert isinstance(result, ValidationResult)
        assert result.samples_tested == 4
        # 3 successful tool calls + 1 correct no-tool response
        assert result.tool_call_parse_rate >= 0.5

    def test_run_calls_progress_callback(
        self,
        validation_config: ValidationConfig,
        weather_tool: ToolDefinition,
    ):
        """run() calls progress callback with correct values."""
        runner = ValidationRunner(validation_config, [weather_tool])
        runner.model = MagicMock()
        runner.tokenizer = MagicMock()
        runner.generate_response = MagicMock(return_value="Response")

        progress_calls = []

        def track_progress(current: int, total: int, pct: float):
            progress_calls.append((current, total, pct))

        samples = [
            ValidationSample("Test 1", None, None),
            ValidationSample("Test 2", None, None),
        ]

        runner.run(samples, progress_callback=track_progress)

        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2, 0.5)
        assert progress_calls[1] == (2, 2, 1.0)
