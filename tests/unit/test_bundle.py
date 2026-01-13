"""Unit tests for bundle packaging.

Tests BundleConfig, BundleResult, BundleEngine, and file generation methods.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from mcp_forge.export.bundle import (
    BundleConfig,
    BundleEngine,
    BundleResult,
    verify_bundle,
)
from mcp_forge.state import ToolDefinition


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_tools() -> list[ToolDefinition]:
    """Create sample tool definitions."""
    return [
        ToolDefinition(
            name="get_weather",
            description="Get the current weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        ),
        ToolDefinition(
            name="get_forecast",
            description="Get weather forecast for multiple days",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer", "minimum": 1, "maximum": 7},
                },
                "required": ["location", "days"],
            },
        ),
    ]


@pytest.fixture
def mock_gguf_file(tmp_path: Path) -> Path:
    """Create a mock GGUF file with valid magic bytes."""
    gguf_path = tmp_path / "model.gguf"
    # Write GGUF magic bytes followed by some dummy content
    with open(gguf_path, "wb") as f:
        f.write(b"GGUF")  # Magic bytes
        f.write(b"\x00" * 1024)  # Dummy content
    return gguf_path


@pytest.fixture
def bundle_config(tmp_path: Path, mock_gguf_file: Path, sample_tools: list[ToolDefinition]) -> BundleConfig:
    """Create a BundleConfig for testing."""
    output_dir = tmp_path / "bundle"
    return BundleConfig(
        gguf_path=mock_gguf_file,
        tools=sample_tools,
        output_dir=output_dir,
        model_name="test-agent",
        model_description="Test model for unit tests",
        model_family="deepseek-r1",
        tool_accuracy=0.94,
        schema_conformance=0.97,
        benchmark_score=0.91,
        training_samples=500,
    )


# =============================================================================
# BundleConfig Tests
# =============================================================================


class TestBundleConfig:
    """Tests for BundleConfig dataclass."""

    def test_basic_config(self, tmp_path: Path, sample_tools: list[ToolDefinition]):
        """Test basic BundleConfig initialization."""
        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = BundleConfig(
            gguf_path=gguf_path,
            tools=sample_tools,
            output_dir=tmp_path / "output",
        )

        assert config.gguf_path == gguf_path
        assert len(config.tools) == 2
        assert config.include_modelfile is True
        assert config.include_readme is True

    def test_default_model_name_from_path(self, tmp_path: Path, sample_tools: list[ToolDefinition]):
        """Test that model_name is derived from gguf_path when not specified."""
        gguf_path = tmp_path / "my-model.q8_0.gguf"
        gguf_path.touch()

        config = BundleConfig(
            gguf_path=gguf_path,
            tools=sample_tools,
            output_dir=tmp_path / "output",
        )

        # Should be derived from filename
        assert config.model_name == "my-model-q8_0"

    def test_explicit_model_name(self, tmp_path: Path, sample_tools: list[ToolDefinition]):
        """Test that explicit model_name overrides default."""
        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = BundleConfig(
            gguf_path=gguf_path,
            tools=sample_tools,
            output_dir=tmp_path / "output",
            model_name="custom-name",
        )

        assert config.model_name == "custom-name"

    def test_default_values(self, tmp_path: Path, sample_tools: list[ToolDefinition]):
        """Test default configuration values."""
        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = BundleConfig(
            gguf_path=gguf_path,
            tools=sample_tools,
            output_dir=tmp_path / "output",
        )

        assert config.default_temperature == 0.3
        assert config.default_context_size == 8192
        assert config.model_family == "unknown"
        assert config.training_samples == 0

    def test_optional_metrics(self, tmp_path: Path, sample_tools: list[ToolDefinition]):
        """Test optional quality metrics."""
        gguf_path = tmp_path / "model.gguf"
        gguf_path.touch()

        config = BundleConfig(
            gguf_path=gguf_path,
            tools=sample_tools,
            output_dir=tmp_path / "output",
            tool_accuracy=0.95,
            schema_conformance=0.98,
            benchmark_score=0.92,
        )

        assert config.tool_accuracy == 0.95
        assert config.schema_conformance == 0.98
        assert config.benchmark_score == 0.92


# =============================================================================
# BundleResult Tests
# =============================================================================


class TestBundleResult:
    """Tests for BundleResult dataclass."""

    def test_successful_result(self, tmp_path: Path):
        """Test successful bundle result."""
        result = BundleResult(
            success=True,
            bundle_path=tmp_path,
            files_created=["model.gguf", "tools.json", "config.yaml"],
            bundle_size_mb=100.5,
            validation_passed=True,
        )

        assert result.success is True
        assert result.bundle_path == tmp_path
        assert len(result.files_created) == 3
        assert result.bundle_size_mb == 100.5
        assert result.validation_passed is True
        assert result.error is None

    def test_failed_result(self):
        """Test failed bundle result."""
        result = BundleResult(
            success=False,
            bundle_path=None,
            validation_errors=["Missing model.gguf"],
            error="File not found",
        )

        assert result.success is False
        assert result.bundle_path is None
        assert "Missing model.gguf" in result.validation_errors
        assert result.error == "File not found"

    def test_to_dict(self, tmp_path: Path):
        """Test BundleResult serialization."""
        result = BundleResult(
            success=True,
            bundle_path=tmp_path,
            files_created=["model.gguf"],
            bundle_size_mb=50.0,
            validation_passed=True,
        )

        data = result.to_dict()
        assert data["success"] is True
        assert data["bundle_path"] == str(tmp_path)
        assert data["files_created"] == ["model.gguf"]
        assert data["bundle_size_mb"] == 50.0


# =============================================================================
# BundleEngine Core Tests
# =============================================================================


class TestBundleEngine:
    """Tests for BundleEngine core functionality."""

    def test_engine_initialization(self, bundle_config: BundleConfig):
        """Test BundleEngine initialization."""
        engine = BundleEngine(bundle_config)
        assert engine.config == bundle_config

    def test_full_package(self, bundle_config: BundleConfig):
        """Test full bundle packaging."""
        engine = BundleEngine(bundle_config)
        result = engine.package()

        assert result.success is True
        assert result.bundle_path == bundle_config.output_dir
        assert "model.gguf" in result.files_created
        assert "tools.json" in result.files_created
        assert "config.yaml" in result.files_created
        assert "README.md" in result.files_created
        assert "Modelfile" in result.files_created

    def test_package_without_readme(self, bundle_config: BundleConfig):
        """Test packaging without README."""
        bundle_config.include_readme = False
        engine = BundleEngine(bundle_config)
        result = engine.package()

        assert result.success is True
        assert "README.md" not in result.files_created

    def test_package_without_modelfile(self, bundle_config: BundleConfig):
        """Test packaging without Modelfile."""
        bundle_config.include_modelfile = False
        engine = BundleEngine(bundle_config)
        result = engine.package()

        assert result.success is True
        assert "Modelfile" not in result.files_created

    def test_progress_callback(self, bundle_config: BundleConfig):
        """Test progress callback is called."""
        engine = BundleEngine(bundle_config)
        messages: list[str] = []

        def on_progress(msg: str) -> None:
            messages.append(msg)

        engine.package(progress_callback=on_progress)

        assert len(messages) > 0
        assert any("Creating" in m for m in messages)
        assert any("Copying" in m for m in messages)
        assert any("Validating" in m for m in messages)


# =============================================================================
# tools.json Generation Tests
# =============================================================================


class TestToolsJsonGeneration:
    """Tests for tools.json generation."""

    def test_tools_json_structure(self, bundle_config: BundleConfig):
        """Test tools.json has correct structure."""
        engine = BundleEngine(bundle_config)
        engine.package()

        tools_path = bundle_config.output_dir / "tools.json"
        assert tools_path.exists()

        with open(tools_path) as f:
            data = json.load(f)

        assert "version" in data
        assert data["version"] == "1.0"
        assert "tools" in data
        assert "metadata" in data

    def test_tools_json_tool_format(self, bundle_config: BundleConfig):
        """Test tools.json tools have OpenAI function format."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "tools.json") as f:
            data = json.load(f)

        assert len(data["tools"]) == 2

        tool = data["tools"][0]
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]

    def test_tools_json_metadata(self, bundle_config: BundleConfig):
        """Test tools.json metadata."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "tools.json") as f:
            data = json.load(f)

        metadata = data["metadata"]
        assert metadata["source"] == "mcp-forge"
        assert metadata["tool_count"] == 2
        assert "generated_at" in metadata


# =============================================================================
# config.yaml Generation Tests
# =============================================================================


class TestConfigYamlGeneration:
    """Tests for config.yaml generation."""

    def test_config_yaml_structure(self, bundle_config: BundleConfig):
        """Test config.yaml has correct structure."""
        engine = BundleEngine(bundle_config)
        engine.package()

        config_path = bundle_config.output_dir / "config.yaml"
        assert config_path.exists()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        assert "model" in data
        assert "inference" in data
        assert "tools" in data
        assert "deployment" in data

    def test_config_yaml_model_section(self, bundle_config: BundleConfig):
        """Test config.yaml model section."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "config.yaml") as f:
            data = yaml.safe_load(f)

        model = data["model"]
        assert model["name"] == "test-agent"
        assert model["file"] == "model.gguf"
        assert model["family"] == "deepseek-r1"

    def test_config_yaml_inference_section(self, bundle_config: BundleConfig):
        """Test config.yaml inference section."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "config.yaml") as f:
            data = yaml.safe_load(f)

        inference = data["inference"]
        assert inference["temperature"] == 0.3
        assert inference["context_size"] == 8192
        assert "<|im_end|>" in inference["stop_sequences"]

    def test_config_yaml_quality_metrics(self, bundle_config: BundleConfig):
        """Test config.yaml includes quality metrics."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "config.yaml") as f:
            data = yaml.safe_load(f)

        quality = data["quality"]
        assert quality["tool_accuracy"] == 0.94
        assert quality["schema_conformance"] == 0.97
        assert quality["benchmark_score"] == 0.91

    def test_config_yaml_deployment_commands(self, bundle_config: BundleConfig):
        """Test config.yaml deployment commands."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "config.yaml") as f:
            data = yaml.safe_load(f)

        deployment = data["deployment"]
        assert "ollama" in deployment
        assert "llama_cpp" in deployment
        assert "create_command" in deployment["ollama"]


# =============================================================================
# README.md Generation Tests
# =============================================================================


class TestReadmeGeneration:
    """Tests for README.md generation."""

    def test_readme_created(self, bundle_config: BundleConfig):
        """Test README.md is created."""
        engine = BundleEngine(bundle_config)
        engine.package()

        readme_path = bundle_config.output_dir / "README.md"
        assert readme_path.exists()

    def test_readme_contains_model_name(self, bundle_config: BundleConfig):
        """Test README.md contains model name."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "README.md") as f:
            content = f.read()

        assert "test-agent" in content

    def test_readme_contains_tools_table(self, bundle_config: BundleConfig):
        """Test README.md contains tools table."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "README.md") as f:
            content = f.read()

        assert "| Tool | Description |" in content
        assert "get_weather" in content
        assert "get_forecast" in content

    def test_readme_contains_quality_metrics(self, bundle_config: BundleConfig):
        """Test README.md contains quality metrics."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "README.md") as f:
            content = f.read()

        assert "Tool Selection Accuracy" in content
        assert "94%" in content

    def test_readme_contains_usage_instructions(self, bundle_config: BundleConfig):
        """Test README.md contains usage instructions."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "README.md") as f:
            content = f.read()

        assert "ollama create" in content
        assert "llama-cli" in content


# =============================================================================
# Modelfile Generation Tests
# =============================================================================


class TestModelfileGeneration:
    """Tests for Ollama Modelfile generation."""

    def test_modelfile_created(self, bundle_config: BundleConfig):
        """Test Modelfile is created."""
        engine = BundleEngine(bundle_config)
        engine.package()

        modelfile_path = bundle_config.output_dir / "Modelfile"
        assert modelfile_path.exists()

    def test_modelfile_from_directive(self, bundle_config: BundleConfig):
        """Test Modelfile has FROM directive."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "Modelfile") as f:
            content = f.read()

        assert "FROM ./model.gguf" in content

    def test_modelfile_system_prompt(self, bundle_config: BundleConfig):
        """Test Modelfile has SYSTEM directive."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "Modelfile") as f:
            content = f.read()

        assert "SYSTEM" in content
        assert "tool_call" in content

    def test_modelfile_parameters(self, bundle_config: BundleConfig):
        """Test Modelfile has inference parameters."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "Modelfile") as f:
            content = f.read()

        assert "PARAMETER temperature" in content
        assert "PARAMETER num_ctx" in content
        assert "PARAMETER stop" in content

    def test_modelfile_template(self, bundle_config: BundleConfig):
        """Test Modelfile has ChatML template."""
        engine = BundleEngine(bundle_config)
        engine.package()

        with open(bundle_config.output_dir / "Modelfile") as f:
            content = f.read()

        assert "TEMPLATE" in content
        assert "<|im_start|>" in content
        assert "<|im_end|>" in content


# =============================================================================
# Bundle Validation Tests
# =============================================================================


class TestBundleValidation:
    """Tests for bundle validation."""

    def test_valid_bundle_passes(self, bundle_config: BundleConfig):
        """Test valid bundle passes validation."""
        engine = BundleEngine(bundle_config)
        result = engine.package()

        assert result.validation_passed is True
        assert len(result.validation_errors) == 0

    def test_missing_required_file(self, bundle_config: BundleConfig):
        """Test validation catches missing required files."""
        engine = BundleEngine(bundle_config)
        engine.package()

        # Remove a required file
        (bundle_config.output_dir / "tools.json").unlink()

        valid, errors = engine._validate_bundle()
        assert valid is False
        assert any("tools.json" in e for e in errors)

    def test_invalid_json_detected(self, bundle_config: BundleConfig):
        """Test validation catches invalid JSON."""
        engine = BundleEngine(bundle_config)
        engine.package()

        # Corrupt tools.json
        with open(bundle_config.output_dir / "tools.json", "w") as f:
            f.write("not valid json {")

        valid, errors = engine._validate_bundle()
        assert valid is False
        assert any("JSON" in e for e in errors)

    def test_invalid_yaml_detected(self, bundle_config: BundleConfig):
        """Test validation catches invalid YAML."""
        engine = BundleEngine(bundle_config)
        engine.package()

        # Corrupt config.yaml
        with open(bundle_config.output_dir / "config.yaml", "w") as f:
            f.write("invalid: yaml: content: [")

        valid, errors = engine._validate_bundle()
        assert valid is False
        assert any("YAML" in e for e in errors)

    def test_tool_count_mismatch(self, bundle_config: BundleConfig):
        """Test validation catches tool count mismatch."""
        engine = BundleEngine(bundle_config)
        engine.package()

        # Modify tool count in config
        config_path = bundle_config.output_dir / "config.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        data["tools"]["count"] = 10  # Wrong count
        with open(config_path, "w") as f:
            yaml.dump(data, f)

        valid, errors = engine._validate_bundle()
        assert valid is False
        assert any("mismatch" in e for e in errors)


# =============================================================================
# verify_bundle Function Tests
# =============================================================================


class TestVerifyBundle:
    """Tests for verify_bundle standalone function."""

    def test_verify_valid_bundle(self, bundle_config: BundleConfig):
        """Test verify_bundle on valid bundle."""
        engine = BundleEngine(bundle_config)
        engine.package()

        valid, errors = verify_bundle(bundle_config.output_dir)
        assert valid is True
        assert len(errors) == 0

    def test_verify_missing_file(self, bundle_config: BundleConfig):
        """Test verify_bundle detects missing files."""
        engine = BundleEngine(bundle_config)
        engine.package()

        (bundle_config.output_dir / "config.yaml").unlink()

        valid, errors = verify_bundle(bundle_config.output_dir)
        assert valid is False
        assert any("config.yaml" in e for e in errors)

    def test_verify_invalid_gguf_magic(self, bundle_config: BundleConfig):
        """Test verify_bundle detects invalid GGUF magic bytes."""
        engine = BundleEngine(bundle_config)
        engine.package()

        # Corrupt model.gguf magic bytes
        model_path = bundle_config.output_dir / "model.gguf"
        with open(model_path, "wb") as f:
            f.write(b"XXXX")  # Wrong magic

        valid, errors = verify_bundle(bundle_config.output_dir)
        assert valid is False
        assert any("magic bytes" in e for e in errors)

    def test_verify_empty_directory(self, tmp_path: Path):
        """Test verify_bundle on empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        valid, errors = verify_bundle(empty_dir)
        assert valid is False
        assert len(errors) >= 3  # Missing model.gguf, tools.json, config.yaml


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_tools_list(self, tmp_path: Path, mock_gguf_file: Path):
        """Test bundle with empty tools list."""
        config = BundleConfig(
            gguf_path=mock_gguf_file,
            tools=[],
            output_dir=tmp_path / "bundle",
        )

        engine = BundleEngine(config)
        result = engine.package()

        assert result.success is True
        # tools.json should still be valid
        with open(config.output_dir / "tools.json") as f:
            data = json.load(f)
        assert data["metadata"]["tool_count"] == 0

    def test_missing_gguf_file(self, tmp_path: Path, sample_tools: list[ToolDefinition]):
        """Test error when GGUF file doesn't exist."""
        config = BundleConfig(
            gguf_path=tmp_path / "nonexistent.gguf",
            tools=sample_tools,
            output_dir=tmp_path / "bundle",
        )

        engine = BundleEngine(config)
        result = engine.package()

        assert result.success is False
        assert result.error is not None

    def test_long_tool_description_truncation(self, tmp_path: Path, mock_gguf_file: Path):
        """Test README handles long tool descriptions."""
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A" * 200,  # Very long description
                input_schema={"type": "object", "properties": {}},
            )
        ]

        config = BundleConfig(
            gguf_path=mock_gguf_file,
            tools=tools,
            output_dir=tmp_path / "bundle",
        )

        engine = BundleEngine(config)
        result = engine.package()

        assert result.success is True
        # README should truncate long descriptions
        with open(config.output_dir / "README.md") as f:
            content = f.read()
        assert "..." in content  # Should be truncated

    def test_special_characters_in_name(self, tmp_path: Path, mock_gguf_file: Path, sample_tools: list[ToolDefinition]):
        """Test bundle handles special characters in model name."""
        config = BundleConfig(
            gguf_path=mock_gguf_file,
            tools=sample_tools,
            output_dir=tmp_path / "bundle",
            model_name="test model (v1.0)",
        )

        engine = BundleEngine(config)
        result = engine.package()

        # Should succeed (special chars are allowed)
        assert result.success is True
