"""Integration tests for bundle packaging pipeline.

Tests CLI commands and pipeline stage integration.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from mcp_forge.cli import cli
from mcp_forge.export.bundle import BundleConfig, BundleEngine, verify_bundle
from mcp_forge.state import (
    BenchmarkResult,
    PipelineStage,
    PipelineState,
    StateManager,
    ToolDefinition,
    ValidationResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


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
                },
                "required": ["location"],
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
def tools_json_file(tmp_path: Path, sample_tools: list[ToolDefinition]) -> Path:
    """Create a tools.json file."""
    tools_path = tmp_path / "tools.json"
    with open(tools_path, "w") as f:
        json.dump([t.to_dict() for t in sample_tools], f)
    return tools_path


@pytest.fixture
def state_manager(tmp_path: Path) -> StateManager:
    """Create a StateManager for testing."""
    manager = StateManager(tmp_path)
    manager.ensure_dirs()
    return manager


# =============================================================================
# CLI pack Command Tests
# =============================================================================


class TestPackCommand:
    """Tests for mcp-forge pack command."""

    def test_pack_basic(self, runner: CliRunner, mock_gguf_file: Path, tools_json_file: Path, tmp_path: Path):
        """Test basic pack command."""
        output_dir = tmp_path / "bundle"

        result = runner.invoke(
            cli,
            [
                "pack",
                "-m", str(mock_gguf_file),
                "-t", str(tools_json_file),
                "-o", str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert output_dir.exists()
        assert (output_dir / "model.gguf").exists()
        assert (output_dir / "tools.json").exists()
        assert (output_dir / "config.yaml").exists()

    def test_pack_with_name(self, runner: CliRunner, mock_gguf_file: Path, tools_json_file: Path, tmp_path: Path):
        """Test pack command with custom name."""
        output_dir = tmp_path / "bundle"

        result = runner.invoke(
            cli,
            [
                "pack",
                "-m", str(mock_gguf_file),
                "-t", str(tools_json_file),
                "-o", str(output_dir),
                "--name", "my-custom-agent",
            ],
        )

        assert result.exit_code == 0

        with open(output_dir / "config.yaml") as f:
            config = yaml.safe_load(f)
        assert config["model"]["name"] == "my-custom-agent"

    def test_pack_no_modelfile(self, runner: CliRunner, mock_gguf_file: Path, tools_json_file: Path, tmp_path: Path):
        """Test pack command with --no-modelfile."""
        output_dir = tmp_path / "bundle"

        result = runner.invoke(
            cli,
            [
                "pack",
                "-m", str(mock_gguf_file),
                "-t", str(tools_json_file),
                "-o", str(output_dir),
                "--no-modelfile",
            ],
        )

        assert result.exit_code == 0
        assert not (output_dir / "Modelfile").exists()

    def test_pack_no_readme(self, runner: CliRunner, mock_gguf_file: Path, tools_json_file: Path, tmp_path: Path):
        """Test pack command with --no-readme."""
        output_dir = tmp_path / "bundle"

        result = runner.invoke(
            cli,
            [
                "pack",
                "-m", str(mock_gguf_file),
                "-t", str(tools_json_file),
                "-o", str(output_dir),
                "--no-readme",
            ],
        )

        assert result.exit_code == 0
        assert not (output_dir / "README.md").exists()

    def test_pack_missing_tools_no_state(self, runner: CliRunner, mock_gguf_file: Path, tmp_path: Path):
        """Test pack command fails without tools and no state."""
        output_dir = tmp_path / "bundle"

        result = runner.invoke(
            cli,
            [
                "pack",
                "-m", str(mock_gguf_file),
                "-o", str(output_dir),
            ],
        )

        assert result.exit_code == 1
        assert "No tools specified" in result.output

    def test_pack_from_state(
        self,
        runner: CliRunner,
        mock_gguf_file: Path,
        sample_tools: list[ToolDefinition],
        tmp_path: Path,
    ):
        """Test pack command uses tools from pipeline state."""
        output_dir = tmp_path / "bundle"

        # Create pipeline state with tools
        state_manager = StateManager(tmp_path)
        state = state_manager.create_session(
            mcp_command="test",
            system_prompt="test",
            model_family="deepseek-r1",
            output_path=str(output_dir),
        )
        state.tools = sample_tools
        state_manager.save_state(state)

        # Run pack without --tools flag
        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create .mcp-forge directory and copy state
            mcp_forge_dir = Path(".mcp-forge")
            mcp_forge_dir.mkdir(exist_ok=True)
            import shutil
            shutil.copy(
                state_manager.state_file,
                mcp_forge_dir / "state.json"
            )

            result = runner.invoke(
                cli,
                [
                    "pack",
                    "-m", str(mock_gguf_file),
                    "-o", str(output_dir),
                ],
            )

        assert result.exit_code == 0
        assert "Using tools from pipeline state" in result.output


# =============================================================================
# CLI verify-bundle Command Tests
# =============================================================================


class TestVerifyBundleCommand:
    """Tests for mcp-forge verify-bundle command."""

    def test_verify_valid_bundle(self, runner: CliRunner, mock_gguf_file: Path, sample_tools: list[ToolDefinition], tmp_path: Path):
        """Test verify-bundle on valid bundle."""
        # First create a bundle
        config = BundleConfig(
            gguf_path=mock_gguf_file,
            tools=sample_tools,
            output_dir=tmp_path / "bundle",
        )
        engine = BundleEngine(config)
        engine.package()

        # Then verify it
        result = runner.invoke(
            cli,
            ["verify-bundle", str(tmp_path / "bundle")],
        )

        assert result.exit_code == 0
        assert "PASSED" in result.output

    def test_verify_invalid_bundle(self, runner: CliRunner, tmp_path: Path):
        """Test verify-bundle on invalid bundle."""
        # Create incomplete bundle
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "tools.json").write_text("{}")

        result = runner.invoke(
            cli,
            ["verify-bundle", str(bundle_dir)],
        )

        assert result.exit_code == 1
        assert "FAILED" in result.output

    def test_verify_shows_contents(self, runner: CliRunner, mock_gguf_file: Path, sample_tools: list[ToolDefinition], tmp_path: Path):
        """Test verify-bundle shows bundle contents."""
        config = BundleConfig(
            gguf_path=mock_gguf_file,
            tools=sample_tools,
            output_dir=tmp_path / "bundle",
        )
        engine = BundleEngine(config)
        engine.package()

        result = runner.invoke(
            cli,
            ["verify-bundle", str(tmp_path / "bundle")],
        )

        assert result.exit_code == 0
        assert "model.gguf" in result.output
        assert "tools.json" in result.output
        assert "config.yaml" in result.output


# =============================================================================
# Pipeline Integration Tests
# =============================================================================


class TestPipelineIntegration:
    """Tests for bundle packaging in the full pipeline."""

    def test_state_tracking(self, state_manager: StateManager, sample_tools: list[ToolDefinition]):
        """Test pipeline state tracks bundle path."""
        state = state_manager.create_session(
            mcp_command="test",
            system_prompt="test",
            model_family="deepseek-r1",
            output_path="./dist",
        )

        # Simulate progression through pipeline
        state.tools = sample_tools
        state.stage = PipelineStage.PACKAGING
        state.gguf_path = "./model.gguf"
        state.bundle_path = "./dist"
        state_manager.save_state(state)

        # Reload and verify
        loaded = state_manager.load_state()
        assert loaded.bundle_path == "./dist"
        assert loaded.stage == PipelineStage.PACKAGING

    def test_metrics_propagation(self, mock_gguf_file: Path, sample_tools: list[ToolDefinition], tmp_path: Path):
        """Test quality metrics propagate to bundle."""
        config = BundleConfig(
            gguf_path=mock_gguf_file,
            tools=sample_tools,
            output_dir=tmp_path / "bundle",
            tool_accuracy=0.95,
            schema_conformance=0.98,
            benchmark_score=0.92,
        )

        engine = BundleEngine(config)
        engine.package()

        # Check metrics in config.yaml
        with open(tmp_path / "bundle" / "config.yaml") as f:
            data = yaml.safe_load(f)

        assert data["quality"]["tool_accuracy"] == 0.95
        assert data["quality"]["schema_conformance"] == 0.98
        assert data["quality"]["benchmark_score"] == 0.92

        # Check metrics in README
        with open(tmp_path / "bundle" / "README.md") as f:
            content = f.read()

        assert "95%" in content
        assert "98%" in content
        assert "92%" in content

    def test_stage_progression(self, state_manager: StateManager, sample_tools: list[ToolDefinition]):
        """Test proper stage progression to packaging."""
        state = state_manager.create_session(
            mcp_command="test",
            system_prompt="test",
            model_family="deepseek-r1",
            output_path="./dist",
        )

        # Verify PACKAGING comes after EXPORTING
        stages = list(PipelineStage)
        exporting_idx = stages.index(PipelineStage.EXPORTING)
        packaging_idx = stages.index(PipelineStage.PACKAGING)
        assert packaging_idx == exporting_idx + 1

    def test_full_bundle_workflow(
        self,
        mock_gguf_file: Path,
        sample_tools: list[ToolDefinition],
        tmp_path: Path,
        state_manager: StateManager,
    ):
        """Test complete bundle workflow from state."""
        # Create state simulating completed pipeline
        state = state_manager.create_session(
            mcp_command="npx @mcp/server",
            system_prompt="You are helpful",
            model_family="deepseek-r1",
            output_path=str(tmp_path / "bundle"),
        )
        state.tools = sample_tools
        state.gguf_path = str(mock_gguf_file)
        state.stage = PipelineStage.EXPORTING

        # Add validation result
        state.validation_result = ValidationResult(
            passed=True,
            samples_tested=20,
            samples_passed=19,
            tool_call_parse_rate=0.95,
            schema_conformance_rate=0.97,
            tool_selection_accuracy=0.93,
            loop_completion_rate=0.96,
        )

        # Add benchmark result
        state.benchmark_result = BenchmarkResult(
            model_name="test-model",
            timestamp="2026-01-13T00:00:00Z",
            overall_score=0.91,
        )

        state_manager.save_state(state)

        # Create bundle using state
        config = BundleConfig(
            gguf_path=Path(state.gguf_path),
            tools=state.tools,
            output_dir=Path(state.output_path),
            model_name=f"mcp-forge-{state.model_family}",
            model_family=state.model_family,
            tool_accuracy=state.validation_result.tool_selection_accuracy,
            schema_conformance=state.validation_result.schema_conformance_rate,
            benchmark_score=state.benchmark_result.overall_score,
        )

        engine = BundleEngine(config)
        result = engine.package()

        # Update state
        state.bundle_path = str(result.bundle_path)
        state.stage = PipelineStage.COMPLETE
        state_manager.save_state(state)

        # Verify final state
        final_state = state_manager.load_state()
        assert final_state.stage == PipelineStage.COMPLETE
        assert final_state.bundle_path is not None

        # Verify bundle contents
        bundle_path = Path(final_state.bundle_path)
        assert (bundle_path / "model.gguf").exists()
        assert (bundle_path / "tools.json").exists()

        # Verify metrics in bundle
        with open(bundle_path / "config.yaml") as f:
            config_data = yaml.safe_load(f)
        assert config_data["quality"]["tool_accuracy"] == 0.93


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in bundle packaging."""

    def test_pack_nonexistent_model(self, runner: CliRunner, tools_json_file: Path, tmp_path: Path):
        """Test pack command with nonexistent model file."""
        result = runner.invoke(
            cli,
            [
                "pack",
                "-m", str(tmp_path / "nonexistent.gguf"),
                "-t", str(tools_json_file),
                "-o", str(tmp_path / "bundle"),
            ],
        )

        # Click should catch this with exists=True
        assert result.exit_code != 0

    def test_pack_invalid_tools_json(self, runner: CliRunner, mock_gguf_file: Path, tmp_path: Path):
        """Test pack command with invalid tools.json."""
        tools_path = tmp_path / "invalid_tools.json"
        tools_path.write_text("not valid json")

        result = runner.invoke(
            cli,
            [
                "pack",
                "-m", str(mock_gguf_file),
                "-t", str(tools_path),
                "-o", str(tmp_path / "bundle"),
            ],
        )

        assert result.exit_code != 0

    def test_verify_nonexistent_bundle(self, runner: CliRunner, tmp_path: Path):
        """Test verify-bundle with nonexistent path."""
        result = runner.invoke(
            cli,
            ["verify-bundle", str(tmp_path / "nonexistent")],
        )

        # Click should catch this with exists=True
        assert result.exit_code != 0

    def test_bundle_creation_failure_recovery(
        self,
        sample_tools: list[ToolDefinition],
        tmp_path: Path,
    ):
        """Test bundle engine handles failures gracefully."""
        # Use nonexistent GGUF path
        config = BundleConfig(
            gguf_path=tmp_path / "nonexistent.gguf",
            tools=sample_tools,
            output_dir=tmp_path / "bundle",
        )

        engine = BundleEngine(config)
        result = engine.package()

        assert result.success is False
        assert result.error is not None


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent bundle access."""

    def test_multiple_bundles_same_tools(
        self,
        mock_gguf_file: Path,
        sample_tools: list[ToolDefinition],
        tmp_path: Path,
    ):
        """Test creating multiple bundles from same tools."""
        bundles = []
        for i in range(3):
            config = BundleConfig(
                gguf_path=mock_gguf_file,
                tools=sample_tools,
                output_dir=tmp_path / f"bundle_{i}",
                model_name=f"agent-{i}",
            )
            engine = BundleEngine(config)
            result = engine.package()
            bundles.append(result)

        # All should succeed
        for i, result in enumerate(bundles):
            assert result.success is True
            assert (tmp_path / f"bundle_{i}" / "model.gguf").exists()

    def test_overwrite_existing_bundle(
        self,
        mock_gguf_file: Path,
        sample_tools: list[ToolDefinition],
        tmp_path: Path,
    ):
        """Test overwriting existing bundle directory."""
        output_dir = tmp_path / "bundle"

        # Create first bundle
        config1 = BundleConfig(
            gguf_path=mock_gguf_file,
            tools=sample_tools,
            output_dir=output_dir,
            model_name="first-agent",
        )
        engine1 = BundleEngine(config1)
        result1 = engine1.package()
        assert result1.success is True

        # Create second bundle in same location
        config2 = BundleConfig(
            gguf_path=mock_gguf_file,
            tools=sample_tools,
            output_dir=output_dir,
            model_name="second-agent",
        )
        engine2 = BundleEngine(config2)
        result2 = engine2.package()
        assert result2.success is True

        # Second config should have overwritten
        with open(output_dir / "config.yaml") as f:
            data = yaml.safe_load(f)
        assert data["model"]["name"] == "second-agent"
