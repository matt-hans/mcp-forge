"""Integration tests for QC gate functionality.

Tests the full QC pipeline including:
- QC validation with configurable thresholds
- Pipeline blocking on QC failure
- Repair mode with statistics
- Report generation in multiple formats
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcp_forge.config import ForgeConfig, load_config, merge_config_with_cli
from mcp_forge.data.qc import (
    DataQualityController,
    QCConfig,
    QCFailedError,
)
from mcp_forge.state import QCReport, ToolDefinition


@pytest.fixture
def sample_tools() -> list[ToolDefinition]:
    """Create sample tool definitions for testing."""
    return [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        ),
        ToolDefinition(
            name="search_files",
            description="Search for files",
            input_schema={
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        ),
    ]


@pytest.fixture
def valid_samples() -> list[dict[str, Any]]:
    """Create valid training samples."""
    return [
        {
            "id": f"valid_{i:03d}",
            "source": "seed",
            "scenario": "standard",
            "tool_name": "get_weather",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"What's the weather in City {i}?"},
                {
                    "role": "assistant",
                    "content": f'<tool_call>{{"name": "get_weather", "arguments": {{"location": "City {i}"}}}}</tool_call>',
                },
            ],
        }
        for i in range(20)
    ]


@pytest.fixture
def mixed_samples(valid_samples: list[dict]) -> list[dict[str, Any]]:
    """Create samples with mix of valid and invalid."""
    invalid_samples = [
        {
            "id": "invalid_001",
            "source": "seed",
            "scenario": "standard",
            "tool_name": "get_weather",
            "messages": [
                {"role": "user", "content": "Weather please"},
                # Missing tool call
                {"role": "assistant", "content": "Let me check the weather."},
            ],
        },
        {
            "id": "invalid_002",
            "source": "seed",
            "scenario": "standard",
            "tool_name": "unknown_tool",  # Invalid tool
            "messages": [
                {"role": "user", "content": "Do something"},
                {"role": "assistant", "content": "Here you go."},
            ],
        },
    ]
    return valid_samples + invalid_samples


@pytest.fixture
def samples_with_whitespace_issues() -> list[dict[str, Any]]:
    """Create samples with whitespace issues that can be repaired."""
    return [
        {
            "id": "whitespace_001",
            "source": "seed",
            "scenario": "standard",
            "tool_name": "get_weather",
            "messages": [
                {"role": "system", "content": "You are helpful.  \n  "},  # Trailing whitespace
                {"role": "user", "content": "Weather?\r\n"},  # Windows line endings
                {
                    "role": "assistant",
                    "content": '<tool_call>{"name": "get_weather", "arguments": {"location": "NYC"}}</tool_call>',
                },
            ],
        }
        for _ in range(5)
    ]


class TestQCGateIntegration:
    """Integration tests for QC gate pipeline blocking."""

    def test_qc_passes_valid_data(
        self,
        sample_tools: list[ToolDefinition],
        valid_samples: list[dict],
        tmp_path: Path,
    ) -> None:
        """Test QC passes when data is valid."""
        # Write samples to file
        data_path = tmp_path / "data.jsonl"
        with open(data_path, "w") as f:
            for sample in valid_samples:
                f.write(json.dumps(sample) + "\n")

        # Run QC
        config = QCConfig(
            schema_pass_threshold=0.90,
            min_samples_per_tool=1,
        )
        qc = DataQualityController(sample_tools, config)
        report, validated = qc.validate_dataset(data_path)

        # Should pass
        assert report.passes_threshold(0.90, 1)
        assert len(validated) > 0
        assert report.schema_pass_rate >= 0.90

    def test_qc_blocks_invalid_data(
        self,
        sample_tools: list[ToolDefinition],
        tmp_path: Path,
    ) -> None:
        """Test QC blocks pipeline when tool coverage threshold not met."""
        # Create minimal samples - only 5 for get_weather
        samples = [
            {
                "id": f"test_{i:03d}",
                "source": "seed",
                "scenario": "standard",
                "tool_name": "get_weather",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Weather in City {i}?"},
                    {
                        "role": "assistant",
                        "content": f'<tool_call>{{"name": "get_weather", "arguments": {{"location": "City {i}"}}}}</tool_call>',
                    },
                ],
            }
            for i in range(5)  # Only 5 samples
        ]

        # Write samples to file
        data_path = tmp_path / "data.jsonl"
        with open(data_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        # Run QC with min_samples_per_tool=10 (but we only have 5)
        config = QCConfig(
            schema_pass_threshold=0.50,  # Low schema threshold
            min_samples_per_tool=10,  # Requires 10, we have 5
        )
        qc = DataQualityController(sample_tools, config)
        report, validated = qc.validate_dataset(data_path)

        # Should fail tool coverage threshold (get_weather has 5, needs 10)
        assert not report.passes_threshold(0.50, 10)
        assert report.tool_coverage.get("get_weather", 0) == 5

        # Raise QCFailedError
        with pytest.raises(QCFailedError) as exc_info:
            if not report.passes_threshold(0.50, 10):
                raise QCFailedError(report, threshold=0.50, min_samples=10)

        error = exc_info.value
        assert len(error.failures) > 0
        assert any("get_weather" in f for f in error.failures)
        assert len(error.remediation) > 0

    def test_qc_error_format(
        self,
        sample_tools: list[ToolDefinition],
        tmp_path: Path,
    ) -> None:
        """Test QCFailedError provides useful formatting."""
        # Create minimal data that will fail
        data_path = tmp_path / "data.jsonl"
        with open(data_path, "w") as f:
            f.write(json.dumps({
                "id": "test",
                "source": "seed",
                "scenario": "standard",
                "messages": [{"role": "user", "content": "Hi"}],
            }) + "\n")

        config = QCConfig(schema_pass_threshold=0.98, min_samples_per_tool=10)
        qc = DataQualityController(sample_tools, config)
        report, _ = qc.validate_dataset(data_path)

        # Create error
        error = QCFailedError(report, threshold=0.98, min_samples=10)
        formatted = error.format_error()

        # Verify format includes all sections
        assert "QC Validation Failed" in formatted
        assert "Failures:" in formatted
        assert "Remediation suggestions:" in formatted


class TestQCConfigIntegration:
    """Integration tests for config file handling."""

    def test_config_yaml_loading(self, tmp_path: Path) -> None:
        """Test loading config from YAML file."""
        pytest.importorskip("yaml")

        config_dir = tmp_path / ".mcp-forge"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("""
qc:
  schema_pass_threshold: 0.85
  min_samples_per_tool: 5
  dedup_enabled: false
synthesis:
  total_samples: 1000
""")

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            config = load_config()

            assert config.qc_schema_pass_threshold == 0.85
            assert config.qc_min_samples_per_tool == 5
            assert config.qc_dedup_enabled is False
            assert config.synthesis_total_samples == 1000
        finally:
            os.chdir(original_cwd)

    def test_cli_overrides_config(self) -> None:
        """Test CLI options override config file values."""
        base = ForgeConfig(
            qc_schema_pass_threshold=0.98,
            qc_min_samples_per_tool=10,
        )

        merged = merge_config_with_cli(
            base,
            threshold=0.80,
            min_samples=5,
        )

        assert merged.qc_schema_pass_threshold == 0.80
        assert merged.qc_min_samples_per_tool == 5


class TestRepairIntegration:
    """Integration tests for repair functionality."""

    def test_repair_pipeline(
        self,
        sample_tools: list[ToolDefinition],
        samples_with_whitespace_issues: list[dict],
        tmp_path: Path,
    ) -> None:
        """Test full repair pipeline with statistics tracking."""
        # Write samples to file
        data_path = tmp_path / "data.jsonl"
        with open(data_path, "w") as f:
            for sample in samples_with_whitespace_issues:
                f.write(json.dumps(sample) + "\n")

        # Run QC with auto_repair enabled
        config = QCConfig(
            auto_repair=True,
            min_samples_per_tool=1,
        )
        qc = DataQualityController(sample_tools, config)

        # Repair samples
        repaired_count = 0
        with open(data_path) as f:
            for line in f:
                sample = json.loads(line)
                was_repaired, repaired = qc.repair_sample(sample, sample["id"])
                if was_repaired:
                    repaired_count += 1

        # Should have repaired some samples
        assert repaired_count > 0
        assert qc.repair_stats.repairs_successful > 0

    def test_dry_run_no_modifications(
        self,
        sample_tools: list[ToolDefinition],
        valid_samples: list[dict],
        tmp_path: Path,
    ) -> None:
        """Test dry run doesn't write files."""
        # Write samples to file
        data_path = tmp_path / "data.jsonl"
        with open(data_path, "w") as f:
            for sample in valid_samples:
                f.write(json.dumps(sample) + "\n")

        output_path = tmp_path / "output.jsonl"

        config = QCConfig(min_samples_per_tool=1)
        qc = DataQualityController(sample_tools, config)
        report, validated = qc.validate_dataset(data_path, output_path, dry_run=True)

        # Output file should not exist
        assert not output_path.exists()

        # But report should still be generated
        assert report.total_samples == len(valid_samples)


class TestReportFormats:
    """Integration tests for report format generation."""

    def test_report_roundtrip(
        self,
        sample_tools: list[ToolDefinition],
        valid_samples: list[dict],
        tmp_path: Path,
    ) -> None:
        """Test report can be serialized and deserialized."""
        # Write samples to file
        data_path = tmp_path / "data.jsonl"
        with open(data_path, "w") as f:
            for sample in valid_samples:
                f.write(json.dumps(sample) + "\n")

        config = QCConfig(min_samples_per_tool=1)
        qc = DataQualityController(sample_tools, config)
        report, _ = qc.validate_dataset(data_path)

        # Serialize and deserialize
        report_dict = report.to_dict()
        restored = QCReport.from_dict(report_dict)

        assert restored.total_samples == report.total_samples
        assert restored.schema_pass_rate == report.schema_pass_rate
        assert restored.tool_coverage == report.tool_coverage

    def test_qcreport_passes_threshold(self) -> None:
        """Test QCReport threshold checking."""
        report = QCReport(
            total_samples=100,
            valid_samples=95,
            dropped_samples=5,
            schema_pass_rate=0.95,
            dedup_rate=0.02,
            tool_coverage={"tool1": 50, "tool2": 45},
            scenario_coverage={"standard": 80, "no_tool": 15},
            issues=[],
        )

        # Should pass with low thresholds
        assert report.passes_threshold(0.90, 10)

        # Should fail with high thresholds
        assert not report.passes_threshold(0.98, 10)
        assert not report.passes_threshold(0.90, 50)
