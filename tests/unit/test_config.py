"""Tests for mcp_forge.config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_forge.config import (
    ForgeConfig,
    load_config,
    merge_config_with_cli,
)


class TestForgeConfig:
    """Tests for ForgeConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ForgeConfig()
        assert config.qc_schema_pass_threshold == 0.98
        assert config.qc_min_samples_per_tool == 10
        assert config.qc_dedup_enabled is True
        assert config.qc_auto_repair is True
        assert config.qc_require_scenario_coverage is True

    def test_default_scenario_targets(self) -> None:
        """Test default scenario target weights."""
        config = ForgeConfig()
        assert config.qc_scenario_targets["standard"] == 0.60
        assert config.qc_scenario_targets["no_tool"] == 0.15
        assert sum(config.qc_scenario_targets.values()) == pytest.approx(1.0)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ForgeConfig(
            qc_schema_pass_threshold=0.95,
            qc_min_samples_per_tool=5,
            qc_dedup_enabled=False,
        )
        assert config.qc_schema_pass_threshold == 0.95
        assert config.qc_min_samples_per_tool == 5
        assert config.qc_dedup_enabled is False

    def test_to_dict(self) -> None:
        """Test config serialization."""
        config = ForgeConfig()
        result = config.to_dict()

        assert "qc" in result
        assert "synthesis" in result
        assert "training" in result
        assert result["qc"]["schema_pass_threshold"] == 0.98
        assert result["synthesis"]["total_samples"] == 500
        assert result["training"]["profile"] == "balanced"

    def test_from_dict_full(self) -> None:
        """Test creating config from full dictionary."""
        data = {
            "qc": {
                "schema_pass_threshold": 0.90,
                "min_samples_per_tool": 15,
                "dedup_enabled": False,
                "auto_repair": False,
                "require_scenario_coverage": False,
                "scenario_targets": {
                    "standard": 0.70,
                    "no_tool": 0.10,
                    "error": 0.10,
                    "ambiguous": 0.05,
                    "edge": 0.05,
                },
            },
            "synthesis": {
                "total_samples": 1000,
                "seed_samples": 200,
            },
            "training": {
                "profile": "max_quality",
                "quantization": "q4_k_m",
            },
        }

        config = ForgeConfig.from_dict(data)

        assert config.qc_schema_pass_threshold == 0.90
        assert config.qc_min_samples_per_tool == 15
        assert config.qc_dedup_enabled is False
        assert config.qc_auto_repair is False
        assert config.qc_scenario_targets["standard"] == 0.70
        assert config.synthesis_total_samples == 1000
        assert config.training_profile == "max_quality"

    def test_from_dict_partial(self) -> None:
        """Test creating config from partial dictionary uses defaults."""
        data = {
            "qc": {
                "schema_pass_threshold": 0.95,
            },
        }

        config = ForgeConfig.from_dict(data)

        assert config.qc_schema_pass_threshold == 0.95
        assert config.qc_min_samples_per_tool == 10  # Default
        assert config.qc_dedup_enabled is True  # Default
        assert config.synthesis_total_samples == 500  # Default

    def test_from_dict_empty(self) -> None:
        """Test creating config from empty dictionary uses all defaults."""
        config = ForgeConfig.from_dict({})

        assert config.qc_schema_pass_threshold == 0.98
        assert config.qc_min_samples_per_tool == 10
        assert config.synthesis_total_samples == 500

    def test_roundtrip(self) -> None:
        """Test to_dict -> from_dict roundtrip preserves values."""
        original = ForgeConfig(
            qc_schema_pass_threshold=0.92,
            qc_min_samples_per_tool=8,
            synthesis_total_samples=750,
        )

        result = ForgeConfig.from_dict(original.to_dict())

        assert result.qc_schema_pass_threshold == original.qc_schema_pass_threshold
        assert result.qc_min_samples_per_tool == original.qc_min_samples_per_tool
        assert result.synthesis_total_samples == original.synthesis_total_samples


class TestLoadConfig:
    """Tests for load_config function."""

    def test_returns_defaults_when_no_file(self, tmp_path: Path) -> None:
        """Test returns defaults when no config file exists."""
        # Point to non-existent path
        config = load_config(tmp_path / "nonexistent.yaml")

        assert config.qc_schema_pass_threshold == 0.98
        assert config.qc_min_samples_per_tool == 10

    def test_loads_from_yaml_file(self, tmp_path: Path) -> None:
        """Test loading config from YAML file."""
        pytest.importorskip("yaml")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
qc:
  schema_pass_threshold: 0.90
  min_samples_per_tool: 5
synthesis:
  total_samples: 1000
""")

        config = load_config(config_file)

        assert config.qc_schema_pass_threshold == 0.90
        assert config.qc_min_samples_per_tool == 5
        assert config.synthesis_total_samples == 1000

    def test_handles_invalid_yaml(self, tmp_path: Path) -> None:
        """Test returns defaults for invalid YAML."""
        pytest.importorskip("yaml")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        config = load_config(config_file)

        # Should return defaults, not crash
        assert config.qc_schema_pass_threshold == 0.98

    def test_handles_empty_yaml(self, tmp_path: Path) -> None:
        """Test handles empty YAML file."""
        pytest.importorskip("yaml")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        config = load_config(config_file)

        assert config.qc_schema_pass_threshold == 0.98

    def test_discovery_project_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test config discovery finds project-level config."""
        pytest.importorskip("yaml")

        # Set up project config
        monkeypatch.chdir(tmp_path)
        config_dir = tmp_path / ".mcp-forge"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("""
qc:
  schema_pass_threshold: 0.85
""")

        config = load_config()  # No explicit path

        assert config.qc_schema_pass_threshold == 0.85


class TestMergeConfigWithCli:
    """Tests for merge_config_with_cli function."""

    def test_cli_overrides_config(self) -> None:
        """Test CLI options override config values."""
        base = ForgeConfig(
            qc_schema_pass_threshold=0.98,
            qc_min_samples_per_tool=10,
        )

        merged = merge_config_with_cli(
            base,
            threshold=0.90,
            min_samples=5,
        )

        assert merged.qc_schema_pass_threshold == 0.90
        assert merged.qc_min_samples_per_tool == 5

    def test_none_preserves_config(self) -> None:
        """Test None CLI values preserve config."""
        base = ForgeConfig(
            qc_schema_pass_threshold=0.95,
            qc_min_samples_per_tool=15,
        )

        merged = merge_config_with_cli(
            base,
            threshold=None,
            min_samples=None,
        )

        assert merged.qc_schema_pass_threshold == 0.95
        assert merged.qc_min_samples_per_tool == 15

    def test_no_dedup_flag(self) -> None:
        """Test --no-dedup flag disables deduplication."""
        base = ForgeConfig(qc_dedup_enabled=True)

        merged = merge_config_with_cli(base, no_dedup=True)

        assert merged.qc_dedup_enabled is False

    def test_no_auto_repair_flag(self) -> None:
        """Test --no-auto-repair flag disables auto-repair."""
        base = ForgeConfig(qc_auto_repair=True)

        merged = merge_config_with_cli(base, no_auto_repair=True)

        assert merged.qc_auto_repair is False

    def test_partial_override(self) -> None:
        """Test partial CLI overrides."""
        base = ForgeConfig(
            qc_schema_pass_threshold=0.98,
            qc_min_samples_per_tool=10,
            qc_dedup_enabled=True,
        )

        merged = merge_config_with_cli(
            base,
            threshold=0.85,
            min_samples=None,  # Keep default
            no_dedup=False,  # Keep enabled
        )

        assert merged.qc_schema_pass_threshold == 0.85
        assert merged.qc_min_samples_per_tool == 10  # Preserved
        assert merged.qc_dedup_enabled is True  # Preserved

    def test_preserves_non_qc_settings(self) -> None:
        """Test merge preserves non-QC settings from base config."""
        base = ForgeConfig(
            synthesis_total_samples=1000,
            training_profile="max_quality",
        )

        merged = merge_config_with_cli(base, threshold=0.90)

        assert merged.synthesis_total_samples == 1000
        assert merged.training_profile == "max_quality"
