"""Configuration management for MCP-Forge.

Supports loading configuration from YAML files with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ForgeConfig:
    """MCP-Forge configuration with QC thresholds.

    Configuration is loaded with the following precedence:
    1. CLI flags (highest priority)
    2. Project config file (.mcp-forge/config.yaml)
    3. User config file (~/.config/mcp-forge/config.yaml)
    4. Default values (lowest priority)
    """

    # QC thresholds
    qc_schema_pass_threshold: float = 0.98
    qc_min_samples_per_tool: int = 10
    qc_dedup_enabled: bool = True
    qc_auto_repair: bool = True
    qc_require_scenario_coverage: bool = True

    # Scenario targets
    qc_scenario_targets: dict[str, float] = field(default_factory=lambda: {
        "standard": 0.60,
        "no_tool": 0.15,
        "error": 0.10,
        "ambiguous": 0.10,
        "edge": 0.05
    })

    # Synthesis settings
    synthesis_total_samples: int = 500
    synthesis_seed_samples: int = 100

    # Training settings
    training_profile: str = "balanced"
    training_quantization: str = "q8_0"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "qc": {
                "schema_pass_threshold": self.qc_schema_pass_threshold,
                "min_samples_per_tool": self.qc_min_samples_per_tool,
                "dedup_enabled": self.qc_dedup_enabled,
                "auto_repair": self.qc_auto_repair,
                "require_scenario_coverage": self.qc_require_scenario_coverage,
                "scenario_targets": self.qc_scenario_targets,
            },
            "synthesis": {
                "total_samples": self.synthesis_total_samples,
                "seed_samples": self.synthesis_seed_samples,
            },
            "training": {
                "profile": self.training_profile,
                "quantization": self.training_quantization,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForgeConfig:
        """Create config from dictionary (YAML structure)."""
        qc = data.get("qc", {})
        synthesis = data.get("synthesis", {})
        training = data.get("training", {})

        return cls(
            qc_schema_pass_threshold=qc.get("schema_pass_threshold", 0.98),
            qc_min_samples_per_tool=qc.get("min_samples_per_tool", 10),
            qc_dedup_enabled=qc.get("dedup_enabled", True),
            qc_auto_repair=qc.get("auto_repair", True),
            qc_require_scenario_coverage=qc.get("require_scenario_coverage", True),
            qc_scenario_targets=qc.get("scenario_targets", {
                "standard": 0.60,
                "no_tool": 0.15,
                "error": 0.10,
                "ambiguous": 0.10,
                "edge": 0.05
            }),
            synthesis_total_samples=synthesis.get("total_samples", 500),
            synthesis_seed_samples=synthesis.get("seed_samples", 100),
            training_profile=training.get("profile", "balanced"),
            training_quantization=training.get("quantization", "q8_0"),
        )


def _find_config_file() -> Path | None:
    """Find configuration file in standard locations.

    Checks in order:
    1. .mcp-forge/config.yaml (project-level)
    2. ~/.config/mcp-forge/config.yaml (user-level)

    Returns first found path or None.
    """
    # Project-level config
    project_config = Path.cwd() / ".mcp-forge" / "config.yaml"
    if project_config.exists():
        return project_config

    # User-level config
    user_config = Path.home() / ".config" / "mcp-forge" / "config.yaml"
    if user_config.exists():
        return user_config

    return None


def load_config(path: Path | None = None) -> ForgeConfig:
    """Load configuration from file or use defaults.

    Args:
        path: Optional explicit path to config file.
              If None, searches standard locations.

    Returns:
        ForgeConfig with values from file or defaults.
    """
    config_path = path or _find_config_file()

    if config_path is None:
        return ForgeConfig()

    if not config_path.exists():
        return ForgeConfig()

    try:
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return ForgeConfig.from_dict(data)
    except ImportError:
        # YAML not available, return defaults
        return ForgeConfig()
    except Exception:
        # Any error loading config, return defaults
        return ForgeConfig()


def merge_config_with_cli(
    config: ForgeConfig,
    *,
    threshold: float | None = None,
    min_samples: int | None = None,
    no_dedup: bool = False,
    no_auto_repair: bool = False,
    strict: bool = False,
) -> ForgeConfig:
    """Merge CLI options into config (CLI takes precedence).

    Args:
        config: Base configuration
        threshold: Schema pass threshold override
        min_samples: Min samples per tool override
        no_dedup: Disable deduplication flag
        no_auto_repair: Disable auto-repair flag
        strict: Enable strict mode (not used directly, for future)

    Returns:
        New ForgeConfig with CLI overrides applied.
    """
    return ForgeConfig(
        qc_schema_pass_threshold=threshold if threshold is not None else config.qc_schema_pass_threshold,
        qc_min_samples_per_tool=min_samples if min_samples is not None else config.qc_min_samples_per_tool,
        qc_dedup_enabled=not no_dedup if no_dedup else config.qc_dedup_enabled,
        qc_auto_repair=not no_auto_repair if no_auto_repair else config.qc_auto_repair,
        qc_require_scenario_coverage=config.qc_require_scenario_coverage,
        qc_scenario_targets=config.qc_scenario_targets,
        synthesis_total_samples=config.synthesis_total_samples,
        synthesis_seed_samples=config.synthesis_seed_samples,
        training_profile=config.training_profile,
        training_quantization=config.training_quantization,
    )
