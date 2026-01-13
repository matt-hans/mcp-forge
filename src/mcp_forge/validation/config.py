"""Configuration dataclasses for validation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class InferenceConfig:
    """Configuration for model inference during validation."""

    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    do_sample: bool = True


@dataclass
class StubConfig:
    """Configuration for MCP stub behavior."""

    stub_type: str  # "weather", "filesystem"
    deterministic: bool = True
    seed: int = 42
    response_delay: float = 0.0  # Simulated latency


@dataclass
class ValidationConfig:
    """Configuration for a validation run."""

    model_path: Path  # LoRA adapter or GGUF path
    samples: int = 20  # Number of validation samples
    timeout: float = 30.0  # Per-sample timeout
    retry_count: int = 3  # Retries for real MCP connections
    retry_delay: float = 1.0  # Delay between retries

    # Thresholds (from ValidationResult.meets_release_criteria)
    parse_threshold: float = 0.98
    schema_threshold: float = 0.95
    accuracy_threshold: float = 0.90
    loop_threshold: float = 0.95

    # Inference settings
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Optional stub configuration (mutually exclusive with mcp_command)
    stub_config: StubConfig | None = None
    mcp_command: str | None = None

    def __post_init__(self) -> None:
        """Validate that either stub_config or mcp_command is provided."""
        if self.stub_config is None and self.mcp_command is None:
            raise ValueError("Either stub_config or mcp_command must be provided")
