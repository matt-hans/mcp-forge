"""Benchmark suite for comprehensive model evaluation.

Provides detailed per-tool and per-scenario metrics with optional
baseline comparison and latency tracking.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp_forge.state import BenchmarkResult, Scenario, ToolDefinition
from mcp_forge.validation.config import InferenceConfig, StubConfig, ValidationConfig
from mcp_forge.validation.runner import ValidationRunner, ValidationSample
from mcp_forge.validation.stubs import StubRegistry


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    model_path: Path
    model_name: str  # Human-readable name for reports

    # Sample counts per category
    samples_per_tool: int = 20
    samples_per_scenario: int = 20

    # Latency tracking
    measure_latency: bool = True
    warmup_samples: int = 3

    # Thresholds for pass/fail
    accuracy_threshold: float = 0.90
    no_tool_threshold: float = 0.85
    loop_threshold: float = 0.95

    # Inference settings
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Stub configuration (required for deterministic benchmarks)
    stub_config: StubConfig | None = None
    mcp_command: str | None = None

    # Baseline comparison
    baseline_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate that either stub_config or mcp_command is provided."""
        if self.stub_config is None and self.mcp_command is None:
            raise ValueError("Either stub_config or mcp_command must be provided")


@dataclass
class LatencyStats:
    """Latency statistics for a benchmark run."""

    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    @classmethod
    def from_samples(cls, latencies_ms: list[float]) -> LatencyStats:
        """Calculate stats from a list of latency samples.

        Args:
            latencies_ms: List of latency measurements in milliseconds

        Returns:
            LatencyStats with calculated metrics
        """
        if not latencies_ms:
            return cls()

        sorted_latencies = sorted(latencies_ms)
        n = len(sorted_latencies)

        return cls(
            min_ms=sorted_latencies[0],
            max_ms=sorted_latencies[-1],
            mean_ms=sum(sorted_latencies) / n,
            p50_ms=sorted_latencies[n // 2],
            p95_ms=sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1],
            p99_ms=sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize to dictionary.

        Returns:
            Dictionary with all latency metrics
        """
        return {
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }
