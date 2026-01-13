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


class BenchmarkRunner:
    """Executes comprehensive benchmark evaluation.

    Generates samples per-tool and per-scenario, tracks latency metrics,
    and produces BenchmarkResult with detailed breakdowns.
    """

    def __init__(self, config: BenchmarkConfig, tools: list[ToolDefinition]) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration
            tools: List of tool definitions for benchmarking
        """
        self.config = config
        self.tools = tools
        self._validation_runner: ValidationRunner | None = None
        self._latencies: list[float] = []

    def _get_validation_runner(self) -> ValidationRunner:
        """Get or create the underlying validation runner.

        Returns:
            ValidationRunner instance configured for this benchmark
        """
        if self._validation_runner is None:
            val_config = ValidationConfig(
                model_path=self.config.model_path,
                samples=0,  # We control sample count
                stub_config=self.config.stub_config,
                mcp_command=self.config.mcp_command,
                inference=self.config.inference,
            )
            self._validation_runner = ValidationRunner(val_config, self.tools)
        return self._validation_runner

    def _generate_benchmark_samples(self) -> dict[str, list[ValidationSample]]:
        """Generate samples organized by tool and scenario.

        Returns:
            Dictionary with keys like 'tool:get_weather', 'scenario:no_tool'
            and values as lists of ValidationSample objects
        """
        import random

        rng = random.Random(42)

        samples: dict[str, list[ValidationSample]] = {}

        # Generate per-tool samples
        for tool in self.tools:
            key = f"tool:{tool.name}"
            samples[key] = self._generate_tool_samples(tool, self.config.samples_per_tool, rng)

        # Generate per-scenario samples
        for scenario in Scenario:
            key = f"scenario:{scenario.value}"
            samples[key] = self._generate_scenario_samples(scenario, self.config.samples_per_scenario, rng)

        return samples

    def _generate_tool_samples(
        self,
        tool: ToolDefinition,
        count: int,
        rng: "random.Random",
    ) -> list[ValidationSample]:
        """Generate benchmark samples for a specific tool.

        Args:
            tool: Tool definition to generate samples for
            count: Number of samples to generate
            rng: Random number generator for determinism

        Returns:
            List of ValidationSample objects for this tool
        """
        import random

        samples: list[ValidationSample] = []

        # Tool-specific prompt templates
        TEMPLATES: dict[str, list[str]] = {
            "get_weather": [
                "What's the weather in {location}?",
                "Tell me the forecast for {location}.",
                "Is it raining in {location}?",
                "What's the temperature in {location} right now?",
                "How's the weather looking in {location}?",
            ],
            "list_files": [
                "List the files in {path}.",
                "What files are in {path}?",
                "Show me the contents of {path}.",
                "What's inside the {path} directory?",
            ],
        }

        LOCATIONS = ["Paris", "London", "Tokyo", "New York", "Sydney", "Berlin", "Mumbai"]
        PATHS = ["/home/user/documents", "/home/user/projects", "/tmp", "/var/log"]

        templates = TEMPLATES.get(tool.name, [f"Use the {tool.name} tool."])

        for _ in range(count):
            template = rng.choice(templates)

            if "{location}" in template:
                location = rng.choice(LOCATIONS)
                prompt = template.format(location=location)
                expected_args = {"location": location}
            elif "{path}" in template:
                path = rng.choice(PATHS)
                prompt = template.format(path=path)
                expected_args = {"path": path}
            else:
                prompt = template
                expected_args = {}

            samples.append(
                ValidationSample(
                    prompt=prompt,
                    expected_tool=tool.name,
                    expected_args=expected_args,
                )
            )

        return samples

    def _generate_scenario_samples(
        self,
        scenario: Scenario,
        count: int,
        rng: "random.Random",
    ) -> list[ValidationSample]:
        """Generate benchmark samples for a specific scenario type.

        Args:
            scenario: Scenario type to generate samples for
            count: Number of samples to generate
            rng: Random number generator for determinism

        Returns:
            List of ValidationSample objects for this scenario
        """
        import random

        samples: list[ValidationSample] = []

        if scenario == Scenario.NO_TOOL:
            # Questions that should NOT trigger tool use
            NO_TOOL_PROMPTS = [
                "What is the capital of France?",
                "How do I make pasta?",
                "Tell me a joke.",
                "What year did World War II end?",
                "Explain quantum computing.",
                "What's 15 * 7?",
                "Who wrote Romeo and Juliet?",
                "What is the meaning of life?",
            ]
            for prompt in rng.sample(NO_TOOL_PROMPTS, min(count, len(NO_TOOL_PROMPTS))):
                samples.append(
                    ValidationSample(
                        prompt=prompt,
                        expected_tool=None,
                        expected_args=None,
                    )
                )

        elif scenario == Scenario.ERROR:
            # Prompts that should trigger tool use but with potentially problematic args
            ERROR_PROMPTS = [
                ("What's the weather in ???", "get_weather"),
                ("List files in /nonexistent/path/12345", "list_files"),
            ]
            for prompt, tool in rng.sample(ERROR_PROMPTS, min(count, len(ERROR_PROMPTS))):
                samples.append(
                    ValidationSample(
                        prompt=prompt,
                        expected_tool=tool,
                        expected_args={},
                    )
                )

        elif scenario == Scenario.AMBIGUOUS:
            # Prompts where multiple tools could apply
            # For now, just use standard prompts distributed across tools
            if self.tools:
                for tool in self.tools:
                    tool_samples = self._generate_tool_samples(
                        tool, count // len(self.tools), rng
                    )
                    samples.extend(tool_samples)

        else:
            # STANDARD, EDGE: use standard tool prompts
            if self.tools:
                for tool in self.tools:
                    tool_samples = self._generate_tool_samples(
                        tool, count // len(self.tools), rng
                    )
                    samples.extend(tool_samples)

        return samples[:count]
