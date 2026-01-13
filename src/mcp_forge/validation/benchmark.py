"""Benchmark suite for comprehensive model evaluation.

Provides detailed per-tool and per-scenario metrics with optional
baseline comparison and latency tracking.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp_forge.state import BenchmarkResult, Scenario, ToolDefinition
from mcp_forge.validation.config import InferenceConfig, StubConfig, ValidationConfig
from mcp_forge.validation.runner import ValidationRunner, ValidationSample


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
        rng: random.Random,
    ) -> list[ValidationSample]:
        """Generate benchmark samples for a specific tool.

        Args:
            tool: Tool definition to generate samples for
            count: Number of samples to generate
            rng: Random number generator for determinism

        Returns:
            List of ValidationSample objects for this tool
        """
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
        rng: random.Random,
    ) -> list[ValidationSample]:
        """Generate benchmark samples for a specific scenario type.

        Args:
            scenario: Scenario type to generate samples for
            count: Number of samples to generate
            rng: Random number generator for determinism

        Returns:
            List of ValidationSample objects for this scenario
        """
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

    def run(
        self,
        progress_callback: Callable[[str, int, int, float], None] | None = None,
    ) -> BenchmarkResult:
        """Run full benchmark and return aggregated results.

        Args:
            progress_callback: Optional callback(category, current, total, progress_pct)

        Returns:
            BenchmarkResult with per-tool and per-scenario metrics
        """
        runner = self._get_validation_runner()

        # Load model once
        if runner.model is None:
            runner.load_model()

        # Warmup (discard latencies)
        if self.config.warmup_samples > 0 and self.config.measure_latency:
            warmup_samples = self._generate_tool_samples(
                self.tools[0], self.config.warmup_samples, random.Random(0)
            )
            for sample in warmup_samples:
                runner.validate_single(sample)

        # Generate all benchmark samples
        all_samples = self._generate_benchmark_samples()

        # Track results
        per_tool_results: dict[str, dict[str, float]] = {}
        per_scenario_results: dict[str, dict[str, float]] = {}
        per_tool_latencies: dict[str, list[float]] = {}

        total_samples = sum(len(samples) for samples in all_samples.values())
        completed = 0

        for category, samples in all_samples.items():
            category_results: list[dict[str, Any]] = []
            category_latencies: list[float] = []

            for sample in samples:
                # Measure latency
                start_time = time.perf_counter()
                result = runner.validate_single(sample)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                category_results.append(result)
                if self.config.measure_latency:
                    category_latencies.append(elapsed_ms)

                completed += 1
                if progress_callback:
                    progress_callback(category, completed, total_samples, completed / total_samples)

            # Aggregate category metrics
            total = len(category_results)
            if total == 0:
                continue

            parsed = sum(1 for r in category_results if r["parsed"])
            schema_valid = sum(1 for r in category_results if r["schema_valid"])
            tool_correct = sum(1 for r in category_results if r["tool_correct"])
            loop_complete = sum(1 for r in category_results if r["loop_complete"])

            latency_stats = LatencyStats.from_samples(category_latencies)

            metrics = {
                "samples": total,
                "accuracy": tool_correct / total,
                "schema": schema_valid / total,
                "parse_rate": parsed / total,
                "loop_rate": loop_complete / total,
                "latency_mean_ms": latency_stats.mean_ms,
                "latency_p95_ms": latency_stats.p95_ms,
            }

            if category.startswith("tool:"):
                tool_name = category.split(":", 1)[1]
                per_tool_results[tool_name] = metrics
                per_tool_latencies[tool_name] = category_latencies
            elif category.startswith("scenario:"):
                scenario_name = category.split(":", 1)[1]
                per_scenario_results[scenario_name] = {"pass_rate": tool_correct / total}

        # Calculate overall score (weighted average of key metrics)
        all_tool_accuracies = [m["accuracy"] for m in per_tool_results.values()]
        overall_accuracy = sum(all_tool_accuracies) / len(all_tool_accuracies) if all_tool_accuracies else 0.0

        no_tool_rate = per_scenario_results.get("no_tool", {}).get("pass_rate", 0.0)

        # Overall score: 70% accuracy, 30% no-tool correctness
        overall_score = (overall_accuracy * 0.7) + (no_tool_rate * 0.3)

        # Store latencies for potential baseline comparison
        self._latencies = [lat for lats in per_tool_latencies.values() for lat in lats]

        return BenchmarkResult(
            model_name=self.config.model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_score=overall_score,
            per_tool_results=per_tool_results,
            per_scenario_results=per_scenario_results,
            baseline_comparison=None,  # Set separately if needed
        )

    def compare_to_baseline(
        self,
        current: BenchmarkResult,
        baseline: BenchmarkResult,
    ) -> dict[str, Any]:
        """Compare current results to a baseline.

        Args:
            current: Current benchmark results
            baseline: Baseline benchmark results

        Returns:
            Comparison dictionary with deltas
        """
        comparison: dict[str, Any] = {
            "baseline_model": baseline.model_name,
            "baseline_timestamp": baseline.timestamp,
            "overall_delta": current.overall_score - baseline.overall_score,
            "per_tool_deltas": {},
        }

        for tool, current_metrics in current.per_tool_results.items():
            baseline_metrics = baseline.per_tool_results.get(tool, {})
            if baseline_metrics:
                comparison["per_tool_deltas"][tool] = {
                    "accuracy_delta": current_metrics.get("accuracy", 0) - baseline_metrics.get("accuracy", 0),
                    "latency_delta_ms": current_metrics.get("latency_mean_ms", 0) - baseline_metrics.get("latency_mean_ms", 0),
                }

        return comparison
