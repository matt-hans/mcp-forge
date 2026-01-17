# Phase 7: Benchmark Suite - Execution Plan

**Phase**: 7 of 9
**Milestone**: v1.0 - Full Pipeline Implementation
**Created**: 2026-01-13
**Estimated Scope**: Medium (7 tasks)

---

## Objective

Implement the BENCHMARKING stage with comprehensive evaluation metrics. Create a `validation/benchmark.py` module that:
1. Measures tool selection accuracy across all tool types
2. Tracks no-tool correctness (correctly abstaining when no tool is needed)
3. Measures end-to-end loop success rate
4. Records response latency metrics
5. Supports baseline model comparison
6. Generates reports in JSON and Markdown formats

---

## Execution Context

**Architecture**: Follow patterns in `validation/runner.py`, `data/qc.py`
**State Integration**: `BenchmarkResult` already defined in `state.py:140-159`
**CLI Stub**: `cli.py:1018-1025` - benchmark command exists but not implemented
**Pipeline**: `cli.py:339-346` - Stage 6 transition point (currently TODO)
**Testing**: 315 existing tests, maintain 85%+ coverage
**Conventions**: `CONVENTIONS.md` - Dataclasses, type hints, docstrings

---

## Context

### Existing Infrastructure

| Component | Path | Integration Point |
|-----------|------|-------------------|
| BenchmarkResult | `state.py:140-159` | Metrics dataclass with per-tool/scenario results |
| StateManager.save_benchmark_result | `state.py:456-470` | JSON + Markdown report generation |
| ValidationRunner | `validation/runner.py` | Reusable inference and metric aggregation |
| ValidationSample | `validation/runner.py:16-22` | Test case structure |
| StubRegistry | `validation/stubs.py` | Deterministic MCP stubs |
| CLI Stub | `cli.py:1018-1025` | benchmark command skeleton |
| Pipeline Stub | `cli.py:339-346` | Stage 6 BENCHMARKING transition |

### BenchmarkResult Fields (state.py:140-159)

```python
@dataclass
class BenchmarkResult:
    model_name: str
    timestamp: str
    overall_score: float
    per_tool_results: dict[str, dict[str, float]] = field(default_factory=dict)
    per_scenario_results: dict[str, dict[str, float]] = field(default_factory=dict)
    baseline_comparison: dict[str, Any] | None = None
```

### Quality Thresholds (from CLAUDE.md)

| Metric | Target |
|--------|--------|
| Tool selection accuracy | ≥90% |
| No-tool correctness | ≥85% |
| Loop completion rate | ≥95% |

### Benchmark vs Validation

| Aspect | Validation (Phase 6) | Benchmark (Phase 7) |
|--------|---------------------|---------------------|
| Purpose | Pass/fail gate | Detailed metrics |
| Samples | 20 (quick check) | 100+ (comprehensive) |
| Output | ValidationResult | BenchmarkResult with breakdown |
| Latency | Not measured | Per-sample timing |
| Comparison | None | Optional baseline delta |

---

## Tasks

### Task 1: Create benchmark module structure

**Action**: Create `src/mcp_forge/validation/benchmark.py` with config and runner classes.

**File**: `src/mcp_forge/validation/benchmark.py`

**Implementation**:
```python
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

from mcp_forge.state import BenchmarkResult, ToolDefinition, Scenario
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
        """Calculate stats from a list of latency samples."""
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
        return {
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
        }
```

**Verification**: `python -c "from mcp_forge.validation.benchmark import BenchmarkConfig"`

---

### Task 2: Implement BenchmarkRunner core

**Action**: Add BenchmarkRunner class with sample generation and metric aggregation.

**File**: `src/mcp_forge/validation/benchmark.py` (append to existing)

**Implementation**:
```python
class BenchmarkRunner:
    """Executes comprehensive benchmark evaluation."""

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
        """Get or create the underlying validation runner."""
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
        rng: random.Random,
    ) -> list[ValidationSample]:
        """Generate benchmark samples for a specific tool."""
        samples: list[ValidationSample] = []

        # Tool-specific prompt templates
        TEMPLATES = {
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
                prompt = template.format(location=rng.choice(LOCATIONS))
                expected_args = {"location": prompt.split()[-1].rstrip("?")}
            elif "{path}" in template:
                path = rng.choice(PATHS)
                prompt = template.format(path=path)
                expected_args = {"path": path}
            else:
                prompt = template
                expected_args = {}

            samples.append(ValidationSample(
                prompt=prompt,
                expected_tool=tool.name,
                expected_args=expected_args,
            ))

        return samples

    def _generate_scenario_samples(
        self,
        scenario: Scenario,
        count: int,
        rng: random.Random,
    ) -> list[ValidationSample]:
        """Generate benchmark samples for a specific scenario type."""
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
                samples.append(ValidationSample(
                    prompt=prompt,
                    expected_tool=None,
                    expected_args=None,
                ))

        elif scenario == Scenario.ERROR:
            # Prompts that should trigger tool use but with invalid arguments
            ERROR_PROMPTS = [
                ("What's the weather in ???", "get_weather"),
                ("List files in /nonexistent/path/12345", "list_files"),
            ]
            for prompt, tool in rng.sample(ERROR_PROMPTS, min(count, len(ERROR_PROMPTS))):
                samples.append(ValidationSample(
                    prompt=prompt,
                    expected_tool=tool,
                    expected_args={},
                ))

        elif scenario == Scenario.AMBIGUOUS:
            # Prompts where multiple tools could apply
            # For now, just use standard prompts
            for tool in self.tools:
                samples.extend(self._generate_tool_samples(tool, count // len(self.tools), rng))

        else:
            # STANDARD, EDGE: use standard tool prompts
            for tool in self.tools:
                samples.extend(self._generate_tool_samples(tool, count // len(self.tools), rng))

        return samples[:count]
```

**Verification**: Unit test for sample generation

---

### Task 3: Implement benchmark execution and aggregation

**Action**: Add run method with latency tracking and metric aggregation.

**File**: `src/mcp_forge/validation/benchmark.py` (append to BenchmarkRunner class)

**Implementation**:
```python
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
        comparison = {
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
```

**Verification**: Unit test with mocked model

---

### Task 4: Update module exports

**Action**: Add benchmark exports to `validation/__init__.py`.

**File**: `src/mcp_forge/validation/__init__.py`

**Replace with**:
```python
"""Looped validation module for trained model verification.

Provides validation against real and stubbed MCP servers with
deterministic testing and metric aggregation.
"""

from mcp_forge.validation.benchmark import (
    BenchmarkConfig,
    BenchmarkRunner,
    LatencyStats,
)
from mcp_forge.validation.config import (
    InferenceConfig,
    StubConfig,
    ValidationConfig,
)
from mcp_forge.validation.runner import (
    ValidationRunner,
    ValidationSample,
    generate_validation_samples,
)
from mcp_forge.validation.stubs import (
    FilesystemStub,
    MCPStub,
    StubRegistry,
    WeatherStub,
)

__all__ = [
    # Benchmark
    "BenchmarkConfig",
    "BenchmarkRunner",
    "LatencyStats",
    # Config
    "InferenceConfig",
    "StubConfig",
    "ValidationConfig",
    # Runner
    "ValidationRunner",
    "ValidationSample",
    "generate_validation_samples",
    # Stubs
    "MCPStub",
    "WeatherStub",
    "FilesystemStub",
    "StubRegistry",
]
```

**Verification**: `python -c "from mcp_forge.validation import BenchmarkRunner, BenchmarkConfig"`

---

### Task 5: Integrate with CLI

**Action**: Wire BenchmarkRunner into CLI `benchmark` command.

**File**: `src/mcp_forge/cli.py`

**Replace** benchmark command (lines 1018-1025):
```python
@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to model")
@click.option("--tools", "-t", required=True, type=click.Path(exists=True), help="Tools JSON file")
@click.option("--baseline", type=click.Path(exists=True), help="Baseline benchmark JSON for comparison")
@click.option("--samples-per-tool", default=20, help="Samples per tool (default: 20)")
@click.option("--samples-per-scenario", default=20, help="Samples per scenario (default: 20)")
@click.option("--stub", type=click.Choice(["weather", "filesystem"]), help="Use deterministic stub")
@click.option("--output", "-o", help="Output directory for reports")
@click.pass_context
def benchmark(
    ctx,
    model: str,
    tools: str,
    baseline: str | None,
    samples_per_tool: int,
    samples_per_scenario: int,
    stub: str | None,
    output: str | None,
):
    """Run full evaluation benchmark suite.

    Measures tool accuracy, no-tool correctness, and response latency
    across all tools and scenarios. Generates detailed reports in JSON
    and Markdown format.

    \b
    Metrics Measured:
      - Tool selection accuracy (target: ≥90%)
      - No-tool correctness (target: ≥85%)
      - Loop completion rate (target: ≥95%)
      - Response latency (mean, p50, p95, p99)

    \b
    Example:
      mcp-forge benchmark -m ./adapter -t tools.json --stub weather
    """
    import json

    from mcp_forge.state import BenchmarkResult, ToolDefinition
    from mcp_forge.validation import BenchmarkConfig, BenchmarkRunner
    from mcp_forge.validation.config import StubConfig

    console.print("\n[bold]MCP-Forge Benchmark Suite[/bold]")
    console.print("=" * 50)

    # Load tools
    with open(tools) as f:
        tool_defs = [ToolDefinition.from_dict(t) for t in json.load(f)]

    console.print(f"Model: {model}")
    console.print(f"Tools: {len(tool_defs)}")
    console.print(f"Samples: {samples_per_tool}/tool, {samples_per_scenario}/scenario")

    # Configure benchmark
    if stub:
        stub_config = StubConfig(stub_type=stub, deterministic=True)
        mcp_command = None
    else:
        stub_config = None
        mcp_command = None  # Would need --server option for real MCP

    if not stub_config and not mcp_command:
        console.print("[yellow]Warning: No --stub specified, using weather stub by default[/yellow]")
        stub_config = StubConfig(stub_type="weather", deterministic=True)

    config = BenchmarkConfig(
        model_path=Path(model),
        model_name=Path(model).name,
        samples_per_tool=samples_per_tool,
        samples_per_scenario=samples_per_scenario,
        stub_config=stub_config,
        mcp_command=mcp_command,
        baseline_path=Path(baseline) if baseline else None,
    )

    runner = BenchmarkRunner(config, tool_defs)

    # Run benchmark
    console.print("\nRunning benchmark...")

    def on_progress(category: str, current: int, total: int, pct: float) -> None:
        console.print(f"  {category}: {current}/{total} ({pct:.0%})")

    try:
        result = runner.run(progress_callback=on_progress)
    except Exception as e:
        console.print(f"\n[red]Benchmark failed: {e}[/red]")
        raise SystemExit(1) from e

    # Load and compare to baseline if provided
    if baseline:
        with open(baseline) as f:
            baseline_data = json.load(f)
        baseline_result = BenchmarkResult.from_dict(baseline_data)
        result.baseline_comparison = runner.compare_to_baseline(result, baseline_result)

    # Print results
    console.print("\n" + "=" * 50)
    console.print("[bold]Benchmark Results[/bold]")
    console.print(f"Overall Score: {result.overall_score:.1%}")

    console.print("\n[bold]Per-Tool Results:[/bold]")
    for tool, metrics in result.per_tool_results.items():
        console.print(f"  {tool}:")
        console.print(f"    Accuracy: {metrics.get('accuracy', 0):.1%}")
        console.print(f"    Schema: {metrics.get('schema', 0):.1%}")
        console.print(f"    Latency: {metrics.get('latency_mean_ms', 0):.1f}ms (p95: {metrics.get('latency_p95_ms', 0):.1f}ms)")

    console.print("\n[bold]Per-Scenario Results:[/bold]")
    for scenario, metrics in result.per_scenario_results.items():
        console.print(f"  {scenario}: {metrics.get('pass_rate', 0):.1%}")

    if result.baseline_comparison:
        console.print("\n[bold]Baseline Comparison:[/bold]")
        console.print(f"  vs {result.baseline_comparison['baseline_model']}")
        delta = result.baseline_comparison['overall_delta']
        color = "green" if delta >= 0 else "red"
        console.print(f"  Overall delta: [{color}]{delta:+.1%}[/{color}]")

    # Save reports
    state_manager: StateManager = ctx.obj["state_manager"]
    report_path = state_manager.save_benchmark_result(result)
    console.print(f"\nReports saved: {report_path}")

    # Check thresholds
    no_tool_rate = result.per_scenario_results.get("no_tool", {}).get("pass_rate", 0)
    tool_accuracies = [m.get("accuracy", 0) for m in result.per_tool_results.values()]
    avg_accuracy = sum(tool_accuracies) / len(tool_accuracies) if tool_accuracies else 0

    passes = (
        avg_accuracy >= config.accuracy_threshold and
        no_tool_rate >= config.no_tool_threshold
    )

    if passes:
        console.print("\n[green]Benchmark passed all thresholds![/green]")
    else:
        console.print("\n[yellow]Warning: Some metrics below target thresholds[/yellow]")
        if avg_accuracy < config.accuracy_threshold:
            console.print(f"  Tool accuracy: {avg_accuracy:.1%} (target: {config.accuracy_threshold:.0%})")
        if no_tool_rate < config.no_tool_threshold:
            console.print(f"  No-tool correctness: {no_tool_rate:.1%} (target: {config.no_tool_threshold:.0%})")
```

**Also update** pipeline Stage 6 (cli.py:339-346):
```python
    # Stage 6: Benchmark (optional)
    if not skip_benchmark and state.stage in (PipelineStage.VALIDATING, PipelineStage.BENCHMARKING):
        console.print("\n[bold]Stage 6: Running benchmarks...[/bold]")
        state.update_stage(PipelineStage.BENCHMARKING)
        state_manager.save_state(state)

        from mcp_forge.validation import BenchmarkConfig, BenchmarkRunner
        from mcp_forge.validation.config import StubConfig

        # Use weather stub for pipeline benchmarks
        stub_config = StubConfig(stub_type="weather", deterministic=True)
        bench_config = BenchmarkConfig(
            model_path=Path(state.lora_adapter_path),
            model_name=state.model_family,
            samples_per_tool=10,  # Reduced for pipeline speed
            samples_per_scenario=10,
            stub_config=stub_config,
        )

        runner = BenchmarkRunner(bench_config, state.tools)

        def on_progress(category: str, current: int, total: int, pct: float) -> None:
            console.print(f"   {category}: {pct:.0%}")

        result = runner.run(progress_callback=on_progress)

        state.benchmark_result = result
        state_manager.save_state(state)

        # Save report
        report_path = state_manager.save_benchmark_result(result)

        console.print(f"   Overall score: {result.overall_score:.1%}")
        console.print(f"   Report: {report_path}")

        if result.overall_score >= 0.85:
            console.print("   [green]Benchmark passed[/green]")
        else:
            console.print("   [yellow]Benchmark below target (continuing)[/yellow]")
```

**Verification**: CLI smoke test with `mcp-forge benchmark --help`

---

### Task 6: Add tests

**Action**: Create unit and integration tests for benchmark module.

**Files**:
- `tests/unit/test_benchmark_config.py` - Config validation
- `tests/unit/test_benchmark_runner.py` - Runner logic with mocked model
- `tests/unit/test_latency_stats.py` - Latency calculation
- `tests/integration/test_benchmark_pipeline.py` - Pipeline integration

**Test Cases**:

```python
# test_benchmark_config.py
def test_config_requires_server_or_stub():
    """Config raises if neither server nor stub provided."""

def test_config_accepts_stub_config():
    """Config accepts valid stub configuration."""

def test_config_defaults():
    """Config has sensible defaults for samples and thresholds."""

# test_latency_stats.py
def test_latency_stats_empty_list():
    """LatencyStats handles empty list gracefully."""

def test_latency_stats_single_sample():
    """LatencyStats handles single sample."""

def test_latency_stats_percentiles():
    """LatencyStats calculates correct percentiles."""

def test_latency_stats_to_dict():
    """LatencyStats serializes correctly."""

# test_benchmark_runner.py
@pytest.fixture
def mock_model():
    """Mock FastLanguageModel for tests without GPU."""

def test_generate_tool_samples(mock_model):
    """Sample generation creates diverse prompts for each tool."""

def test_generate_scenario_samples(mock_model):
    """Sample generation creates appropriate samples per scenario."""

def test_run_aggregates_per_tool(mock_model):
    """Run correctly aggregates per-tool metrics."""

def test_run_aggregates_per_scenario(mock_model):
    """Run correctly aggregates per-scenario metrics."""

def test_run_tracks_latency(mock_model):
    """Run tracks and aggregates latency metrics."""

def test_compare_to_baseline(mock_model):
    """Baseline comparison calculates correct deltas."""

def test_overall_score_calculation(mock_model):
    """Overall score combines accuracy and no-tool rate correctly."""

# test_benchmark_pipeline.py
@pytest.mark.integration
def test_pipeline_benchmark_stage(mock_model, state_manager):
    """Benchmark stage integrates with pipeline state."""

@pytest.mark.integration
def test_benchmark_report_generation(mock_model, state_manager):
    """Benchmark generates JSON and Markdown reports."""
```

**Fixtures**:
```python
@pytest.fixture
def mock_unsloth_inference(monkeypatch):
    """Mock Unsloth for benchmark inference tests."""

@pytest.fixture
def sample_benchmark_result():
    """Create a sample BenchmarkResult for comparison tests."""
```

**Verification**: `pytest tests/unit/test_benchmark*.py tests/unit/test_latency*.py -v`

---

### Task 7: Add random import and fix minor issues

**Action**: Add missing import and ensure all code compiles.

**File**: `src/mcp_forge/validation/benchmark.py`

**Add** at top of file after existing imports:
```python
import random
```

**Verification**: `python -c "from mcp_forge.validation.benchmark import BenchmarkRunner; print('OK')"`

---

## Verification

After all tasks complete:

```bash
# 1. Lint and type check
ruff check src/mcp_forge/validation/benchmark.py
mypy src/mcp_forge/validation/benchmark.py

# 2. Run benchmark tests
pytest tests/unit/test_benchmark*.py tests/unit/test_latency*.py -v
pytest tests/integration/test_benchmark*.py -v

# 3. Check coverage
pytest --cov=src/mcp_forge/validation --cov-report=term-missing

# 4. CLI smoke tests
mcp-forge benchmark --help
mcp-forge benchmark --model ./test-adapter --tools tools.json --stub weather --samples-per-tool 5

# 5. Verify imports
python -c "from mcp_forge.validation import BenchmarkRunner, BenchmarkConfig, LatencyStats; print('OK')"
```

---

## Success Criteria

- [ ] `src/mcp_forge/validation/benchmark.py` created with BenchmarkConfig, LatencyStats, BenchmarkRunner
- [ ] BenchmarkConfig supports stub/server modes with thresholds
- [ ] LatencyStats calculates min, max, mean, p50, p95, p99
- [ ] BenchmarkRunner generates samples per-tool and per-scenario
- [ ] BenchmarkRunner tracks latency with warmup
- [ ] BenchmarkRunner produces BenchmarkResult with per-tool and per-scenario metrics
- [ ] Baseline comparison calculates deltas correctly
- [ ] CLI `benchmark` command fully functional
- [ ] Pipeline stage 6 transitions correctly (VALIDATING -> BENCHMARKING)
- [ ] JSON and Markdown reports generated via StateManager
- [ ] All tests pass with mocked Unsloth (no GPU required)
- [ ] Coverage maintained at 85%+
- [ ] ruff and mypy pass on new code

---

## Output

| Artifact | Path |
|----------|------|
| Benchmark module | `src/mcp_forge/validation/benchmark.py` |
| Config tests | `tests/unit/test_benchmark_config.py` |
| Latency tests | `tests/unit/test_latency_stats.py` |
| Runner tests | `tests/unit/test_benchmark_runner.py` |
| Integration tests | `tests/integration/test_benchmark_pipeline.py` |
| Benchmark reports (runtime) | `.mcp-forge/reports/benchmark_*.json`, `.mcp-forge/reports/benchmark_*.md` |

---

## Notes

### Benchmarking vs Validation

| Stage | Purpose | Speed | Detail |
|-------|---------|-------|--------|
| Validation (5) | Quick pass/fail check | Fast (20 samples) | Aggregated metrics |
| Benchmarking (6) | Comprehensive evaluation | Slower (100+ samples) | Per-tool, per-scenario, latency |

Validation is a gate; benchmarking is for reporting and comparison.

### Latency Measurement

- Warmup samples excluded from latency stats
- Time measured per-sample, not per-batch
- Includes model inference + stub execution
- P95/P99 require sufficient samples (20/100 respectively)

### Baseline Comparison

Load a previous benchmark JSON to compare:
```bash
mcp-forge benchmark -m ./new-adapter -t tools.json --baseline ./reports/benchmark_old.json
```

Produces delta for:
- Overall score
- Per-tool accuracy
- Per-tool latency

---

*Plan created: 2026-01-13*
*Execute with: /gsd:execute-plan*
