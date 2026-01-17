"""Pipeline state management with checkpoint and resume support.

v1.1: Added QC reports, looped validation, benchmark results, bundle tracking.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


class PipelineStage(str, Enum):
    """Pipeline execution stages."""

    IDLE = "idle"
    INSPECTING = "inspecting"
    SYNTHESIZING = "synthesizing"
    QC_VALIDATING = "qc_validating"  # v1.1: New QC stage
    FORMATTING = "formatting"
    TRAINING = "training"
    VALIDATING = "validating"
    BENCHMARKING = "benchmarking"  # v1.1: New benchmark stage
    EXPORTING = "exporting"
    PACKAGING = "packaging"  # v1.1: New packaging stage
    COMPLETE = "complete"
    FAILED = "failed"


class Scenario(str, Enum):
    """Training data scenario types (v1.1)."""

    STANDARD = "standard"      # Normal tool call
    NO_TOOL = "no_tool"        # Answer directly, no tool needed
    ERROR = "error"            # Tool returns error
    AMBIGUOUS = "ambiguous"    # Multiple tools could apply
    EDGE = "edge"              # Boundary conditions


@dataclass
class ToolDefinition:
    """MCP tool definition extracted from server."""

    name: str
    description: str
    input_schema: dict[str, Any]
    source: str = "mcp"  # v1.1: Track source (mcp, file, openai)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolDefinition:
        return cls(
            name=data["name"],
            description=data["description"],
            input_schema=data["input_schema"],
            source=data.get("source", "mcp")
        )

    def schema_hash(self) -> str:
        """Generate hash of the tool schema for drift detection."""
        schema_str = json.dumps(self.input_schema, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


@dataclass
class QCReport:
    """Quality control analysis results (v1.1)."""

    total_samples: int
    valid_samples: int
    dropped_samples: int

    schema_pass_rate: float
    dedup_rate: float

    tool_coverage: dict[str, int]  # tool_name -> count
    scenario_coverage: dict[str, int]  # scenario -> count

    issues: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QCReport:
        return cls(**data)

    def passes_threshold(self, min_schema_rate: float = 0.98, min_per_tool: int = 10) -> bool:
        """Check if QC report passes quality thresholds."""
        if self.schema_pass_rate < min_schema_rate:
            return False
        return not any(count < min_per_tool for count in self.tool_coverage.values())


@dataclass
class ValidationResult:
    """Results from looped validation (v1.1 enhanced)."""

    passed: bool
    samples_tested: int
    samples_passed: int

    # v1.1: Detailed metrics
    tool_call_parse_rate: float = 0.0
    schema_conformance_rate: float = 0.0
    tool_selection_accuracy: float = 0.0
    loop_completion_rate: float = 0.0
    error_handling_rate: float = 0.0

    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult:
        return cls(**data)

    def meets_release_criteria(self) -> bool:
        """Check if validation meets release thresholds."""
        return (
            self.tool_call_parse_rate >= 0.98 and
            self.schema_conformance_rate >= 0.95 and
            self.tool_selection_accuracy >= 0.90 and
            self.loop_completion_rate >= 0.95
        )


@dataclass
class BenchmarkResult:
    """Full benchmark evaluation results (v1.1)."""

    model_name: str
    timestamp: str

    overall_score: float

    per_tool_results: dict[str, dict[str, float]] = field(default_factory=dict)
    per_scenario_results: dict[str, dict[str, float]] = field(default_factory=dict)

    baseline_comparison: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        return cls(**data)


@dataclass
class SynthesisPlan:
    """Controls data generation distribution (v1.1)."""

    total_samples: int
    seed_samples: int
    augmented_samples: int

    tool_weights: dict[str, float] = field(default_factory=dict)  # tool -> weight

    scenario_weights: dict[str, float] = field(default_factory=lambda: {
        "standard": 0.60,
        "no_tool": 0.15,
        "error": 0.10,
        "ambiguous": 0.10,
        "edge": 0.05
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SynthesisPlan:
        return cls(**data)

    def get_samples_per_scenario(self) -> dict[str, int]:
        """Calculate target samples per scenario."""
        return {
            scenario: int(self.total_samples * weight)
            for scenario, weight in self.scenario_weights.items()
        }


@dataclass
class PipelineState:
    """Full pipeline state for checkpoint/resume (v1.1 enhanced)."""

    session_id: str
    stage: PipelineStage
    mcp_command: str
    system_prompt: str
    model_family: str
    output_path: str
    quantization: str

    # v1.1: Training profile
    profile: str = "balanced"

    # Stage outputs (populated as pipeline progresses)
    tools: list[ToolDefinition] = field(default_factory=list)
    synthesis_plan: SynthesisPlan | None = None  # v1.1
    seed_data_path: str | None = None
    training_data_path: str | None = None

    # v1.1: QC tracking
    qc_report: QCReport | None = None

    lora_adapter_path: str | None = None
    validation_result: ValidationResult | None = None

    # v1.1: Benchmark tracking
    benchmark_result: BenchmarkResult | None = None

    gguf_path: str | None = None

    # v1.1: Bundle tracking
    bundle_path: str | None = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: str | None = None

    # Training progress
    training_progress: float = 0.0
    training_loss: float | None = None

    # v1.1: Toolset hash for drift detection
    toolset_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "stage": self.stage.value,
            "mcp_command": self.mcp_command,
            "system_prompt": self.system_prompt,
            "model_family": self.model_family,
            "output_path": self.output_path,
            "quantization": self.quantization,
            "profile": self.profile,
            "tools": [t.to_dict() for t in self.tools],
            "synthesis_plan": self.synthesis_plan.to_dict() if self.synthesis_plan else None,
            "seed_data_path": self.seed_data_path,
            "training_data_path": self.training_data_path,
            "qc_report": self.qc_report.to_dict() if self.qc_report else None,
            "lora_adapter_path": self.lora_adapter_path,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "benchmark_result": self.benchmark_result.to_dict() if self.benchmark_result else None,
            "gguf_path": self.gguf_path,
            "bundle_path": self.bundle_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "training_progress": self.training_progress,
            "training_loss": self.training_loss,
            "toolset_hash": self.toolset_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineState:
        tools = [ToolDefinition.from_dict(t) for t in data.get("tools", [])]

        synthesis_plan = None
        if data.get("synthesis_plan"):
            synthesis_plan = SynthesisPlan.from_dict(data["synthesis_plan"])

        qc_report = None
        if data.get("qc_report"):
            qc_report = QCReport.from_dict(data["qc_report"])

        validation = None
        if data.get("validation_result"):
            validation = ValidationResult.from_dict(data["validation_result"])

        benchmark = None
        if data.get("benchmark_result"):
            benchmark = BenchmarkResult.from_dict(data["benchmark_result"])

        return cls(
            session_id=data["session_id"],
            stage=PipelineStage(data["stage"]),
            mcp_command=data["mcp_command"],
            system_prompt=data["system_prompt"],
            model_family=data["model_family"],
            output_path=data["output_path"],
            quantization=data["quantization"],
            profile=data.get("profile", "balanced"),
            tools=tools,
            synthesis_plan=synthesis_plan,
            seed_data_path=data.get("seed_data_path"),
            training_data_path=data.get("training_data_path"),
            qc_report=qc_report,
            lora_adapter_path=data.get("lora_adapter_path"),
            validation_result=validation,
            benchmark_result=benchmark,
            gguf_path=data.get("gguf_path"),
            bundle_path=data.get("bundle_path"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            error=data.get("error"),
            training_progress=data.get("training_progress", 0.0),
            training_loss=data.get("training_loss"),
            toolset_hash=data.get("toolset_hash"),
        )

    def update_stage(self, stage: PipelineStage) -> None:
        """Update stage and timestamp."""
        self.stage = stage
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def set_error(self, error: str) -> None:
        """Mark pipeline as failed with error message."""
        self.error = error
        self.stage = PipelineStage.FAILED
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def compute_toolset_hash(self) -> str:
        """Compute hash of all tool schemas for drift detection."""
        if not self.tools:
            return ""
        tool_hashes = sorted([t.schema_hash() for t in self.tools])
        combined = ":".join(tool_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


class StateManager:
    """Manages pipeline state persistence (v1.1 enhanced)."""

    STATE_DIR = ".mcp-forge"
    STATE_FILE = "state.json"
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    REPORTS_DIR = "reports"  # v1.1: QC and benchmark reports

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path.cwd()
        self.state_dir = self.base_path / self.STATE_DIR
        self.state_file = self.state_dir / self.STATE_FILE
        self.data_dir = self.state_dir / self.DATA_DIR
        self.logs_dir = self.state_dir / self.LOGS_DIR
        self.reports_dir = self.state_dir / self.REPORTS_DIR  # v1.1

    def ensure_dirs(self) -> None:
        """Create necessary directories."""
        self.state_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)  # v1.1

    def create_session(
        self,
        mcp_command: str,
        system_prompt: str,
        model_family: str,
        output_path: str,
        quantization: str = "q8_0",
        profile: str = "balanced",  # v1.1
    ) -> PipelineState:
        """Create a new pipeline session."""
        self.ensure_dirs()

        state = PipelineState(
            session_id=str(uuid.uuid4())[:8],
            stage=PipelineStage.IDLE,
            mcp_command=mcp_command,
            system_prompt=system_prompt,
            model_family=model_family,
            output_path=output_path,
            quantization=quantization,
            profile=profile,  # v1.1
        )

        self.save_state(state)
        return state

    def save_state(self, state: PipelineState) -> None:
        """Atomically save state to disk."""
        self.ensure_dirs()
        state.updated_at = datetime.now(timezone.utc).isoformat()

        # Write to temp file first, then rename (atomic on POSIX)
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        temp_file.rename(self.state_file)

    def load_state(self) -> PipelineState | None:
        """Load existing state from disk."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file) as f:
                data = json.load(f)
            return PipelineState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            console.print(f"[yellow]Warning: Could not load state file: {e}[/yellow]")
            return None

    def clear_state(self) -> None:
        """Remove existing state (start fresh)."""
        if self.state_file.exists():
            self.state_file.unlink()

    def get_data_path(self, filename: str) -> Path:
        """Get path for a data file."""
        self.ensure_dirs()
        return self.data_dir / filename

    def get_log_path(self, filename: str) -> Path:
        """Get path for a log file."""
        self.ensure_dirs()
        return self.logs_dir / filename

    def get_report_path(self, filename: str) -> Path:
        """Get path for a report file (v1.1)."""
        self.ensure_dirs()
        return self.reports_dir / filename

    def can_resume(self) -> bool:
        """Check if there's a resumable session."""
        state = self.load_state()
        if state is None:
            return False
        return state.stage not in (PipelineStage.IDLE, PipelineStage.COMPLETE, PipelineStage.FAILED)

    def get_resume_stage(self) -> PipelineStage | None:
        """Get the stage to resume from."""
        state = self.load_state()
        if state is None:
            return None

        # Resume from the beginning of the current stage
        return state.stage

    def save_qc_report(self, report: QCReport) -> Path:
        """Save QC report to reports directory (v1.1)."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"qc_{timestamp}.json"
        path = self.get_report_path(filename)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        return path

    def save_benchmark_result(self, result: BenchmarkResult) -> Path:
        """Save benchmark result to reports directory (v1.1)."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_path = self.get_report_path(f"benchmark_{timestamp}.json")
        with open(json_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Save Markdown summary
        md_path = self.get_report_path(f"benchmark_{timestamp}.md")
        with open(md_path, "w") as f:
            f.write(self._generate_benchmark_markdown(result))

        return json_path

    def _generate_benchmark_markdown(self, result: BenchmarkResult) -> str:
        """Generate markdown report from benchmark result."""
        lines = [
            f"# Benchmark Report: {result.model_name}",
            "",
            f"**Generated:** {result.timestamp}",
            f"**Overall Score:** {result.overall_score:.1%}",
            "",
            "## Per-Tool Results",
            "",
            "| Tool | Accuracy | Schema | Latency |",
            "|------|----------|--------|---------|",
        ]

        for tool, metrics in result.per_tool_results.items():
            lines.append(
                f"| {tool} | {metrics.get('accuracy', 0):.0%} | "
                f"{metrics.get('schema', 0):.0%} | {metrics.get('latency', 0):.1f}s |"
            )

        lines.extend([
            "",
            "## Per-Scenario Results",
            "",
            "| Scenario | Pass Rate |",
            "|----------|-----------|",
        ])

        for scenario, metrics in result.per_scenario_results.items():
            lines.append(f"| {scenario} | {metrics.get('pass_rate', 0):.0%} |")

        if result.baseline_comparison:
            lines.extend([
                "",
                "## Baseline Comparison",
                "",
                f"Delta: {result.baseline_comparison.get('delta', 'N/A')}",
            ])

        return "\n".join(lines)
