# Phase 8: GGUF Export - Execution Plan

**Phase**: 8 of 9
**Milestone**: v1.0 - Full Pipeline Implementation
**Created**: 2026-01-13
**Estimated Scope**: Medium (7 tasks)

---

## Objective

Implement the EXPORTING stage to convert fine-tuned LoRA adapters to GGUF format. Create an `export/` module that:
1. Merges LoRA adapter weights with base model
2. Converts merged model to GGUF format using llama.cpp
3. Supports multiple quantization levels (Q8_0 default, Q4_K_M for smaller size)
4. Embeds model metadata (training info, tool signatures, timestamp)
5. Verifies the exported GGUF loads correctly
6. Generates export reports with size/quality metrics

---

## Execution Context

**Architecture**: Follow patterns in `validation/benchmark.py`, `training/engine.py`
**State Integration**: `gguf_path` field exists in `state.py:225`
**CLI Stub**: `cli.py:1199-1205` - export command exists but not implemented
**Pipeline**: `cli.py:380-388` - Stage 7 transition point (currently TODO)
**Testing**: 364 existing tests, maintain 85%+ coverage
**Conventions**: Dataclasses, type hints, docstrings per CONVENTIONS.md

---

## Context

### Existing Infrastructure

| Component | Path | Integration Point |
|-----------|------|-------------------|
| PipelineState.gguf_path | `state.py:225` | Output path for exported GGUF |
| PipelineState.lora_adapter_path | `state.py:219` | Input from training stage |
| PipelineState.quantization | `state.py:205` | Selected quantization format |
| CLI Stub | `cli.py:1199-1205` | export command skeleton |
| Pipeline Stub | `cli.py:380-388` | Stage 7 EXPORTING transition |
| TrainingEngine | `training/engine.py:193-198` | LoRA adapter output format |

### GGUF Export Flow

```
LoRA Adapter (training output)
            ↓
    Load Base Model + Adapter
            ↓
    Merge LoRA Weights  ───────────→ Full 16-bit model
            ↓
    Quantize + Convert  ───────────→ Q8_0 or Q4_K_M GGUF
            ↓
    Embed Metadata      ───────────→ tools, training info
            ↓
    Verify Load         ───────────→ llama-cpp validation
            ↓
    ExportResult + Report
```

### Supported Models & Quantization

| Model | Base Size | Q8_0 Size | Q4_K_M Size |
|-------|-----------|-----------|-------------|
| DeepSeek-R1-Distill-8B | ~16GB | ~8GB | ~5GB |
| Qwen-2.5-14B-Instruct | ~28GB | ~14GB | ~9GB |

| Quantization | Quality | Size | Speed | Use Case |
|--------------|---------|------|-------|----------|
| Q8_0 | Best | Larger | Slower | Production, quality-critical |
| Q4_K_M | Good | Smaller | Faster | Edge deployment, VRAM-limited |

### Dependencies Required

llama-cpp-python is **not currently installed**. The plan adds this dependency.

---

## Tasks

### Task 1: Add llama-cpp-python dependency

**Action**: Update `pyproject.toml` to include llama-cpp-python for GGUF conversion.

**File**: `pyproject.toml`

**Edit**: Add to dependencies list (after line 43):
```python
    "llama-cpp-python>=0.2.50",
```

**Verification**: `pip install -e . && python -c "import llama_cpp; print('OK')"`

---

### Task 2: Create export module structure

**Action**: Create `src/mcp_forge/export/` package with module files.

**Files**:
- `src/mcp_forge/export/__init__.py` - Module exports
- `src/mcp_forge/export/config.py` - ExportConfig dataclass
- `src/mcp_forge/export/engine.py` - ExportEngine class
- `src/mcp_forge/export/metadata.py` - GGUF metadata handling

**Structure**:
```
src/mcp_forge/export/
├── __init__.py         # Export ExportEngine, ExportConfig, ExportResult
├── config.py           # ExportConfig with quantization options
├── engine.py           # ExportEngine class for GGUF conversion
└── metadata.py         # GGUF metadata embedding
```

**Verification**: `python -c "from mcp_forge.export import ExportEngine"`

---

### Task 3: Implement configuration dataclasses

**Action**: Create configuration classes for export runs.

**File**: `src/mcp_forge/export/config.py`

**Implementation**:
```python
"""Configuration for GGUF export operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class QuantizationType(str, Enum):
    """Supported GGUF quantization types."""

    Q8_0 = "q8_0"       # 8-bit quantization, best quality
    Q4_K_M = "q4_k_m"   # 4-bit k-quant medium, good balance
    Q4_K_S = "q4_k_s"   # 4-bit k-quant small, smaller size
    Q5_K_M = "q5_k_m"   # 5-bit k-quant medium
    F16 = "f16"         # No quantization (half precision)


@dataclass
class ExportConfig:
    """Configuration for GGUF export."""

    adapter_path: Path              # Path to LoRA adapter directory
    output_path: Path               # Output GGUF file path
    base_model: str                 # Base model name/path for merging

    # Quantization settings
    quantization: QuantizationType = QuantizationType.Q8_0

    # Metadata to embed
    model_name: str = ""            # Human-readable model name
    tool_names: list[str] = field(default_factory=list)
    training_timestamp: str = ""

    # Conversion settings
    vocab_only: bool = False        # Only export vocabulary (for testing)
    allow_requantize: bool = False  # Allow re-quantizing already quantized

    # Verification
    verify_after_export: bool = True  # Load and verify GGUF after conversion

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.adapter_path.exists():
            raise ValueError(f"Adapter path does not exist: {self.adapter_path}")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Set default model name
        if not self.model_name:
            self.model_name = self.adapter_path.name

        # Set training timestamp if not provided
        if not self.training_timestamp:
            self.training_timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ExportResult:
    """Result of a GGUF export operation."""

    success: bool
    output_path: Path | None

    # Size metrics
    adapter_size_mb: float = 0.0
    merged_size_mb: float = 0.0
    gguf_size_mb: float = 0.0
    compression_ratio: float = 0.0

    # Timing
    merge_time_seconds: float = 0.0
    convert_time_seconds: float = 0.0
    total_time_seconds: float = 0.0

    # Verification
    verified: bool = False
    verification_error: str | None = None

    # Metadata embedded
    metadata: dict[str, Any] = field(default_factory=dict)

    # Error info
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "adapter_size_mb": self.adapter_size_mb,
            "merged_size_mb": self.merged_size_mb,
            "gguf_size_mb": self.gguf_size_mb,
            "compression_ratio": self.compression_ratio,
            "merge_time_seconds": self.merge_time_seconds,
            "convert_time_seconds": self.convert_time_seconds,
            "total_time_seconds": self.total_time_seconds,
            "verified": self.verified,
            "verification_error": self.verification_error,
            "metadata": self.metadata,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportResult:
        """Create from dictionary."""
        return cls(
            success=data["success"],
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            adapter_size_mb=data.get("adapter_size_mb", 0.0),
            merged_size_mb=data.get("merged_size_mb", 0.0),
            gguf_size_mb=data.get("gguf_size_mb", 0.0),
            compression_ratio=data.get("compression_ratio", 0.0),
            merge_time_seconds=data.get("merge_time_seconds", 0.0),
            convert_time_seconds=data.get("convert_time_seconds", 0.0),
            total_time_seconds=data.get("total_time_seconds", 0.0),
            verified=data.get("verified", False),
            verification_error=data.get("verification_error"),
            metadata=data.get("metadata", {}),
            error=data.get("error"),
        )
```

**Verification**: Unit tests for config validation

---

### Task 4: Implement GGUF metadata handling

**Action**: Create metadata embedding utilities for GGUF files.

**File**: `src/mcp_forge/export/metadata.py`

**Implementation**:
```python
"""GGUF metadata handling for MCP-Forge exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class GGUFMetadata:
    """Metadata to embed in GGUF model files.

    This metadata helps identify the model's purpose and capabilities
    when loaded by inference engines like llama.cpp or Ollama.
    """

    # Model identification
    model_name: str
    model_family: str  # deepseek-r1, qwen-2.5

    # MCP-Forge specific
    forge_version: str = "0.1.0"
    tool_names: list[str] = field(default_factory=list)
    tool_count: int = 0

    # Training provenance
    training_timestamp: str = ""
    training_samples: int = 0
    training_epochs: int = 0
    base_model: str = ""

    # Quality metrics (from validation/benchmark)
    tool_accuracy: float = 0.0
    schema_conformance: float = 0.0
    benchmark_score: float = 0.0

    # Export info
    export_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    quantization_type: str = "q8_0"

    def __post_init__(self) -> None:
        """Set derived fields."""
        if not self.tool_count:
            self.tool_count = len(self.tool_names)

    def to_gguf_kv(self) -> dict[str, Any]:
        """Convert to GGUF key-value pairs.

        Returns dictionary suitable for embedding in GGUF metadata.
        Keys follow GGUF naming convention (lowercase, dots for namespacing).
        """
        return {
            # Standard GGUF fields
            "general.name": self.model_name,
            "general.architecture": self.model_family,
            "general.quantization_version": 2,

            # MCP-Forge custom fields (namespaced under mcp_forge.)
            "mcp_forge.version": self.forge_version,
            "mcp_forge.tool_count": self.tool_count,
            "mcp_forge.tool_names": ",".join(self.tool_names),
            "mcp_forge.training_timestamp": self.training_timestamp,
            "mcp_forge.training_samples": self.training_samples,
            "mcp_forge.base_model": self.base_model,
            "mcp_forge.tool_accuracy": self.tool_accuracy,
            "mcp_forge.schema_conformance": self.schema_conformance,
            "mcp_forge.benchmark_score": self.benchmark_score,
            "mcp_forge.export_timestamp": self.export_timestamp,
            "mcp_forge.quantization_type": self.quantization_type,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "forge_version": self.forge_version,
            "tool_names": self.tool_names,
            "tool_count": self.tool_count,
            "training_timestamp": self.training_timestamp,
            "training_samples": self.training_samples,
            "training_epochs": self.training_epochs,
            "base_model": self.base_model,
            "tool_accuracy": self.tool_accuracy,
            "schema_conformance": self.schema_conformance,
            "benchmark_score": self.benchmark_score,
            "export_timestamp": self.export_timestamp,
            "quantization_type": self.quantization_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GGUFMetadata:
        """Create from dictionary."""
        return cls(
            model_name=data.get("model_name", "unknown"),
            model_family=data.get("model_family", "unknown"),
            forge_version=data.get("forge_version", "0.1.0"),
            tool_names=data.get("tool_names", []),
            tool_count=data.get("tool_count", 0),
            training_timestamp=data.get("training_timestamp", ""),
            training_samples=data.get("training_samples", 0),
            training_epochs=data.get("training_epochs", 0),
            base_model=data.get("base_model", ""),
            tool_accuracy=data.get("tool_accuracy", 0.0),
            schema_conformance=data.get("schema_conformance", 0.0),
            benchmark_score=data.get("benchmark_score", 0.0),
            export_timestamp=data.get("export_timestamp", ""),
            quantization_type=data.get("quantization_type", "q8_0"),
        )


def read_gguf_metadata(gguf_path: str) -> dict[str, Any]:
    """Read metadata from an existing GGUF file.

    Args:
        gguf_path: Path to GGUF file

    Returns:
        Dictionary of metadata key-value pairs
    """
    try:
        from llama_cpp import Llama

        # Load with minimal context to just read metadata
        llm = Llama(model_path=gguf_path, n_ctx=32, n_gpu_layers=0, verbose=False)

        # Extract metadata
        metadata = {}
        if hasattr(llm, "metadata"):
            metadata = dict(llm.metadata)

        del llm
        return metadata

    except Exception as e:
        return {"error": str(e)}
```

**Verification**: Unit tests for metadata serialization

---

### Task 5: Implement ExportEngine

**Action**: Create the main export engine with model merging and GGUF conversion.

**File**: `src/mcp_forge/export/engine.py`

**Implementation**:
```python
"""GGUF export engine for MCP-Forge.

Converts LoRA-tuned models to GGUF format for deployment with
llama.cpp, Ollama, and other inference engines.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mcp_forge.export.config import ExportConfig, ExportResult, QuantizationType
from mcp_forge.export.metadata import GGUFMetadata


class ExportEngine:
    """Engine for exporting LoRA adapters to GGUF format."""

    # Model family to base model mapping
    BASE_MODELS = {
        "deepseek-r1": "unsloth/DeepSeek-R1-Distill-Qwen-7B",
        "qwen-2.5": "unsloth/Qwen2.5-14B-Instruct",
    }

    def __init__(self, config: ExportConfig) -> None:
        """Initialize export engine.

        Args:
            config: Export configuration
        """
        self.config = config
        self._temp_dir: Path | None = None

    def _get_dir_size_mb(self, path: Path) -> float:
        """Calculate directory size in MB."""
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total / (1024 * 1024)

    def _get_file_size_mb(self, path: Path) -> float:
        """Calculate file size in MB."""
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
        return 0.0

    def merge_adapter(
        self,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Path:
        """Merge LoRA adapter with base model.

        Args:
            progress_callback: Optional callback(stage, progress_pct)

        Returns:
            Path to merged model directory
        """
        from unsloth import FastLanguageModel

        if progress_callback:
            progress_callback("Loading base model", 0.1)

        # Load base model with adapter
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.config.adapter_path),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        if progress_callback:
            progress_callback("Merging LoRA weights", 0.4)

        # Merge LoRA weights into base model
        model = model.merge_and_unload()

        if progress_callback:
            progress_callback("Saving merged model", 0.7)

        # Save to temp directory
        self._temp_dir = Path(tempfile.mkdtemp(prefix="mcp_forge_export_"))
        merged_path = self._temp_dir / "merged"
        merged_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)

        if progress_callback:
            progress_callback("Merge complete", 1.0)

        return merged_path

    def convert_to_gguf(
        self,
        merged_path: Path,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Path:
        """Convert merged model to GGUF format.

        Args:
            merged_path: Path to merged model directory
            progress_callback: Optional callback(stage, progress_pct)

        Returns:
            Path to output GGUF file
        """
        if progress_callback:
            progress_callback("Preparing GGUF conversion", 0.1)

        # Use Unsloth's built-in GGUF export
        from unsloth import FastLanguageModel

        # Reload the merged model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(merged_path),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,  # Full precision for conversion
        )

        if progress_callback:
            progress_callback("Converting to GGUF", 0.5)

        # Map our quantization enum to Unsloth's format
        quant_method = self._get_unsloth_quant_method()

        # Export using Unsloth
        model.save_pretrained_gguf(
            str(self.config.output_path.parent),
            tokenizer,
            quantization_method=quant_method,
        )

        # Unsloth saves with model name, we need to rename
        expected_output = self.config.output_path.parent / f"unsloth.{quant_method.upper()}.gguf"
        if expected_output.exists() and expected_output != self.config.output_path:
            shutil.move(str(expected_output), str(self.config.output_path))

        if progress_callback:
            progress_callback("GGUF conversion complete", 1.0)

        return self.config.output_path

    def _get_unsloth_quant_method(self) -> str:
        """Map QuantizationType to Unsloth quantization method string."""
        mapping = {
            QuantizationType.Q8_0: "q8_0",
            QuantizationType.Q4_K_M: "q4_k_m",
            QuantizationType.Q4_K_S: "q4_k_s",
            QuantizationType.Q5_K_M: "q5_k_m",
            QuantizationType.F16: "f16",
        }
        return mapping.get(self.config.quantization, "q8_0")

    def verify_gguf(self, gguf_path: Path) -> tuple[bool, str | None]:
        """Verify exported GGUF file loads correctly.

        Args:
            gguf_path: Path to GGUF file

        Returns:
            Tuple of (success, error_message)
        """
        try:
            from llama_cpp import Llama

            # Try to load with minimal context
            llm = Llama(
                model_path=str(gguf_path),
                n_ctx=32,
                n_gpu_layers=0,  # CPU-only for verification
                verbose=False,
            )

            # Basic sanity check - model should have vocabulary
            if llm.n_vocab() < 1000:
                return False, f"Model vocabulary too small: {llm.n_vocab()}"

            del llm
            return True, None

        except Exception as e:
            return False, str(e)

    def export(
        self,
        metadata: GGUFMetadata | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ExportResult:
        """Run full export pipeline.

        Args:
            metadata: Optional metadata to embed in GGUF
            progress_callback: Optional callback(stage, progress_pct)

        Returns:
            ExportResult with success status and metrics
        """
        start_time = time.perf_counter()
        result = ExportResult(success=False, output_path=None)

        try:
            # Measure adapter size
            result.adapter_size_mb = self._get_dir_size_mb(self.config.adapter_path)

            # Step 1: Merge LoRA adapter
            if progress_callback:
                progress_callback("Merging adapter", 0.0)

            merge_start = time.perf_counter()
            merged_path = self.merge_adapter(
                progress_callback=lambda s, p: progress_callback(f"Merge: {s}", p * 0.4) if progress_callback else None
            )
            result.merge_time_seconds = time.perf_counter() - merge_start
            result.merged_size_mb = self._get_dir_size_mb(merged_path)

            # Step 2: Convert to GGUF
            if progress_callback:
                progress_callback("Converting to GGUF", 0.4)

            convert_start = time.perf_counter()
            gguf_path = self.convert_to_gguf(
                merged_path,
                progress_callback=lambda s, p: progress_callback(f"Convert: {s}", 0.4 + p * 0.4) if progress_callback else None
            )
            result.convert_time_seconds = time.perf_counter() - convert_start
            result.gguf_size_mb = self._get_file_size_mb(gguf_path)

            # Calculate compression ratio
            if result.merged_size_mb > 0:
                result.compression_ratio = result.merged_size_mb / result.gguf_size_mb

            # Step 3: Verify (optional)
            if self.config.verify_after_export:
                if progress_callback:
                    progress_callback("Verifying GGUF", 0.8)

                result.verified, result.verification_error = self.verify_gguf(gguf_path)

                if not result.verified:
                    result.error = f"Verification failed: {result.verification_error}"
                    return result

            # Store metadata
            if metadata:
                result.metadata = metadata.to_dict()

            result.success = True
            result.output_path = gguf_path

            if progress_callback:
                progress_callback("Export complete", 1.0)

        except Exception as e:
            result.error = str(e)

        finally:
            # Cleanup temp directory
            if self._temp_dir and self._temp_dir.exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)

            result.total_time_seconds = time.perf_counter() - start_time

        return result

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
```

**Verification**: Unit test with mocked Unsloth

---

### Task 6: Integrate with CLI and pipeline

**Action**: Wire ExportEngine into CLI `export` command and pipeline Stage 7.

**File**: `src/mcp_forge/cli.py`

**Replace** export command (lines 1199-1205):
```python
@cli.command("export")
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to LoRA adapter")
@click.option("--format", "quant_format", type=click.Choice(["q8_0", "q4_k_m", "q4_k_s", "q5_k_m", "f16"]), default="q8_0", help="Quantization format")
@click.option("--output", "-o", required=True, help="Output GGUF file path")
@click.option("--base-model", help="Base model (auto-detected from adapter if not specified)")
@click.option("--no-verify", is_flag=True, help="Skip GGUF verification after export")
@click.pass_context
def export_model(
    ctx,
    model: str,
    quant_format: str,
    output: str,
    base_model: str | None,
    no_verify: bool,
):
    """Export LoRA adapter to GGUF format.

    Merges the LoRA adapter with the base model and converts to
    quantized GGUF format for deployment with llama.cpp or Ollama.

    \b
    Quantization Formats:
      q8_0   - 8-bit, best quality (default)
      q4_k_m - 4-bit k-quant, good balance
      q4_k_s - 4-bit k-quant small
      q5_k_m - 5-bit k-quant medium
      f16    - Half precision (no quantization)

    \b
    Example:
      mcp-forge export -m ./adapter -o model.gguf --format q4_k_m
    """
    from pathlib import Path

    from mcp_forge.export import ExportConfig, ExportEngine, QuantizationType
    from mcp_forge.export.metadata import GGUFMetadata

    console.print("\n[bold]MCP-Forge GGUF Export[/bold]")
    console.print("=" * 50)

    adapter_path = Path(model)
    output_path = Path(output)

    # Ensure output has .gguf extension
    if not output_path.suffix == ".gguf":
        output_path = output_path.with_suffix(".gguf")

    console.print(f"Adapter: {adapter_path}")
    console.print(f"Output: {output_path}")
    console.print(f"Quantization: {quant_format}")

    # Detect base model from adapter config if not specified
    detected_base = base_model
    if not detected_base:
        adapter_config = adapter_path / "adapter_config.json"
        if adapter_config.exists():
            import json
            with open(adapter_config) as f:
                config_data = json.load(f)
            detected_base = config_data.get("base_model_name_or_path", "")
            console.print(f"Detected base model: {detected_base}")

    if not detected_base:
        detected_base = "unsloth/DeepSeek-R1-Distill-Qwen-7B"
        console.print(f"[yellow]Using default base model: {detected_base}[/yellow]")

    # Create config
    config = ExportConfig(
        adapter_path=adapter_path,
        output_path=output_path,
        base_model=detected_base,
        quantization=QuantizationType(quant_format),
        verify_after_export=not no_verify,
    )

    # Build metadata
    state_manager: StateManager = ctx.obj.get("state_manager")
    metadata = GGUFMetadata(
        model_name=adapter_path.name,
        model_family=detected_base.split("/")[-1] if "/" in detected_base else detected_base,
        quantization_type=quant_format,
    )

    # If we have pipeline state, enrich metadata
    if state_manager:
        try:
            state = state_manager.load_state()
            if state:
                metadata.tool_names = [t.name for t in state.tools]
                metadata.tool_count = len(state.tools)
                if state.validation_result:
                    metadata.tool_accuracy = state.validation_result.tool_selection_accuracy
                    metadata.schema_conformance = state.validation_result.schema_conformance_rate
                if state.benchmark_result:
                    metadata.benchmark_score = state.benchmark_result.overall_score
        except Exception:
            pass  # No state available, continue without

    engine = ExportEngine(config)

    # Run export with progress
    console.print("\nStarting export...")

    def on_progress(stage: str, pct: float) -> None:
        console.print(f"  [{pct:.0%}] {stage}")

    try:
        result = engine.export(metadata=metadata, progress_callback=on_progress)
    except Exception as e:
        console.print(f"\n[red]Export failed: {e}[/red]")
        raise SystemExit(1) from e

    # Display results
    console.print("\n" + "=" * 50)

    if result.success:
        console.print("[bold green]Export successful![/bold green]")
        console.print(f"\nOutput: {result.output_path}")
        console.print(f"Size: {result.gguf_size_mb:.1f} MB")
        console.print(f"Compression: {result.compression_ratio:.1f}x")
        console.print(f"Time: {result.total_time_seconds:.1f}s")

        if result.verified:
            console.print("[green]Verification: PASSED[/green]")
        elif no_verify:
            console.print("[yellow]Verification: SKIPPED[/yellow]")

        # Save export report
        if state_manager:
            state_manager.ensure_dirs()
            import json
            report_path = state_manager.get_report_path("export_latest.json")
            with open(report_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            console.print(f"\nReport: {report_path}")
    else:
        console.print(f"[red]Export failed: {result.error}[/red]")
        raise SystemExit(1)
```

**Also update** pipeline Stage 7 (cli.py:380-388):
```python
    # Stage 7: Export
    if state.stage in (PipelineStage.VALIDATING, PipelineStage.BENCHMARKING, PipelineStage.EXPORTING):
        console.print("\n[bold]Stage 7: Exporting GGUF...[/bold]")
        state.update_stage(PipelineStage.EXPORTING)
        state_manager.save_state(state)

        from mcp_forge.export import ExportConfig, ExportEngine, QuantizationType
        from mcp_forge.export.metadata import GGUFMetadata

        # Determine output path
        output_dir = Path(state.output_path)
        gguf_path = output_dir / f"{state.model_family}.{state.quantization}.gguf"

        console.print(f"   Format: {state.quantization}")
        console.print(f"   Output: {gguf_path}")

        config = ExportConfig(
            adapter_path=Path(state.lora_adapter_path),
            output_path=gguf_path,
            base_model=state.model_family,
            quantization=QuantizationType(state.quantization),
        )

        # Build metadata from pipeline state
        metadata = GGUFMetadata(
            model_name=f"mcp-forge-{state.model_family}",
            model_family=state.model_family,
            tool_names=[t.name for t in state.tools],
            training_timestamp=state.created_at,
        )
        if state.validation_result:
            metadata.tool_accuracy = state.validation_result.tool_selection_accuracy
            metadata.schema_conformance = state.validation_result.schema_conformance_rate
        if state.benchmark_result:
            metadata.benchmark_score = state.benchmark_result.overall_score

        engine = ExportEngine(config)

        def on_progress(stage: str, pct: float) -> None:
            console.print(f"   [{pct:.0%}] {stage}")

        result = engine.export(metadata=metadata, progress_callback=on_progress)

        if result.success:
            state.gguf_path = str(result.output_path)
            state_manager.save_state(state)

            console.print(f"   Size: {result.gguf_size_mb:.1f} MB")
            console.print(f"   Compression: {result.compression_ratio:.1f}x")
            console.print("   [green]Export complete[/green]")
        else:
            console.print(f"   [red]Export failed: {result.error}[/red]")
            state.error = result.error
            state.update_stage(PipelineStage.FAILED)
            state_manager.save_state(state)
            raise SystemExit(1)
```

**Verification**: CLI smoke test with `mcp-forge export --help`

---

### Task 7: Add module exports and tests

**Action**: Create `__init__.py` and test files.

**File**: `src/mcp_forge/export/__init__.py`

**Implementation**:
```python
"""GGUF export module for MCP-Forge.

Provides conversion of LoRA-tuned models to GGUF format for deployment
with llama.cpp, Ollama, and other inference engines.
"""

from mcp_forge.export.config import (
    ExportConfig,
    ExportResult,
    QuantizationType,
)
from mcp_forge.export.engine import ExportEngine
from mcp_forge.export.metadata import (
    GGUFMetadata,
    read_gguf_metadata,
)

__all__ = [
    # Config
    "ExportConfig",
    "ExportResult",
    "QuantizationType",
    # Engine
    "ExportEngine",
    # Metadata
    "GGUFMetadata",
    "read_gguf_metadata",
]
```

**Test Files**:
- `tests/unit/test_export_config.py` - Config validation
- `tests/unit/test_export_metadata.py` - Metadata serialization
- `tests/unit/test_export_engine.py` - Engine logic with mocked Unsloth
- `tests/integration/test_export_pipeline.py` - Pipeline integration

**Test Cases**:

```python
# test_export_config.py
def test_config_requires_existing_adapter():
    """Config raises if adapter path doesn't exist."""

def test_config_creates_output_dir():
    """Config creates output directory if needed."""

def test_quantization_enum_values():
    """QuantizationType has expected values."""

def test_export_result_serialization():
    """ExportResult round-trips through to_dict/from_dict."""

# test_export_metadata.py
def test_metadata_to_gguf_kv():
    """Metadata converts to GGUF key-value format."""

def test_metadata_sets_tool_count():
    """Metadata auto-calculates tool_count from tool_names."""

def test_metadata_serialization():
    """GGUFMetadata round-trips through to_dict/from_dict."""

# test_export_engine.py
@pytest.fixture
def mock_unsloth_export(monkeypatch):
    """Mock Unsloth for export tests without GPU."""

def test_engine_get_unsloth_quant_method():
    """Engine maps quantization types correctly."""

def test_engine_verify_gguf_invalid_file(tmp_path):
    """Verification fails for invalid GGUF."""

def test_engine_cleanup_temp_dir(mock_unsloth_export, tmp_path):
    """Engine cleans up temporary files."""

def test_engine_export_result_metrics(mock_unsloth_export):
    """Export result includes size and timing metrics."""

# test_export_pipeline.py
@pytest.mark.integration
def test_pipeline_export_stage(mock_unsloth_export, state_manager):
    """Export stage integrates with pipeline state."""

@pytest.mark.integration
def test_export_report_generation(mock_unsloth_export, state_manager):
    """Export generates JSON report."""
```

**Fixtures**:
```python
@pytest.fixture
def mock_unsloth_export(monkeypatch, tmp_path):
    """Mock Unsloth for export tests."""

    class MockModel:
        def merge_and_unload(self):
            return self

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").touch()

        def save_pretrained_gguf(self, path, tokenizer, quantization_method):
            output = Path(path) / f"unsloth.{quantization_method.upper()}.gguf"
            output.write_bytes(b"GGUF" + b"\x00" * 1000)  # Minimal fake GGUF

    class MockTokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").touch()

    def mock_from_pretrained(*args, **kwargs):
        return MockModel(), MockTokenizer()

    monkeypatch.setattr(
        "unsloth.FastLanguageModel.from_pretrained",
        mock_from_pretrained
    )

@pytest.fixture
def sample_adapter_path(tmp_path):
    """Create sample adapter directory."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "test/model"}')
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 100)
    return adapter
```

**Verification**: `pytest tests/unit/test_export*.py tests/integration/test_export*.py -v`

---

## Verification

After all tasks complete:

```bash
# 1. Install updated dependencies
pip install -e ".[dev]"

# 2. Lint and type check
ruff check src/mcp_forge/export/
mypy src/mcp_forge/export/

# 3. Run export tests
pytest tests/unit/test_export*.py -v
pytest tests/integration/test_export*.py -v

# 4. Check coverage
pytest --cov=src/mcp_forge/export --cov-report=term-missing

# 5. CLI smoke tests
mcp-forge export --help
mcp-forge export -m ./test-adapter -o test.gguf --format q8_0 --no-verify

# 6. Verify imports
python -c "from mcp_forge.export import ExportEngine, ExportConfig, QuantizationType; print('OK')"
```

---

## Success Criteria

- [ ] `llama-cpp-python` added to dependencies in pyproject.toml
- [ ] `src/mcp_forge/export/` module created with 4 files
- [ ] ExportConfig supports all quantization types (Q8_0, Q4_K_M, Q4_K_S, Q5_K_M, F16)
- [ ] ExportResult tracks size metrics (adapter, merged, gguf, compression ratio)
- [ ] ExportResult tracks timing (merge, convert, total)
- [ ] GGUFMetadata embeds MCP-Forge custom fields in GGUF format
- [ ] ExportEngine merges LoRA adapter with base model
- [ ] ExportEngine converts to GGUF with specified quantization
- [ ] ExportEngine verifies GGUF loads correctly (optional)
- [ ] CLI `export` command fully functional with all options
- [ ] Pipeline stage 7 transitions correctly (BENCHMARKING → EXPORTING)
- [ ] Export report generated and saved to .mcp-forge/reports/
- [ ] All tests pass with mocked Unsloth (no GPU required)
- [ ] Coverage maintained at 85%+
- [ ] ruff and mypy pass on new code

---

## Output

| Artifact | Path |
|----------|------|
| Export module | `src/mcp_forge/export/` |
| Config file | `src/mcp_forge/export/config.py` |
| Metadata file | `src/mcp_forge/export/metadata.py` |
| Engine file | `src/mcp_forge/export/engine.py` |
| Config tests | `tests/unit/test_export_config.py` |
| Metadata tests | `tests/unit/test_export_metadata.py` |
| Engine tests | `tests/unit/test_export_engine.py` |
| Integration tests | `tests/integration/test_export_pipeline.py` |
| Export reports (runtime) | `.mcp-forge/reports/export_*.json` |
| GGUF output (runtime) | `<output_path>/*.gguf` |

---

## Notes

### GGUF Export Strategy

This implementation uses Unsloth's built-in `save_pretrained_gguf()` method for conversion, which:
1. Handles model architecture detection automatically
2. Supports multiple quantization methods
3. Produces llama.cpp-compatible GGUF files

Alternative approaches considered but deferred:
- Direct llama.cpp python bindings (more control, more complexity)
- ctransformers conversion (less maintained)
- Manual safetensors→GGUF conversion (error-prone)

### Metadata Embedding

GGUF supports custom metadata key-value pairs. We namespace MCP-Forge fields under `mcp_forge.*` to avoid conflicts with standard fields. This metadata can be read by:
- llama.cpp `--info` flag
- llama-cpp-python `Llama.metadata` property
- Ollama model inspection

### Verification

The verification step loads the GGUF with minimal context to ensure:
1. File format is valid
2. Vocabulary loaded correctly
3. No obvious corruption

Full inference verification is optional and requires more resources.

### GPU Requirements

- Merge step requires CUDA GPU (same as training)
- Conversion step requires CUDA GPU
- Verification can run on CPU
- Tests use mocked Unsloth to run without GPU

---

*Plan created: 2026-01-13*
*Execute with: /gsd:execute-plan*
