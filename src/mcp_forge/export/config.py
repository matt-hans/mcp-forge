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
