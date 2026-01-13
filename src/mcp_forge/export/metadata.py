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
