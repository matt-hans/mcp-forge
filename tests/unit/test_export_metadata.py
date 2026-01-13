"""Tests for GGUF metadata handling."""

import pytest

from mcp_forge.export.metadata import GGUFMetadata


class TestGGUFMetadata:
    """Tests for GGUFMetadata dataclass."""

    def test_metadata_to_gguf_kv(self) -> None:
        """Metadata converts to GGUF key-value format."""
        metadata = GGUFMetadata(
            model_name="test-model",
            model_family="deepseek-r1",
            tool_names=["get_weather", "search"],
            tool_accuracy=0.95,
            schema_conformance=0.98,
        )

        kv = metadata.to_gguf_kv()

        # Standard fields
        assert kv["general.name"] == "test-model"
        assert kv["general.architecture"] == "deepseek-r1"
        assert kv["general.quantization_version"] == 2

        # MCP-Forge custom fields
        assert kv["mcp_forge.version"] == "0.1.0"
        assert kv["mcp_forge.tool_count"] == 2
        assert kv["mcp_forge.tool_names"] == "get_weather,search"
        assert kv["mcp_forge.tool_accuracy"] == 0.95
        assert kv["mcp_forge.schema_conformance"] == 0.98

    def test_metadata_sets_tool_count(self) -> None:
        """Metadata auto-calculates tool_count from tool_names."""
        metadata = GGUFMetadata(
            model_name="test",
            model_family="qwen",
            tool_names=["a", "b", "c"],
        )

        assert metadata.tool_count == 3

    def test_metadata_preserves_explicit_tool_count(self) -> None:
        """Metadata preserves explicitly set tool_count."""
        metadata = GGUFMetadata(
            model_name="test",
            model_family="qwen",
            tool_names=["a", "b"],
            tool_count=5,  # Explicitly different
        )

        # Explicit value is preserved (post_init only sets if 0)
        assert metadata.tool_count == 5

    def test_metadata_serialization(self) -> None:
        """GGUFMetadata round-trips through to_dict/from_dict."""
        original = GGUFMetadata(
            model_name="my-model",
            model_family="deepseek",
            forge_version="1.0.0",
            tool_names=["tool1", "tool2"],
            tool_count=2,
            training_timestamp="2026-01-13T12:00:00Z",
            training_samples=1000,
            training_epochs=3,
            base_model="unsloth/model",
            tool_accuracy=0.9,
            schema_conformance=0.95,
            benchmark_score=0.88,
            export_timestamp="2026-01-13T14:00:00Z",
            quantization_type="q4_k_m",
        )

        data = original.to_dict()
        restored = GGUFMetadata.from_dict(data)

        assert restored.model_name == original.model_name
        assert restored.model_family == original.model_family
        assert restored.forge_version == original.forge_version
        assert restored.tool_names == original.tool_names
        assert restored.tool_count == original.tool_count
        assert restored.training_timestamp == original.training_timestamp
        assert restored.training_samples == original.training_samples
        assert restored.training_epochs == original.training_epochs
        assert restored.base_model == original.base_model
        assert restored.tool_accuracy == original.tool_accuracy
        assert restored.schema_conformance == original.schema_conformance
        assert restored.benchmark_score == original.benchmark_score
        assert restored.export_timestamp == original.export_timestamp
        assert restored.quantization_type == original.quantization_type

    def test_metadata_from_dict_with_defaults(self) -> None:
        """GGUFMetadata from_dict handles missing optional fields."""
        data = {"model_name": "test", "model_family": "qwen"}
        metadata = GGUFMetadata.from_dict(data)

        assert metadata.model_name == "test"
        assert metadata.model_family == "qwen"
        assert metadata.forge_version == "0.1.0"
        assert metadata.tool_names == []
        assert metadata.tool_count == 0
        assert metadata.tool_accuracy == 0.0

    def test_metadata_sets_export_timestamp(self) -> None:
        """Metadata auto-sets export timestamp."""
        metadata = GGUFMetadata(
            model_name="test",
            model_family="qwen",
        )

        assert metadata.export_timestamp != ""
        assert "T" in metadata.export_timestamp  # ISO format

    def test_metadata_gguf_kv_with_empty_tools(self) -> None:
        """Metadata handles empty tool list in GGUF KV."""
        metadata = GGUFMetadata(
            model_name="test",
            model_family="qwen",
            tool_names=[],
        )

        kv = metadata.to_gguf_kv()
        assert kv["mcp_forge.tool_names"] == ""
        assert kv["mcp_forge.tool_count"] == 0
