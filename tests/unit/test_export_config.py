"""Tests for export configuration classes."""

from pathlib import Path

import pytest

from mcp_forge.export.config import ExportConfig, ExportResult, QuantizationType


class TestQuantizationType:
    """Tests for QuantizationType enum."""

    def test_quantization_enum_values(self) -> None:
        """QuantizationType has expected values."""
        assert QuantizationType.Q8_0.value == "q8_0"
        assert QuantizationType.Q4_K_M.value == "q4_k_m"
        assert QuantizationType.Q4_K_S.value == "q4_k_s"
        assert QuantizationType.Q5_K_M.value == "q5_k_m"
        assert QuantizationType.F16.value == "f16"

    def test_quantization_from_string(self) -> None:
        """QuantizationType can be created from string."""
        assert QuantizationType("q8_0") == QuantizationType.Q8_0
        assert QuantizationType("q4_k_m") == QuantizationType.Q4_K_M


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_config_requires_existing_adapter(self, tmp_path: Path) -> None:
        """Config raises if adapter path doesn't exist."""
        with pytest.raises(ValueError, match="does not exist"):
            ExportConfig(
                adapter_path=tmp_path / "nonexistent",
                output_path=tmp_path / "output.gguf",
                base_model="test/model",
            )

    def test_config_creates_output_dir(self, tmp_path: Path) -> None:
        """Config creates output directory if needed."""
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")

        output_dir = tmp_path / "nested" / "output" / "model.gguf"
        config = ExportConfig(
            adapter_path=adapter,
            output_path=output_dir,
            base_model="test/model",
        )

        assert config.output_path.parent.exists()

    def test_config_sets_default_model_name(self, tmp_path: Path) -> None:
        """Config sets default model name from adapter path."""
        adapter = tmp_path / "my-adapter"
        adapter.mkdir()

        config = ExportConfig(
            adapter_path=adapter,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
        )

        assert config.model_name == "my-adapter"

    def test_config_preserves_explicit_model_name(self, tmp_path: Path) -> None:
        """Config preserves explicitly set model name."""
        adapter = tmp_path / "adapter"
        adapter.mkdir()

        config = ExportConfig(
            adapter_path=adapter,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
            model_name="custom-name",
        )

        assert config.model_name == "custom-name"

    def test_config_sets_training_timestamp(self, tmp_path: Path) -> None:
        """Config sets training timestamp if not provided."""
        adapter = tmp_path / "adapter"
        adapter.mkdir()

        config = ExportConfig(
            adapter_path=adapter,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
        )

        assert config.training_timestamp != ""
        # Should be ISO format
        assert "T" in config.training_timestamp

    def test_config_default_quantization(self, tmp_path: Path) -> None:
        """Config defaults to Q8_0 quantization."""
        adapter = tmp_path / "adapter"
        adapter.mkdir()

        config = ExportConfig(
            adapter_path=adapter,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
        )

        assert config.quantization == QuantizationType.Q8_0


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_export_result_serialization(self, tmp_path: Path) -> None:
        """ExportResult round-trips through to_dict/from_dict."""
        result = ExportResult(
            success=True,
            output_path=tmp_path / "model.gguf",
            adapter_size_mb=100.0,
            merged_size_mb=500.0,
            gguf_size_mb=250.0,
            compression_ratio=2.0,
            merge_time_seconds=30.0,
            convert_time_seconds=60.0,
            total_time_seconds=95.0,
            verified=True,
            verification_error=None,
            metadata={"key": "value"},
            error=None,
        )

        data = result.to_dict()
        restored = ExportResult.from_dict(data)

        assert restored.success == result.success
        assert restored.output_path == result.output_path
        assert restored.adapter_size_mb == result.adapter_size_mb
        assert restored.merged_size_mb == result.merged_size_mb
        assert restored.gguf_size_mb == result.gguf_size_mb
        assert restored.compression_ratio == result.compression_ratio
        assert restored.merge_time_seconds == result.merge_time_seconds
        assert restored.convert_time_seconds == result.convert_time_seconds
        assert restored.total_time_seconds == result.total_time_seconds
        assert restored.verified == result.verified
        assert restored.metadata == result.metadata

    def test_export_result_failed(self) -> None:
        """ExportResult handles failure case."""
        result = ExportResult(
            success=False,
            output_path=None,
            error="Something went wrong",
        )

        data = result.to_dict()
        assert data["success"] is False
        assert data["output_path"] is None
        assert data["error"] == "Something went wrong"

    def test_export_result_from_dict_with_defaults(self) -> None:
        """ExportResult from_dict handles missing optional fields."""
        data = {"success": True, "output_path": "/path/to/model.gguf"}
        result = ExportResult.from_dict(data)

        assert result.success is True
        assert result.adapter_size_mb == 0.0
        assert result.verified is False
        assert result.metadata == {}
