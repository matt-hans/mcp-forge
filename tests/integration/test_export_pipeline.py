"""Integration tests for export pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mcp_forge.export import ExportConfig, ExportEngine, QuantizationType
from mcp_forge.export.metadata import GGUFMetadata


@pytest.fixture
def sample_adapter_path(tmp_path: Path) -> Path:
    """Create sample adapter directory."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "unsloth/test-model"}'
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 100)
    return adapter


@pytest.fixture
def mock_unsloth_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Mock Unsloth for export tests without GPU."""

    class MockModel:
        def merge_and_unload(self) -> "MockModel":
            return self

        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "model.safetensors").write_bytes(b"\x00" * 1000)

        def save_pretrained_gguf(
            self,
            path: str,
            tokenizer: "MockTokenizer",
            quantization_method: str,
        ) -> None:
            output = Path(path) / f"unsloth.{quantization_method.upper()}.gguf"
            output.write_bytes(b"GGUF" + b"\x00" * 5000)

    class MockTokenizer:
        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").write_text("{}")

    def mock_from_pretrained(*args, **kwargs):  # type: ignore[no-untyped-def]
        return MockModel(), MockTokenizer()

    mock_unsloth = MagicMock()
    mock_unsloth.FastLanguageModel.from_pretrained = mock_from_pretrained
    monkeypatch.setitem(  # type: ignore[attr-defined]
        __import__("sys").modules, "unsloth", mock_unsloth
    )


@pytest.mark.integration
class TestExportPipeline:
    """Integration tests for export pipeline."""

    def test_full_export_flow(
        self,
        mock_unsloth_export: None,
        sample_adapter_path: Path,
        tmp_path: Path,
    ) -> None:
        """Full export flow produces expected artifacts."""
        output_gguf = tmp_path / "output" / "model.gguf"

        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=output_gguf,
            base_model="test/model",
            quantization=QuantizationType.Q4_K_M,
            verify_after_export=False,
        )

        metadata = GGUFMetadata(
            model_name="integration-test-model",
            model_family="test-family",
            tool_names=["tool1", "tool2", "tool3"],
            tool_accuracy=0.95,
            schema_conformance=0.98,
        )

        engine = ExportEngine(config)
        result = engine.export(metadata=metadata)

        # Verify result structure
        assert result.total_time_seconds > 0
        assert result.metadata["model_name"] == "integration-test-model"
        assert result.metadata["tool_names"] == ["tool1", "tool2", "tool3"]
        assert result.metadata["tool_accuracy"] == 0.95

    def test_export_with_all_quantization_types(
        self,
        mock_unsloth_export: None,
        sample_adapter_path: Path,
        tmp_path: Path,
    ) -> None:
        """Export works with all quantization types."""
        for quant_type in QuantizationType:
            output = tmp_path / f"model_{quant_type.value}.gguf"

            config = ExportConfig(
                adapter_path=sample_adapter_path,
                output_path=output,
                base_model="test/model",
                quantization=quant_type,
                verify_after_export=False,
            )

            engine = ExportEngine(config)
            result = engine.export()

            # Basic validation - export completes
            assert result.total_time_seconds >= 0

    def test_export_report_generation(
        self,
        mock_unsloth_export: None,
        sample_adapter_path: Path,
        tmp_path: Path,
    ) -> None:
        """Export generates JSON report."""
        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=tmp_path / "model.gguf",
            base_model="test/model",
            verify_after_export=False,
        )

        metadata = GGUFMetadata(
            model_name="report-test",
            model_family="test",
        )

        engine = ExportEngine(config)
        result = engine.export(metadata=metadata)

        # Serialize to JSON report
        report_path = tmp_path / "export_report.json"
        with open(report_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Verify report is valid JSON
        with open(report_path) as f:
            loaded_report = json.load(f)

        assert "total_time_seconds" in loaded_report
        assert "metadata" in loaded_report
        assert loaded_report["metadata"]["model_name"] == "report-test"

    def test_export_handles_error_gracefully(
        self,
        sample_adapter_path: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Export handles errors gracefully."""
        # Mock unsloth to raise an error
        def failing_from_pretrained(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("Simulated GPU error")

        mock_unsloth = MagicMock()
        mock_unsloth.FastLanguageModel.from_pretrained = failing_from_pretrained
        monkeypatch.setitem(  # type: ignore[attr-defined]
            __import__("sys").modules, "unsloth", mock_unsloth
        )

        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=tmp_path / "model.gguf",
            base_model="test/model",
            verify_after_export=False,
        )

        engine = ExportEngine(config)
        result = engine.export()

        assert result.success is False
        assert result.error is not None
        assert "Simulated GPU error" in result.error

    def test_export_metadata_integration(
        self,
        mock_unsloth_export: None,
        sample_adapter_path: Path,
        tmp_path: Path,
    ) -> None:
        """Export integrates metadata correctly."""
        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=tmp_path / "model.gguf",
            base_model="test/model",
            verify_after_export=False,
        )

        # Create metadata with all fields
        metadata = GGUFMetadata(
            model_name="full-metadata-test",
            model_family="deepseek-r1",
            forge_version="0.1.0",
            tool_names=["search", "get_weather", "calculate"],
            training_timestamp="2026-01-13T10:00:00Z",
            training_samples=5000,
            training_epochs=3,
            base_model="unsloth/DeepSeek-R1-Distill-Qwen-7B",
            tool_accuracy=0.92,
            schema_conformance=0.96,
            benchmark_score=0.89,
            quantization_type="q4_k_m",
        )

        engine = ExportEngine(config)
        result = engine.export(metadata=metadata)

        # Verify all metadata fields are preserved
        assert result.metadata["model_name"] == "full-metadata-test"
        assert result.metadata["model_family"] == "deepseek-r1"
        assert result.metadata["tool_count"] == 3
        assert result.metadata["training_samples"] == 5000
        assert result.metadata["tool_accuracy"] == 0.92
        assert result.metadata["benchmark_score"] == 0.89


@pytest.mark.integration
class TestExportModuleImports:
    """Test module exports are accessible."""

    def test_import_export_engine(self) -> None:
        """ExportEngine can be imported from module."""
        from mcp_forge.export import ExportEngine

        assert ExportEngine is not None

    def test_import_export_config(self) -> None:
        """ExportConfig can be imported from module."""
        from mcp_forge.export import ExportConfig

        assert ExportConfig is not None

    def test_import_quantization_type(self) -> None:
        """QuantizationType can be imported from module."""
        from mcp_forge.export import QuantizationType

        assert QuantizationType is not None
        assert QuantizationType.Q8_0.value == "q8_0"

    def test_import_export_result(self) -> None:
        """ExportResult can be imported from module."""
        from mcp_forge.export import ExportResult

        assert ExportResult is not None

    def test_import_gguf_metadata(self) -> None:
        """GGUFMetadata can be imported from module."""
        from mcp_forge.export import GGUFMetadata

        assert GGUFMetadata is not None

    def test_import_read_gguf_metadata(self) -> None:
        """read_gguf_metadata can be imported from module."""
        from mcp_forge.export import read_gguf_metadata

        assert callable(read_gguf_metadata)
