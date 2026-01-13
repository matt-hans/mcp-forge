"""Tests for ExportEngine with mocked Unsloth."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_forge.export.config import ExportConfig, QuantizationType
from mcp_forge.export.engine import ExportEngine
from mcp_forge.export.metadata import GGUFMetadata


@pytest.fixture
def sample_adapter_path(tmp_path: Path) -> Path:
    """Create sample adapter directory."""
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "test/model"}')
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
            (Path(path) / "model.safetensors").touch()

        def save_pretrained_gguf(
            self,
            path: str,
            tokenizer: "MockTokenizer",
            quantization_method: str,
        ) -> None:
            output = Path(path) / f"unsloth.{quantization_method.upper()}.gguf"
            output.write_bytes(b"GGUF" + b"\x00" * 1000)  # Minimal fake GGUF

    class MockTokenizer:
        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.json").touch()

    def mock_from_pretrained(*args, **kwargs):  # type: ignore[no-untyped-def]
        return MockModel(), MockTokenizer()

    # Mock the unsloth module
    mock_unsloth = MagicMock()
    mock_unsloth.FastLanguageModel.from_pretrained = mock_from_pretrained
    monkeypatch.setitem(  # type: ignore[attr-defined]
        __import__("sys").modules, "unsloth", mock_unsloth
    )


class TestExportEngine:
    """Tests for ExportEngine class."""

    def test_engine_get_unsloth_quant_method(
        self, sample_adapter_path: Path, tmp_path: Path
    ) -> None:
        """Engine maps quantization types correctly."""
        for quant_type, expected in [
            (QuantizationType.Q8_0, "q8_0"),
            (QuantizationType.Q4_K_M, "q4_k_m"),
            (QuantizationType.Q4_K_S, "q4_k_s"),
            (QuantizationType.Q5_K_M, "q5_k_m"),
            (QuantizationType.F16, "f16"),
        ]:
            config = ExportConfig(
                adapter_path=sample_adapter_path,
                output_path=tmp_path / "output.gguf",
                base_model="test/model",
                quantization=quant_type,
            )
            engine = ExportEngine(config)
            assert engine._get_unsloth_quant_method() == expected

    def test_engine_verify_gguf_invalid_file(
        self, sample_adapter_path: Path, tmp_path: Path
    ) -> None:
        """Verification fails for invalid GGUF."""
        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
        )
        engine = ExportEngine(config)

        # Create an invalid GGUF file
        invalid_gguf = tmp_path / "invalid.gguf"
        invalid_gguf.write_bytes(b"not a valid gguf file")

        success, error = engine.verify_gguf(invalid_gguf)
        assert success is False
        assert error is not None

    def test_engine_verify_gguf_nonexistent(
        self, sample_adapter_path: Path, tmp_path: Path
    ) -> None:
        """Verification fails for nonexistent file."""
        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
        )
        engine = ExportEngine(config)

        success, error = engine.verify_gguf(tmp_path / "nonexistent.gguf")
        assert success is False
        assert error is not None

    def test_engine_cleanup_temp_dir(
        self,
        mock_unsloth_export: None,
        sample_adapter_path: Path,
        tmp_path: Path,
    ) -> None:
        """Engine cleans up temporary files."""
        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
            verify_after_export=False,  # Skip verification
        )
        engine = ExportEngine(config)

        # Simulate setting temp dir
        engine._temp_dir = tmp_path / "temp_export"
        engine._temp_dir.mkdir()
        (engine._temp_dir / "file.txt").touch()

        engine.cleanup()

        assert engine._temp_dir is None

    def test_engine_dir_size_calculation(
        self, sample_adapter_path: Path, tmp_path: Path
    ) -> None:
        """Engine calculates directory size correctly."""
        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
        )
        engine = ExportEngine(config)

        # Create test directory with known sizes
        test_dir = tmp_path / "test_size"
        test_dir.mkdir()
        (test_dir / "file1.bin").write_bytes(b"\x00" * 1024 * 1024)  # 1MB
        (test_dir / "file2.bin").write_bytes(b"\x00" * 512 * 1024)  # 0.5MB

        size = engine._get_dir_size_mb(test_dir)
        assert 1.4 <= size <= 1.6  # ~1.5MB

    def test_engine_file_size_calculation(
        self, sample_adapter_path: Path, tmp_path: Path
    ) -> None:
        """Engine calculates file size correctly."""
        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
        )
        engine = ExportEngine(config)

        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00" * 1024 * 1024)  # 1MB

        size = engine._get_file_size_mb(test_file)
        assert 0.99 <= size <= 1.01

    def test_engine_file_size_nonexistent(
        self, sample_adapter_path: Path, tmp_path: Path
    ) -> None:
        """Engine returns 0 for nonexistent file."""
        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
        )
        engine = ExportEngine(config)

        size = engine._get_file_size_mb(tmp_path / "nonexistent.bin")
        assert size == 0.0


class TestExportEngineWithMock:
    """Tests requiring mocked Unsloth."""

    def test_engine_export_creates_result(
        self,
        mock_unsloth_export: None,
        sample_adapter_path: Path,
        tmp_path: Path,
    ) -> None:
        """Export creates result with metrics."""
        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
            verify_after_export=False,
        )
        engine = ExportEngine(config)

        metadata = GGUFMetadata(
            model_name="test-model",
            model_family="test-family",
        )

        result = engine.export(metadata=metadata)

        assert result.total_time_seconds > 0
        assert result.metadata["model_name"] == "test-model"

    def test_engine_export_with_progress_callback(
        self,
        mock_unsloth_export: None,
        sample_adapter_path: Path,
        tmp_path: Path,
    ) -> None:
        """Export calls progress callback."""
        config = ExportConfig(
            adapter_path=sample_adapter_path,
            output_path=tmp_path / "output.gguf",
            base_model="test/model",
            verify_after_export=False,
        )
        engine = ExportEngine(config)

        progress_calls: list[tuple[str, float]] = []

        def on_progress(stage: str, pct: float) -> None:
            progress_calls.append((stage, pct))

        engine.export(progress_callback=on_progress)

        assert len(progress_calls) > 0
        # Should have at least the initial and final callbacks
        stages = [call[0] for call in progress_calls]
        assert any("Merge" in s or "adapter" in s.lower() for s in stages)

    def test_engine_base_models_mapping(self) -> None:
        """Engine has correct base model mappings."""
        assert "deepseek-r1" in ExportEngine.BASE_MODELS
        assert "qwen-2.5" in ExportEngine.BASE_MODELS
        assert "unsloth" in ExportEngine.BASE_MODELS["deepseek-r1"]
