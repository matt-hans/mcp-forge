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
                progress_callback=lambda s, p: (
                    progress_callback(f"Merge: {s}", p * 0.4) if progress_callback else None
                )
            )
            result.merge_time_seconds = time.perf_counter() - merge_start
            result.merged_size_mb = self._get_dir_size_mb(merged_path)

            # Step 2: Convert to GGUF
            if progress_callback:
                progress_callback("Converting to GGUF", 0.4)

            convert_start = time.perf_counter()
            gguf_path = self.convert_to_gguf(
                merged_path,
                progress_callback=lambda s, p: (
                    progress_callback(f"Convert: {s}", 0.4 + p * 0.4) if progress_callback else None
                )
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
