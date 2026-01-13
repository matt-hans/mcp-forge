"""Bundle packaging for MCP-Forge agent bundles.

Creates distributable agent bundles containing:
- GGUF model file
- Tool definitions (tools.json)
- Deployment configuration (config.yaml)
- README documentation
- Ollama Modelfile
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from mcp_forge.state import ToolDefinition

# Package version for bundle metadata
try:
    from mcp_forge import __version__
except ImportError:
    __version__ = "unknown"


@dataclass
class BundleConfig:
    """Configuration for bundle packaging."""

    gguf_path: Path  # Input GGUF model
    tools: list[ToolDefinition]  # Tool definitions from state
    output_dir: Path  # Output bundle directory

    # Optional file paths for enrichment
    tools_path: Path | None = None  # Optional tools.json (or use ToolDefinition list)
    validation_report_path: Path | None = None
    benchmark_report_path: Path | None = None

    # Bundle options
    include_modelfile: bool = True  # Generate Ollama Modelfile
    include_readme: bool = True  # Generate README.md
    model_name: str = ""  # Human-readable name
    model_description: str = ""  # Short description

    # Deployment defaults
    default_temperature: float = 0.3
    default_context_size: int = 8192

    # Quality metrics (from validation/benchmark)
    tool_accuracy: float | None = None
    schema_conformance: float | None = None
    benchmark_score: float | None = None

    # Model metadata
    model_family: str = "unknown"
    training_samples: int = 0

    def __post_init__(self) -> None:
        """Validate config after initialization."""
        if not self.model_name:
            # Derive from GGUF path
            self.model_name = self.gguf_path.stem.replace(".", "-")


@dataclass
class BundleResult:
    """Result of bundle packaging."""

    success: bool
    bundle_path: Path | None

    # Contents
    files_created: list[str] = field(default_factory=list)
    bundle_size_mb: float = 0.0

    # Validation
    validation_passed: bool = False
    validation_errors: list[str] = field(default_factory=list)

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "bundle_path": str(self.bundle_path) if self.bundle_path else None,
            "files_created": self.files_created,
            "bundle_size_mb": self.bundle_size_mb,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "error": self.error,
        }


class BundleEngine:
    """Assembles distributable agent bundles."""

    # Required files in a valid bundle
    REQUIRED_FILES = ["model.gguf", "tools.json", "config.yaml"]

    def __init__(self, config: BundleConfig):
        """Initialize bundle engine.

        Args:
            config: Bundle configuration
        """
        self.config = config

    def package(
        self,
        progress_callback: callable | None = None,
    ) -> BundleResult:
        """Create the complete bundle.

        Args:
            progress_callback: Optional callback for progress updates.
                               Signature: (message: str) -> None

        Returns:
            BundleResult with success status and details
        """
        files_created: list[str] = []
        errors: list[str] = []

        def report(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)

        try:
            # 1. Create output directory
            report("Creating bundle directory...")
            self.config.output_dir.mkdir(parents=True, exist_ok=True)

            # 2. Copy GGUF model
            report("Copying model file...")
            model_path = self._copy_model()
            files_created.append(model_path.name)

            # 3. Generate tools.json
            report("Generating tools.json...")
            tools_path = self._generate_tools_json()
            files_created.append(tools_path.name)

            # 4. Generate config.yaml
            report("Generating config.yaml...")
            config_path = self._generate_config_yaml()
            files_created.append(config_path.name)

            # 5. Generate README.md (if enabled)
            if self.config.include_readme:
                report("Generating README.md...")
                readme_path = self._generate_readme()
                files_created.append(readme_path.name)

            # 6. Generate Modelfile (if enabled)
            if self.config.include_modelfile:
                report("Generating Modelfile...")
                modelfile_path = self._generate_modelfile()
                files_created.append(modelfile_path.name)

            # 7. Validate bundle
            report("Validating bundle...")
            valid, validation_errors = self._validate_bundle()
            errors.extend(validation_errors)

            # 8. Calculate bundle size
            bundle_size = sum(
                f.stat().st_size for f in self.config.output_dir.iterdir() if f.is_file()
            )
            bundle_size_mb = bundle_size / (1024 * 1024)

            report(f"Bundle created: {bundle_size_mb:.1f} MB")

            return BundleResult(
                success=valid and not errors,
                bundle_path=self.config.output_dir,
                files_created=files_created,
                bundle_size_mb=bundle_size_mb,
                validation_passed=valid,
                validation_errors=errors,
            )

        except Exception as e:
            return BundleResult(
                success=False,
                bundle_path=None,
                files_created=files_created,
                validation_errors=errors,
                error=str(e),
            )

    def _copy_model(self) -> Path:
        """Copy GGUF to bundle directory.

        Returns:
            Path to the copied model file
        """
        dest = self.config.output_dir / "model.gguf"
        shutil.copy2(self.config.gguf_path, dest)
        return dest

    def _generate_tools_json(self) -> Path:
        """Create tools.json with schema definitions.

        Returns:
            Path to the generated tools.json
        """
        tools_data = {
            "version": "1.0",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in self.config.tools
            ],
            "metadata": {
                "source": "mcp-forge",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "tool_count": len(self.config.tools),
            },
        }

        dest = self.config.output_dir / "tools.json"
        with open(dest, "w") as f:
            json.dump(tools_data, f, indent=2)

        return dest

    def _generate_config_yaml(self) -> Path:
        """Create deployment config.yaml.

        Returns:
            Path to the generated config.yaml
        """
        config_data = {
            "model": {
                "name": self.config.model_name,
                "file": "model.gguf",
                "family": self.config.model_family,
            },
            "inference": {
                "temperature": self.config.default_temperature,
                "context_size": self.config.default_context_size,
                "stop_sequences": ["<|im_end|>", "</tool_call>"],
            },
            "tools": {
                "file": "tools.json",
                "count": len(self.config.tools),
            },
            "quality": {},
            "deployment": {
                "ollama": {
                    "modelfile": "Modelfile",
                    "create_command": f"ollama create {self.config.model_name} -f Modelfile",
                },
                "llama_cpp": {
                    "command": f"llama-cli -m model.gguf --ctx-size {self.config.default_context_size}",
                },
            },
        }

        # Add quality metrics if available
        if self.config.tool_accuracy is not None:
            config_data["quality"]["tool_accuracy"] = round(self.config.tool_accuracy, 2)
        if self.config.schema_conformance is not None:
            config_data["quality"]["schema_conformance"] = round(self.config.schema_conformance, 2)
        if self.config.benchmark_score is not None:
            config_data["quality"]["benchmark_score"] = round(self.config.benchmark_score, 2)

        dest = self.config.output_dir / "config.yaml"
        with open(dest, "w") as f:
            # Add header comment
            f.write("# MCP-Forge Agent Bundle Configuration\n")
            f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        return dest

    def _generate_readme(self) -> Path:
        """Create README.md with usage instructions.

        Returns:
            Path to the generated README.md
        """
        # Build tool table
        tool_rows = []
        for tool in self.config.tools:
            desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
            tool_rows.append(f"| {tool.name} | {desc} |")

        tool_table = "\n".join(tool_rows) if tool_rows else "| (no tools) | - |"

        # Build quality metrics section
        quality_lines = []
        if self.config.tool_accuracy is not None:
            quality_lines.append(f"- Tool Selection Accuracy: {self.config.tool_accuracy:.0%}")
        if self.config.schema_conformance is not None:
            quality_lines.append(f"- Schema Conformance: {self.config.schema_conformance:.0%}")
        if self.config.benchmark_score is not None:
            quality_lines.append(f"- Benchmark Score: {self.config.benchmark_score:.0%}")

        quality_section = "\n".join(quality_lines) if quality_lines else "- No metrics available"

        # Calculate model size
        model_size_mb = self.config.gguf_path.stat().st_size / (1024 * 1024)

        readme_content = f"""# {self.config.model_name} - MCP Agent Bundle

{self.config.model_description or "Fine-tuned LLM for MCP tool calling."}

## Quick Start

### Ollama

```bash
ollama create {self.config.model_name} -f Modelfile
ollama run {self.config.model_name}
```

### llama.cpp

```bash
llama-cli -m model.gguf --ctx-size {self.config.default_context_size}
```

## Available Tools

| Tool | Description |
|------|-------------|
{tool_table}

## Quality Metrics

{quality_section}

## Files

- `model.gguf` - Quantized model ({model_size_mb:.1f} MB)
- `tools.json` - Tool schema definitions
- `config.yaml` - Deployment configuration
- `Modelfile` - Ollama import file

## Training Provenance

- Base Model: {self.config.model_family}
- Training Samples: {self.config.training_samples}
- Fine-tuning: LoRA + Unsloth

---
Generated by MCP-Forge v{__version__}
"""

        dest = self.config.output_dir / "README.md"
        with open(dest, "w") as f:
            f.write(readme_content)

        return dest

    def _generate_modelfile(self) -> Path:
        """Create Ollama Modelfile.

        Returns:
            Path to the generated Modelfile
        """
        # Build system prompt
        system_prompt = """You are an AI assistant with access to MCP tools. When you need to use a tool,
format your response using <tool_call> tags with valid JSON.

Available tools are defined in tools.json."""

        # ChatML template for Hermes format
        template = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

        modelfile_content = f"""# MCP-Forge Agent Bundle Modelfile
# Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d")}

FROM ./model.gguf

SYSTEM \"\"\"
{system_prompt}
\"\"\"

PARAMETER temperature {self.config.default_temperature}
PARAMETER num_ctx {self.config.default_context_size}
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "</tool_call>"

TEMPLATE \"\"\"{template}\"\"\"

LICENSE \"\"\"
Apache 2.0 - Fine-tuned model for MCP tool calling.
See https://github.com/anthropics/mcp for MCP specification.
\"\"\"
"""

        dest = self.config.output_dir / "Modelfile"
        with open(dest, "w") as f:
            f.write(modelfile_content)

        return dest

    def _validate_bundle(self) -> tuple[bool, list[str]]:
        """Validate bundle contents.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: list[str] = []

        # Check required files exist
        for required_file in self.REQUIRED_FILES:
            file_path = self.config.output_dir / required_file
            if not file_path.exists():
                errors.append(f"Missing required file: {required_file}")

        # Validate tools.json is valid JSON
        tools_path = self.config.output_dir / "tools.json"
        if tools_path.exists():
            try:
                with open(tools_path) as f:
                    tools_data = json.load(f)
                if "tools" not in tools_data:
                    errors.append("tools.json missing 'tools' key")
            except json.JSONDecodeError as e:
                errors.append(f"tools.json is not valid JSON: {e}")

        # Validate config.yaml is valid YAML
        config_path = self.config.output_dir / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                if config_data is None:
                    errors.append("config.yaml is empty")
            except yaml.YAMLError as e:
                errors.append(f"config.yaml is not valid YAML: {e}")

        # Cross-validate tool count
        if tools_path.exists() and config_path.exists():
            try:
                with open(tools_path) as f:
                    tools_data = json.load(f)
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)

                tools_count = len(tools_data.get("tools", []))
                config_count = config_data.get("tools", {}).get("count", 0)

                if tools_count != config_count:
                    errors.append(
                        f"Tool count mismatch: tools.json has {tools_count}, "
                        f"config.yaml has {config_count}"
                    )
            except Exception:
                pass  # Already caught above

        # Validate GGUF file size (basic check)
        model_path = self.config.output_dir / "model.gguf"
        if model_path.exists():
            size = model_path.stat().st_size
            if size < 1024:  # Less than 1KB is definitely wrong
                errors.append(f"model.gguf appears corrupt (size: {size} bytes)")

        return len(errors) == 0, errors


def verify_bundle(bundle_path: Path, smoke_test: bool = False) -> tuple[bool, list[str]]:
    """Verify an existing bundle.

    Args:
        bundle_path: Path to bundle directory
        smoke_test: Whether to run inference smoke test

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: list[str] = []

    # Check required files
    for required_file in BundleEngine.REQUIRED_FILES:
        file_path = bundle_path / required_file
        if not file_path.exists():
            errors.append(f"Missing required file: {required_file}")

    # Validate JSON/YAML files
    tools_path = bundle_path / "tools.json"
    if tools_path.exists():
        try:
            with open(tools_path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"tools.json is not valid JSON: {e}")

    config_path = bundle_path / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            errors.append(f"config.yaml is not valid YAML: {e}")

    # Check model file integrity
    model_path = bundle_path / "model.gguf"
    if model_path.exists():
        # Check GGUF magic bytes
        try:
            with open(model_path, "rb") as f:
                magic = f.read(4)
            # GGUF magic: "GGUF" (0x46554747)
            if magic != b"GGUF":
                errors.append(f"model.gguf has invalid magic bytes: {magic!r}")
        except Exception as e:
            errors.append(f"Could not read model.gguf: {e}")

    # Smoke test (optional)
    if smoke_test and not errors:
        try:
            # Try to load with llama-cpp-python if available
            from llama_cpp import Llama

            model = Llama(
                model_path=str(model_path),
                n_ctx=512,
                n_gpu_layers=0,
                verbose=False,
            )
            # Quick inference test
            output = model("Hello", max_tokens=5)
            if not output:
                errors.append("Smoke test: model produced no output")
            del model
        except ImportError:
            errors.append("Smoke test: llama-cpp-python not installed")
        except Exception as e:
            errors.append(f"Smoke test failed: {e}")

    return len(errors) == 0, errors
