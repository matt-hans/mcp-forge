# Phase 9: Bundle Packaging - Execution Plan

## Objective

Implement the PACKAGING stage to create distributable Agent Bundles containing all artifacts needed to deploy an MCP-aware fine-tuned model: GGUF model, tool definitions, deployment configuration, documentation, and optional Ollama Modelfile.

## Execution Context

**Files to create:**
- `src/mcp_forge/export/bundle.py` - Bundle assembly engine
- `tests/unit/test_bundle.py` - Unit tests
- `tests/integration/test_bundle_pipeline.py` - Pipeline integration tests

**Files to modify:**
- `src/mcp_forge/export/__init__.py` - Export bundle classes
- `src/mcp_forge/cli.py` - Implement `pack` and `verify-bundle` commands
- `src/mcp_forge/state.py` - Add BundleResult dataclass if needed

**Dependencies available:**
- PyYAML (in pyproject.toml)
- pathlib, shutil (stdlib)
- GGUFMetadata, ExportResult from export module
- ToolDefinition, ValidationResult, BenchmarkResult from state module

## Context

**Prior phase outputs:**
- Phase 8 provides GGUF model at `state.gguf_path`
- Phase 7 provides `state.benchmark_result` with quality metrics
- Phase 6 provides `state.validation_result` with accuracy metrics
- Phase 1-3 provide `state.tools` with ToolDefinition list

**CLI stub exists at `cli.py:1386-1394`:**
```python
@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--tools", "-t", required=True, type=click.Path(exists=True))
@click.option("--validation", type=click.Path(exists=True))
@click.option("--benchmark", "bench", type=click.Path(exists=True))
@click.option("--output", "-o", required=True)
def pack(model, tools, validation, bench, output):
    """Create distributable agent bundle."""
```

**Ollama Modelfile format (2026):**
- `FROM ./model.gguf` - Base model path (required)
- `REQUIRES 0.12.0` - Minimum Ollama version
- `SYSTEM "..."` - System prompt
- `PARAMETER temperature 0.3` - Inference parameters
- `TEMPLATE "..."` - Go template for ChatML/Hermes format
- `LICENSE "..."` - License text
- `MESSAGE user/assistant "..."` - Few-shot examples

---

## Tasks

### Task 1: Create BundleConfig and BundleResult Dataclasses

**File:** `src/mcp_forge/export/bundle.py`

Create configuration and result dataclasses:

```python
@dataclass
class BundleConfig:
    """Configuration for bundle packaging."""

    gguf_path: Path               # Input GGUF model
    tools_path: Path | None       # Optional tools.json (or use ToolDefinition list)
    tools: list[ToolDefinition]   # Tool definitions from state
    output_dir: Path              # Output bundle directory

    # Optional metadata sources
    validation_report_path: Path | None = None
    benchmark_report_path: Path | None = None

    # Bundle options
    include_modelfile: bool = True      # Generate Ollama Modelfile
    include_readme: bool = True         # Generate README.md
    model_name: str = ""                # Human-readable name
    model_description: str = ""         # Short description

    # Deployment defaults
    default_temperature: float = 0.3
    default_context_size: int = 8192

@dataclass
class BundleResult:
    """Result of bundle packaging."""

    success: bool
    bundle_path: Path | None

    # Contents
    files_created: list[str]
    bundle_size_mb: float

    # Validation
    validation_passed: bool
    validation_errors: list[str]

    error: str | None = None
```

**Verification:** Unit tests for dataclass initialization and validation.

---

### Task 2: Implement BundleEngine Core

**File:** `src/mcp_forge/export/bundle.py`

Implement the bundle assembly engine:

```python
class BundleEngine:
    """Assembles distributable agent bundles."""

    def __init__(self, config: BundleConfig):
        self.config = config

    def package(self, progress_callback=None) -> BundleResult:
        """Create the complete bundle."""
        # 1. Create output directory
        # 2. Copy GGUF model
        # 3. Generate tools.json
        # 4. Generate config.yaml
        # 5. Generate README.md (if enabled)
        # 6. Generate Modelfile (if enabled)
        # 7. Validate bundle
        # 8. Return result

    def _copy_model(self) -> Path:
        """Copy GGUF to bundle directory."""

    def _generate_tools_json(self) -> Path:
        """Create tools.json with schema definitions."""

    def _generate_config_yaml(self) -> Path:
        """Create deployment config.yaml."""

    def _generate_readme(self) -> Path:
        """Create README.md with usage instructions."""

    def _generate_modelfile(self) -> Path:
        """Create Ollama Modelfile."""

    def _validate_bundle(self) -> tuple[bool, list[str]]:
        """Validate bundle contents."""
```

**Verification:** Unit tests with mocked file operations.

---

### Task 3: Implement tools.json Generation

**Method:** `BundleEngine._generate_tools_json()`

Generate OpenAI-compatible function definitions:

```json
{
  "version": "1.0",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "metadata": {
    "source": "mcp-forge",
    "generated_at": "2026-01-13T...",
    "tool_count": 5
  }
}
```

**Verification:** Unit test validates JSON structure and schema conformance.

---

### Task 4: Implement config.yaml Generation

**Method:** `BundleEngine._generate_config_yaml()`

Create deployment configuration:

```yaml
# MCP-Forge Agent Bundle Configuration
# Generated: 2026-01-13T...

model:
  name: mcp-weather-agent
  file: model.gguf
  family: deepseek-r1

inference:
  temperature: 0.3
  context_size: 8192
  stop_sequences:
    - "<|im_end|>"
    - "</tool_call>"

tools:
  file: tools.json
  count: 5

quality:
  tool_accuracy: 0.94
  schema_conformance: 0.97
  benchmark_score: 0.91

deployment:
  ollama:
    modelfile: Modelfile
    create_command: "ollama create {name} -f Modelfile"
  llama_cpp:
    command: "llama-cli -m model.gguf --ctx-size 8192"
```

**Verification:** Unit test validates YAML structure and required fields.

---

### Task 5: Implement README.md Generation

**Method:** `BundleEngine._generate_readme()`

Create usage documentation:

```markdown
# {Model Name} - MCP Agent Bundle

Fine-tuned LLM for MCP tool calling.

## Quick Start

### Ollama
```bash
ollama create {name} -f Modelfile
ollama run {name}
```

### llama.cpp
```bash
llama-cli -m model.gguf --ctx-size 8192
```

## Available Tools

| Tool | Description |
|------|-------------|
| get_weather | Get current weather for a location |
| ... | ... |

## Quality Metrics

- Tool Selection Accuracy: 94%
- Schema Conformance: 97%
- Benchmark Score: 91%

## Files

- `model.gguf` - Quantized model ({size} MB)
- `tools.json` - Tool schema definitions
- `config.yaml` - Deployment configuration
- `Modelfile` - Ollama import file

## Training Provenance

- Base Model: deepseek-r1
- Training Samples: 1000
- Fine-tuning: LoRA + Unsloth

---
Generated by MCP-Forge v{version}
```

**Verification:** Unit test validates README structure and placeholder substitution.

---

### Task 6: Implement Ollama Modelfile Generation

**Method:** `BundleEngine._generate_modelfile()`

Create Ollama-compatible Modelfile:

```dockerfile
# MCP-Forge Agent Bundle Modelfile
# Generated: 2026-01-13

FROM ./model.gguf

REQUIRES 0.12.0

SYSTEM """
You are an AI assistant with access to MCP tools. When you need to use a tool,
format your response using <tool_call> tags with valid JSON.

Available tools are defined in tools.json.
"""

PARAMETER temperature 0.3
PARAMETER num_ctx 8192
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "</tool_call>"

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

LICENSE """
Apache 2.0 - Fine-tuned model for MCP tool calling.
See https://github.com/anthropics/mcp for MCP specification.
"""
```

**Verification:** Unit test validates Modelfile syntax and required instructions.

---

### Task 7: Implement Bundle Validation

**Method:** `BundleEngine._validate_bundle()`

Validate bundle before completion:

1. **Required files exist:**
   - model.gguf
   - tools.json
   - config.yaml

2. **File integrity:**
   - GGUF file is valid (load with llama-cpp-python)
   - tools.json is valid JSON
   - config.yaml is valid YAML

3. **Cross-validation:**
   - Tool count in config matches tools.json
   - Model name consistent across files

**Verification:** Integration test with real bundle directory.

---

### Task 8: Integrate with CLI `pack` Command

**File:** `src/mcp_forge/cli.py`

Replace stub with full implementation:

```python
@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--tools", "-t", type=click.Path(exists=True))
@click.option("--output", "-o", required=True)
@click.option("--name", help="Model name for bundle")
@click.option("--no-modelfile", is_flag=True, help="Skip Ollama Modelfile")
@click.option("--no-readme", is_flag=True, help="Skip README generation")
@click.pass_context
def pack(ctx, model, tools, output, name, no_modelfile, no_readme):
    """Create distributable agent bundle."""
```

Features:
- Load tools from file or pipeline state
- Enrich with validation/benchmark from state
- Progress display with Rich
- Report generation

**Verification:** CLI integration test with mock data.

---

### Task 9: Implement `verify-bundle` Command

**File:** `src/mcp_forge/cli.py`

Implement bundle verification:

```python
@cli.command("verify-bundle")
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--smoke-test", is_flag=True, help="Run inference smoke test")
def verify_bundle(bundle_path, smoke_test):
    """Verify an agent bundle and optionally run smoke tests."""
```

Checks:
1. Required files present
2. GGUF loads correctly
3. tools.json valid
4. config.yaml valid
5. Optional: Run test inference

**Verification:** Integration test with valid/invalid bundles.

---

### Task 10: Pipeline Stage Integration

**File:** `src/mcp_forge/cli.py` (in main `run` command)

Add PACKAGING stage to pipeline:

```python
# Stage 8: Package bundle
if state.stage == PipelineStage.EXPORTING:
    state.stage = PipelineStage.PACKAGING
    state_manager.save_state(state)

if state.stage == PipelineStage.PACKAGING:
    config = BundleConfig(
        gguf_path=Path(state.gguf_path),
        tools=state.tools,
        output_dir=Path(state.output_path),
    )
    engine = BundleEngine(config)
    result = engine.package()

    if result.success:
        state.bundle_path = str(result.bundle_path)
        state.stage = PipelineStage.COMPLETE
    else:
        state.error = result.error
        state.stage = PipelineStage.FAILED
```

**Verification:** Full pipeline integration test.

---

### Task 11: Export Module Updates

**File:** `src/mcp_forge/export/__init__.py`

Add exports:

```python
from mcp_forge.export.bundle import BundleConfig, BundleEngine, BundleResult

__all__ = [
    # Existing
    "ExportConfig",
    "ExportEngine",
    "ExportResult",
    "QuantizationType",
    "GGUFMetadata",
    "read_gguf_metadata",
    # New
    "BundleConfig",
    "BundleEngine",
    "BundleResult",
]
```

**Verification:** Import test.

---

### Task 12: Comprehensive Test Suite

**Files:**
- `tests/unit/test_bundle.py`
- `tests/integration/test_bundle_pipeline.py`

**Unit tests (~20):**
- BundleConfig validation
- BundleResult serialization
- tools.json generation
- config.yaml generation
- README.md generation
- Modelfile generation
- Bundle validation logic

**Integration tests (~10):**
- Full bundle creation with mocked GGUF
- CLI pack command
- CLI verify-bundle command
- Pipeline stage progression
- Error handling (missing files, invalid inputs)

**Verification:** `pytest tests/unit/test_bundle.py tests/integration/test_bundle_pipeline.py -v`

---

## Verification

### Automated Checks

```bash
# Run all Phase 9 tests
pytest tests/unit/test_bundle.py tests/integration/test_bundle_pipeline.py -v

# Type checking
mypy src/mcp_forge/export/bundle.py

# Linting
ruff check src/mcp_forge/export/bundle.py

# Coverage (target: 85%+)
pytest --cov=src/mcp_forge/export --cov-report=term-missing
```

### Manual Verification

```bash
# Create bundle from GGUF
mcp-forge pack -m ./model.gguf -t tools.json -o ./dist/agent

# Verify bundle
mcp-forge verify-bundle ./dist/agent

# Test Ollama import (requires Ollama)
cd ./dist/agent && ollama create test-agent -f Modelfile
```

---

## Success Criteria

- [ ] `mcp-forge pack --model model.gguf --tools tools.json --output ./dist/agent` creates bundle
- [ ] Bundle contains: model.gguf, tools.json, config.yaml
- [ ] README.md documents usage instructions
- [ ] Modelfile enables `ollama create` workflow
- [ ] Bundle validates before packaging completes
- [ ] All tests pass (`pytest` green)
- [ ] Type checking passes (`mypy` clean)
- [ ] Linting passes (`ruff check` clean)
- [ ] Test count increases by ~30 (target: 424 total)

---

## Output

**New files:**
- `src/mcp_forge/export/bundle.py` - Bundle assembly engine
- `tests/unit/test_bundle.py` - Unit tests
- `tests/integration/test_bundle_pipeline.py` - Integration tests

**Modified files:**
- `src/mcp_forge/export/__init__.py` - Export bundle classes
- `src/mcp_forge/cli.py` - pack and verify-bundle commands

**Artifacts:**
- Bundle directory structure at output path
- Validation report in `.mcp-forge/reports/`

---

## Scope Estimate

**Complexity:** Medium
**Tasks:** 12
**Estimated tests:** ~30 new tests
**Files to create:** 3
**Files to modify:** 2

This plan fits within a single execution session.

---

*Created: 2026-01-13*
