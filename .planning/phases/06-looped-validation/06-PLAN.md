# Phase 6: Looped Validation - Execution Plan

**Phase**: 6 of 9
**Milestone**: v1.0 - Full Pipeline Implementation
**Created**: 2026-01-13
**Estimated Scope**: Medium (8 tasks)

---

## Objective

Implement the VALIDATING stage with real and stubbed MCP server testing. Create a `validation/` module that:
1. Provides deterministic MCP stubs for reproducible testing (weather, filesystem)
2. Executes validation loops against trained models (LoRA adapter or GGUF)
3. Calculates accuracy metrics (parse rate, schema conformance, tool selection)
4. Supports retry logic for real MCP server connections
5. Integrates with pipeline state and CLI

---

## Execution Context

**Architecture**: Follow patterns in `training/engine.py`, `data/qc.py`
**State Integration**: `ValidationResult` already defined in `state.py:107-137`
**CLI Stub**: `cli.py:877-883` - validate command exists but not implemented
**Pipeline**: `cli.py:302-308` - Stage 5 transition point
**Testing**: 242 existing tests, maintain 85%+ coverage
**Conventions**: `CONVENTIONS.md` - Dataclasses, type hints, docstrings

---

## Context

### Existing Infrastructure

| Component | Path | Integration Point |
|-----------|------|-------------------|
| ValidationResult | `state.py:107-137` | Metrics dataclass with thresholds |
| Formatter | `formatter.py:137-156` | `parse_tool_call()` for parsing responses |
| Inspector | `inspector.py:159-200` | `validate_tool_call()` for schema validation |
| CLI Stub | `cli.py:877-883` | validate command skeleton |
| Pipeline Stub | `cli.py:302-308` | Stage 5 VALIDATING transition |
| MCP Client | `inspector.py:41-101` | `stdio_client`, `ClientSession` patterns |

### ValidationResult Fields (state.py:107-137)

```python
@dataclass
class ValidationResult:
    passed: bool
    samples_tested: int
    samples_passed: int
    tool_call_parse_rate: float = 0.0      # Target: ≥98%
    schema_conformance_rate: float = 0.0   # Target: ≥95%
    tool_selection_accuracy: float = 0.0   # Target: ≥90%
    loop_completion_rate: float = 0.0      # Target: ≥95%
    error_handling_rate: float = 0.0
    failures: list[dict] = field(default_factory=list)
```

### Validation Flow

```
Model (LoRA/GGUF) + Test Prompts
            ↓
    Generate Response
            ↓
    Parse Tool Call  ───────────────→ parse_rate metric
            ↓
    Validate Schema  ───────────────→ schema_conformance metric
            ↓
    Execute on MCP   ───────────────→ (real or stub)
            ↓
    Verify Result    ───────────────→ tool_selection, loop_completion
            ↓
    Aggregate Metrics
            ↓
    ValidationResult
```

---

## Tasks

### Task 1: Create validation module structure

**Action**: Create `src/mcp_forge/validation/` package with module files.

**Files**:
- `src/mcp_forge/validation/__init__.py` - Module exports
- `src/mcp_forge/validation/config.py` - ValidationConfig, StubConfig dataclasses
- `src/mcp_forge/validation/stubs.py` - Deterministic MCP stub implementations
- `src/mcp_forge/validation/runner.py` - ValidationRunner class

**Structure**:
```
src/mcp_forge/validation/
├── __init__.py         # Export ValidationRunner, ValidationConfig, StubRegistry
├── config.py           # ValidationConfig, StubConfig, InferenceConfig
├── stubs.py            # StubRegistry, WeatherStub, FilesystemStub
└── runner.py           # ValidationRunner class
```

**Verification**: `python -c "from mcp_forge.validation import ValidationRunner"`

---

### Task 2: Implement configuration dataclasses

**Action**: Create configuration classes for validation runs.

**File**: `src/mcp_forge/validation/config.py`

**Implementation**:
```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class InferenceConfig:
    """Configuration for model inference during validation."""
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    do_sample: bool = True

@dataclass
class StubConfig:
    """Configuration for MCP stub behavior."""
    stub_type: str  # "weather", "filesystem"
    deterministic: bool = True
    seed: int = 42
    response_delay: float = 0.0  # Simulated latency

@dataclass
class ValidationConfig:
    """Configuration for a validation run."""
    model_path: Path              # LoRA adapter or GGUF path
    samples: int = 20             # Number of validation samples
    timeout: float = 30.0         # Per-sample timeout
    retry_count: int = 3          # Retries for real MCP connections
    retry_delay: float = 1.0      # Delay between retries

    # Thresholds (from ValidationResult.meets_release_criteria)
    parse_threshold: float = 0.98
    schema_threshold: float = 0.95
    accuracy_threshold: float = 0.90
    loop_threshold: float = 0.95

    # Inference settings
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Optional stub configuration (mutually exclusive with mcp_command)
    stub_config: StubConfig | None = None
    mcp_command: str | None = None

    def __post_init__(self):
        if self.stub_config is None and self.mcp_command is None:
            raise ValueError("Either stub_config or mcp_command must be provided")
```

**Verification**: Unit tests for config validation

---

### Task 3: Implement deterministic MCP stubs

**Action**: Create stub implementations for reproducible testing.

**File**: `src/mcp_forge/validation/stubs.py`

**Implementation**:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import random

class MCPStub(ABC):
    """Base class for deterministic MCP stubs."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._call_count = 0

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """Return tool definitions for this stub."""
        pass

    @abstractmethod
    def execute(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool call and return deterministic result."""
        pass

class WeatherStub(MCPStub):
    """Deterministic weather service stub."""

    CITIES = {
        "Paris": {"temp": 18, "condition": "cloudy"},
        "London": {"temp": 14, "condition": "rainy"},
        "Tokyo": {"temp": 22, "condition": "sunny"},
        "New York": {"temp": 16, "condition": "partly_cloudy"},
    }

    def get_tools(self) -> list[dict]:
        return [{
            "name": "get_weather",
            "description": "Get current weather for a location",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }]

    def execute(self, tool_name: str, arguments: dict) -> dict:
        self._call_count += 1
        if tool_name != "get_weather":
            return {"error": f"Unknown tool: {tool_name}"}

        location = arguments.get("location", "Unknown")
        units = arguments.get("units", "celsius")

        weather = self.CITIES.get(location, {"temp": 20, "condition": "unknown"})
        temp = weather["temp"]

        if units == "fahrenheit":
            temp = int(temp * 9/5 + 32)

        return {
            "location": location,
            "temperature": temp,
            "units": units,
            "condition": weather["condition"],
        }

class FilesystemStub(MCPStub):
    """Deterministic filesystem operations stub."""

    VIRTUAL_FS = {
        "/home/user/documents": ["report.pdf", "notes.txt"],
        "/home/user/projects": ["app.py", "test.py", "README.md"],
    }

    def get_tools(self) -> list[dict]:
        return [{
            "name": "list_files",
            "description": "List files in a directory",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": ["path"],
            },
        }]

    def execute(self, tool_name: str, arguments: dict) -> dict:
        self._call_count += 1
        if tool_name != "list_files":
            return {"error": f"Unknown tool: {tool_name}"}

        path = arguments.get("path", "/")
        files = self.VIRTUAL_FS.get(path, [])
        return {"path": path, "files": files}

class StubRegistry:
    """Registry of available MCP stubs."""

    STUBS = {
        "weather": WeatherStub,
        "filesystem": FilesystemStub,
    }

    @classmethod
    def get(cls, stub_type: str, seed: int = 42) -> MCPStub:
        if stub_type not in cls.STUBS:
            raise ValueError(f"Unknown stub: {stub_type}. Available: {list(cls.STUBS.keys())}")
        return cls.STUBS[stub_type](seed=seed)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls.STUBS.keys())
```

**Verification**: Unit tests for stub determinism

---

### Task 4: Implement ValidationRunner core

**Action**: Create the main validation runner with model loading and inference.

**File**: `src/mcp_forge/validation/runner.py`

**Implementation**:
```python
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass
import json

from mcp_forge.state import ToolDefinition, ValidationResult
from mcp_forge.data.formatter import parse_tool_call
from mcp_forge.tools.inspector import validate_tool_call
from mcp_forge.validation.config import ValidationConfig
from mcp_forge.validation.stubs import StubRegistry, MCPStub

@dataclass
class ValidationSample:
    """A single validation test case."""
    prompt: str
    expected_tool: str | None
    expected_args: dict[str, Any] | None

class ValidationRunner:
    """Executes validation loops against trained models."""

    def __init__(self, config: ValidationConfig, tools: list[ToolDefinition]):
        self.config = config
        self.tools = tools
        self.model = None
        self.tokenizer = None
        self._stub: MCPStub | None = None

    def load_model(self) -> None:
        """Load the trained model (LoRA adapter)."""
        # Import here to avoid heavy dependencies at module level
        from unsloth import FastLanguageModel

        model_path = self.config.model_path

        # Detect if GGUF or LoRA adapter
        if model_path.suffix == ".gguf":
            raise NotImplementedError("GGUF validation not yet implemented")

        # Load LoRA adapter
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Enable inference mode
        FastLanguageModel.for_inference(self.model)

    def _get_stub(self) -> MCPStub:
        """Get or create MCP stub."""
        if self._stub is None:
            if self.config.stub_config is None:
                raise ValueError("No stub configuration provided")
            self._stub = StubRegistry.get(
                self.config.stub_config.stub_type,
                seed=self.config.stub_config.seed,
            )
        return self._stub

    def generate_response(self, prompt: str) -> str:
        """Generate model response for a prompt."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model must be loaded before generating responses")

        # Build system prompt with tools
        from mcp_forge.data.formatter import format_system_prompt
        system_prompt = format_system_prompt(self.tools)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=self.config.inference.max_new_tokens,
            temperature=self.config.inference.temperature,
            top_p=self.config.inference.top_p,
            do_sample=self.config.inference.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the generated part
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True,
        )

        return response

    def validate_single(
        self,
        sample: ValidationSample,
    ) -> dict[str, Any]:
        """Validate a single sample and return metrics."""
        result = {
            "prompt": sample.prompt,
            "parsed": False,
            "schema_valid": False,
            "tool_correct": False,
            "loop_complete": False,
            "error": None,
        }

        try:
            # Generate response
            response = self.generate_response(sample.prompt)
            result["response"] = response

            # Parse tool call
            tool_call = parse_tool_call(response)
            if tool_call is None:
                if sample.expected_tool is None:
                    # Correctly identified no tool needed
                    result["parsed"] = True
                    result["schema_valid"] = True
                    result["tool_correct"] = True
                    result["loop_complete"] = True
                else:
                    result["error"] = "No tool call found in response"
                return result

            result["parsed"] = True
            result["tool_call"] = tool_call

            # Validate schema
            is_valid, error = validate_tool_call(tool_call, self.tools)
            result["schema_valid"] = is_valid
            if not is_valid:
                result["error"] = error
                return result

            # Check tool selection
            if sample.expected_tool:
                result["tool_correct"] = (tool_call["name"] == sample.expected_tool)
            else:
                result["tool_correct"] = True  # Any valid tool is acceptable

            # Execute against stub/server
            if self.config.stub_config:
                stub = self._get_stub()
                exec_result = stub.execute(tool_call["name"], tool_call.get("arguments", {}))
                result["execution_result"] = exec_result
                result["loop_complete"] = "error" not in exec_result
            else:
                # Real MCP server execution
                result["loop_complete"] = self._execute_on_real_server(tool_call)

        except Exception as e:
            result["error"] = str(e)

        return result

    def _execute_on_real_server(self, tool_call: dict) -> bool:
        """Execute tool call on real MCP server with retries."""
        import asyncio
        from mcp_forge.tools.inspector import inspect_mcp_server

        # TODO: Implement real MCP tool execution
        # For now, just verify server is reachable
        for attempt in range(self.config.retry_count):
            try:
                asyncio.run(inspect_mcp_server(
                    self.config.mcp_command,
                    timeout=self.config.timeout,
                ))
                return True
            except Exception:
                if attempt < self.config.retry_count - 1:
                    import time
                    time.sleep(self.config.retry_delay)
        return False

    def run(
        self,
        samples: list[ValidationSample],
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> ValidationResult:
        """Run full validation and return aggregated results."""
        if self.model is None:
            self.load_model()

        total = len(samples)
        results = []

        for i, sample in enumerate(samples):
            result = self.validate_single(sample)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, (i + 1) / total)

        # Aggregate metrics
        parsed = sum(1 for r in results if r["parsed"])
        schema_valid = sum(1 for r in results if r["schema_valid"])
        tool_correct = sum(1 for r in results if r["tool_correct"])
        loop_complete = sum(1 for r in results if r["loop_complete"])

        failures = [r for r in results if r.get("error")]

        validation_result = ValidationResult(
            passed=all([
                parsed / total >= self.config.parse_threshold,
                schema_valid / total >= self.config.schema_threshold,
                tool_correct / total >= self.config.accuracy_threshold,
                loop_complete / total >= self.config.loop_threshold,
            ]),
            samples_tested=total,
            samples_passed=sum(1 for r in results if r["loop_complete"] and not r.get("error")),
            tool_call_parse_rate=parsed / total if total > 0 else 0.0,
            schema_conformance_rate=schema_valid / total if total > 0 else 0.0,
            tool_selection_accuracy=tool_correct / total if total > 0 else 0.0,
            loop_completion_rate=loop_complete / total if total > 0 else 0.0,
            failures=[{"sample": f["prompt"], "error": f["error"]} for f in failures],
        )

        return validation_result
```

**Verification**: Unit test with mocked model

---

### Task 5: Add test sample generation

**Action**: Add utility to generate validation samples from training data or tool definitions.

**File**: `src/mcp_forge/validation/runner.py` (add to existing file)

**Implementation**:
```python
def generate_validation_samples(
    tools: list[ToolDefinition],
    count: int = 20,
    include_no_tool: bool = True,
    seed: int = 42,
) -> list[ValidationSample]:
    """Generate validation samples from tool definitions.

    Creates diverse test prompts for each tool, ensuring coverage
    of different argument combinations and edge cases.

    Args:
        tools: Tool definitions to generate samples for
        count: Total number of samples to generate
        include_no_tool: Whether to include no-tool test cases
        seed: Random seed for reproducibility

    Returns:
        List of ValidationSample objects
    """
    import random
    rng = random.Random(seed)

    samples = []

    # Templates for generating prompts
    WEATHER_PROMPTS = [
        "What's the weather in {location}?",
        "Is it going to rain in {location} today?",
        "Tell me the current temperature in {location}.",
        "What should I wear in {location} based on the weather?",
    ]

    FILESYSTEM_PROMPTS = [
        "List the files in {path}.",
        "What files are in the {path} directory?",
        "Show me what's inside {path}.",
    ]

    NO_TOOL_PROMPTS = [
        "What is the capital of France?",
        "How do I make pasta?",
        "Tell me a joke.",
        "What year did World War II end?",
    ]

    # Generate tool-specific samples
    samples_per_tool = count // len(tools) if tools else 0

    for tool in tools:
        if tool.name == "get_weather":
            locations = ["Paris", "London", "Tokyo", "New York", "Sydney"]
            for i in range(samples_per_tool):
                location = rng.choice(locations)
                prompt = rng.choice(WEATHER_PROMPTS).format(location=location)
                samples.append(ValidationSample(
                    prompt=prompt,
                    expected_tool="get_weather",
                    expected_args={"location": location},
                ))

        elif tool.name == "list_files":
            paths = ["/home/user/documents", "/home/user/projects", "/tmp"]
            for i in range(samples_per_tool):
                path = rng.choice(paths)
                prompt = rng.choice(FILESYSTEM_PROMPTS).format(path=path)
                samples.append(ValidationSample(
                    prompt=prompt,
                    expected_tool="list_files",
                    expected_args={"path": path},
                ))

        else:
            # Generic sample generation for unknown tools
            for i in range(samples_per_tool):
                samples.append(ValidationSample(
                    prompt=f"Use the {tool.name} tool with default arguments.",
                    expected_tool=tool.name,
                    expected_args={},
                ))

    # Add no-tool samples
    if include_no_tool:
        no_tool_count = max(1, count // 5)  # ~20% no-tool
        for prompt in rng.sample(NO_TOOL_PROMPTS, min(no_tool_count, len(NO_TOOL_PROMPTS))):
            samples.append(ValidationSample(
                prompt=prompt,
                expected_tool=None,
                expected_args=None,
            ))

    rng.shuffle(samples)
    return samples[:count]
```

**Verification**: Unit test for sample generation diversity

---

### Task 6: Integrate with CLI

**Action**: Wire ValidationRunner into CLI `validate` command and pipeline.

**File**: `src/mcp_forge/cli.py`

**Replace** validate command (lines 873-883):
```python
@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to LoRA adapter or GGUF")
@click.option("--server", "-s", help="MCP server command for live validation")
@click.option("--stub", type=click.Choice(["weather", "filesystem"]), help="Use deterministic stub")
@click.option("--samples", default=20, help="Number of validation samples")
@click.option("--tools", "-t", type=click.Path(exists=True), help="Tools JSON file (required for stub)")
@click.option("--threshold", type=float, help="Override default pass threshold (0.90)")
@click.pass_context
def validate(ctx, model: str, server: str | None, stub: str | None, samples: int, tools: str | None, threshold: float | None):
    """Run looped validation against real or stubbed MCP server."""
    import json
    from pathlib import Path

    from mcp_forge.state import ToolDefinition
    from mcp_forge.validation import ValidationRunner, ValidationConfig, StubConfig
    from mcp_forge.validation.runner import generate_validation_samples

    if not server and not stub:
        console.print("[red]Error: Either --server or --stub is required[/red]")
        raise SystemExit(1)

    console.print("\n[bold]MCP-Forge Looped Validation[/bold]")
    console.print("=" * 50)

    # Load tools
    if stub:
        # Get tools from stub
        from mcp_forge.validation.stubs import StubRegistry
        stub_instance = StubRegistry.get(stub)
        tool_defs = [
            ToolDefinition(
                name=t["name"],
                description=t["description"],
                input_schema=t["inputSchema"],
            )
            for t in stub_instance.get_tools()
        ]
    elif tools:
        with open(tools) as f:
            tool_defs = [ToolDefinition.from_dict(t) for t in json.load(f)]
    else:
        # Extract from MCP server
        tool_defs = asyncio.run(inspect_mcp_server(server))

    console.print(f"Model: {model}")
    console.print(f"Mode: {'Stub (' + stub + ')' if stub else 'Live MCP'}")
    console.print(f"Tools: {len(tool_defs)}")
    console.print(f"Samples: {samples}")

    # Configure validation
    stub_config = StubConfig(stub_type=stub, deterministic=True) if stub else None
    config = ValidationConfig(
        model_path=Path(model),
        samples=samples,
        stub_config=stub_config,
        mcp_command=server,
        accuracy_threshold=threshold or 0.90,
    )

    runner = ValidationRunner(config, tool_defs)

    # Generate validation samples
    console.print("\nGenerating validation samples...")
    validation_samples = generate_validation_samples(tool_defs, count=samples)
    console.print(f"Generated {len(validation_samples)} samples")

    # Run validation
    console.print("\nRunning validation...")

    def on_progress(current: int, total: int, pct: float):
        console.print(f"  Sample {current}/{total} ({pct:.0%})")

    try:
        result = runner.run(validation_samples, progress_callback=on_progress)
    except Exception as e:
        console.print(f"\n[red]Validation failed: {e}[/red]")
        raise SystemExit(1)

    # Print results
    console.print("\n" + "=" * 50)
    console.print("[bold]Validation Results[/bold]")
    console.print(f"Passed: {'[green]YES[/green]' if result.passed else '[red]NO[/red]'}")
    console.print(f"Samples: {result.samples_passed}/{result.samples_tested}")
    console.print(f"Parse rate: {result.tool_call_parse_rate:.1%}")
    console.print(f"Schema conformance: {result.schema_conformance_rate:.1%}")
    console.print(f"Tool selection accuracy: {result.tool_selection_accuracy:.1%}")
    console.print(f"Loop completion: {result.loop_completion_rate:.1%}")

    if result.failures:
        console.print(f"\n[yellow]Failures ({len(result.failures)}):[/yellow]")
        for f in result.failures[:5]:
            console.print(f"  - {f['error'][:60]}...")

    # Save report
    state_manager: StateManager = ctx.obj["state_manager"]
    state_manager.ensure_dirs()

    report_path = state_manager.get_report_path("validation_latest.json")
    with open(report_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    console.print(f"\nReport saved: {report_path}")

    if not result.passed:
        raise SystemExit(1)
```

**Also update** pipeline Stage 5 (cli.py:302-308):
```python
# Stage 5: Validation
if state.stage in (PipelineStage.TRAINING, PipelineStage.VALIDATING):
    console.print("\n[bold]Stage 5: Validating model...[/bold]")
    state.update_stage(PipelineStage.VALIDATING)
    state_manager.save_state(state)

    from mcp_forge.validation import ValidationRunner, ValidationConfig, StubConfig
    from mcp_forge.validation.runner import generate_validation_samples

    # Default to weather stub for pipeline validation
    stub_config = StubConfig(stub_type="weather", deterministic=True)
    val_config = ValidationConfig(
        model_path=Path(state.lora_adapter_path),
        samples=20,
        stub_config=stub_config,
    )

    runner = ValidationRunner(val_config, state.tools)
    samples = generate_validation_samples(state.tools, count=20)

    def on_progress(current: int, total: int, pct: float):
        console.print(f"   {current}/{total} ({pct:.0%})")

    result = runner.run(samples, progress_callback=on_progress)

    state.validation_result = result
    state_manager.save_state(state)

    console.print(f"   Parse rate: {result.tool_call_parse_rate:.1%}")
    console.print(f"   Schema conformance: {result.schema_conformance_rate:.1%}")
    console.print(f"   Tool accuracy: {result.tool_selection_accuracy:.1%}")

    if result.passed:
        console.print("   [green]✓[/green] Validation passed")
    else:
        console.print("   [yellow]⚠[/yellow] Validation below thresholds (continuing)")
```

**Verification**: CLI smoke test with `mcp-forge validate --help`

---

### Task 7: Add module exports

**Action**: Create proper `__init__.py` with exports.

**File**: `src/mcp_forge/validation/__init__.py`

**Implementation**:
```python
"""Looped validation module for trained model verification.

Provides validation against real and stubbed MCP servers with
deterministic testing and metric aggregation.
"""

from mcp_forge.validation.config import (
    InferenceConfig,
    StubConfig,
    ValidationConfig,
)
from mcp_forge.validation.runner import (
    ValidationRunner,
    ValidationSample,
    generate_validation_samples,
)
from mcp_forge.validation.stubs import (
    FilesystemStub,
    MCPStub,
    StubRegistry,
    WeatherStub,
)

__all__ = [
    # Config
    "InferenceConfig",
    "StubConfig",
    "ValidationConfig",
    # Runner
    "ValidationRunner",
    "ValidationSample",
    "generate_validation_samples",
    # Stubs
    "MCPStub",
    "WeatherStub",
    "FilesystemStub",
    "StubRegistry",
]
```

**Verification**: `python -c "from mcp_forge.validation import *; print('OK')"`

---

### Task 8: Add tests

**Action**: Create unit and integration tests for validation module.

**Files**:
- `tests/unit/test_validation_config.py` - Config validation
- `tests/unit/test_validation_stubs.py` - Stub determinism and correctness
- `tests/unit/test_validation_runner.py` - Runner logic with mocked model
- `tests/integration/test_validation_pipeline.py` - End-to-end pipeline validation

**Test Cases**:

```python
# test_validation_config.py
def test_config_requires_server_or_stub():
    """Config raises if neither server nor stub provided."""

def test_config_accepts_stub_config():
    """Config accepts valid stub configuration."""

def test_config_accepts_mcp_command():
    """Config accepts MCP server command."""

# test_validation_stubs.py
def test_weather_stub_deterministic():
    """Same seed produces same results."""

def test_weather_stub_returns_valid_response():
    """Weather stub returns expected format."""

def test_filesystem_stub_deterministic():
    """Filesystem stub is deterministic."""

def test_stub_registry_get():
    """Registry returns correct stub type."""

def test_stub_registry_unknown_raises():
    """Registry raises for unknown stub type."""

# test_validation_runner.py
@pytest.fixture
def mock_model():
    """Mock FastLanguageModel for tests without GPU."""

def test_runner_requires_loaded_model(mock_model):
    """Runner raises if model not loaded."""

def test_validate_single_parses_tool_call(mock_model):
    """Single validation correctly parses tool call."""

def test_validate_single_handles_no_tool(mock_model):
    """Validation handles no-tool scenarios."""

def test_run_aggregates_metrics(mock_model):
    """Full run aggregates metrics correctly."""

def test_generate_samples_covers_tools():
    """Sample generation covers all tools."""

def test_generate_samples_includes_no_tool():
    """Sample generation includes no-tool cases."""

# test_validation_pipeline.py
@pytest.mark.integration
def test_pipeline_validation_stage(mock_model, state_manager):
    """Validation stage integrates with pipeline state."""
```

**Fixtures**:
```python
@pytest.fixture
def mock_unsloth_inference(monkeypatch):
    """Mock Unsloth for validation inference tests."""
    # Similar to training tests but with generate() mocked
```

**Verification**: `pytest tests/unit/test_validation*.py tests/integration/test_validation*.py -v`

---

## Verification

After all tasks complete:

```bash
# 1. Lint and type check
ruff check src/mcp_forge/validation/
mypy src/mcp_forge/validation/

# 2. Run validation tests
pytest tests/unit/test_validation*.py -v
pytest tests/integration/test_validation*.py -v

# 3. Check coverage
pytest --cov=src/mcp_forge/validation --cov-report=term-missing

# 4. CLI smoke tests
mcp-forge validate --help
mcp-forge validate --model ./test-adapter --stub weather --samples 5

# 5. Verify imports
python -c "from mcp_forge.validation import ValidationRunner, StubRegistry; print('OK')"
```

---

## Success Criteria

- [ ] `src/mcp_forge/validation/` module created with 4 files
- [ ] ValidationConfig and StubConfig dataclasses defined
- [ ] WeatherStub and FilesystemStub produce deterministic results
- [ ] StubRegistry provides access to available stubs
- [ ] ValidationRunner loads LoRA adapters and runs inference
- [ ] ValidationRunner calculates parse_rate, schema_conformance, accuracy, loop_completion
- [ ] ValidationResult integrates with pipeline state
- [ ] CLI `validate` command wired to ValidationRunner
- [ ] Pipeline stage transitions correctly (TRAINING → VALIDATING)
- [ ] Retry logic handles flaky MCP connections
- [ ] All tests pass with mocked Unsloth (no GPU required)
- [ ] Coverage maintained at 85%+
- [ ] ruff and mypy pass on new code

---

## Output

| Artifact | Path |
|----------|------|
| Validation module | `src/mcp_forge/validation/` |
| Config tests | `tests/unit/test_validation_config.py` |
| Stub tests | `tests/unit/test_validation_stubs.py` |
| Runner tests | `tests/unit/test_validation_runner.py` |
| Integration tests | `tests/integration/test_validation_pipeline.py` |
| Validation reports (runtime) | `.mcp-forge/reports/validation_*.json` |

---

## Notes

### GPU Requirements

- Validation requires CUDA GPU for model inference (same as training)
- Tests use mocked Unsloth to run without GPU
- Integration tests marked with `@pytest.mark.integration` can be skipped in CI

### Stub vs Real MCP

| Mode | Use Case | Determinism | Speed |
|------|----------|-------------|-------|
| Stub | CI/Testing, Development | Yes | Fast |
| Real MCP | Pre-release validation | No | Slower |

Default to stub mode for pipeline validation; real MCP for final verification.

### Future Enhancements (Phase 7)

- GGUF model inference support (currently LoRA only)
- More stub types (database, API)
- Parallel validation for large sample sets
- Baseline comparison with pre-training model

---

*Plan created: 2026-01-13*
*Execute with: /gsd:execute-plan*
