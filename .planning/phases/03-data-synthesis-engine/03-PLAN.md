# Phase 3: Data Synthesis Engine - Implementation Plan

**Created**: 2026-01-13
**Phase**: 3 of 9 (v1.0 Milestone)
**Estimated Scope**: Medium-Large (~800-1000 lines new code)
**Research**: [RESEARCH.md](./RESEARCH.md) completed

---

## Objective

Implement the SYNTHESIZING pipeline stage to generate high-quality training data for tool-calling LLMs using GPT-5 for seed generation and LLM-based paraphrasing for augmentation.

---

## Prerequisites

- [x] Phase 1: Foundation & Package Structure (complete)
- [x] Phase 2: Test Infrastructure (complete)
- [x] Phase 3 Research: RESEARCH.md created
- [ ] OpenAI API key with GPT-5 access (`OPENAI_API_KEY` env var)

---

## Architecture Summary

```
src/mcp_forge/data/
├── __init__.py          [UPDATE - add exports]
├── qc.py                [EXISTS - reuse for validation]
├── synthesizer.py       [NEW - main orchestrator]
├── seed_generator.py    [NEW - GPT-5 seed generation]
├── augmenter.py         [NEW - paraphrasing expansion]
└── formatter.py         [NEW - Hermes ChatML formatting]
```

**Integration Points**:
- `cli.py:162-170` - Replace TODO in Stage 2
- `state.py:163-192` - Use existing `SynthesisPlan` dataclass
- `inspector.py:127-151` - Use `generate_tool_use_prompt()`
- `qc.py:106-192` - Validate output with `DataQualityController`

---

## Tasks

### Task 1: Create Hermes ChatML Formatter

**File**: `src/mcp_forge/data/formatter.py`

**Purpose**: Format training samples in Hermes ChatML format for tool-calling.

**Implementation**:

```python
# Key functions to implement:

def format_system_prompt(tools: list[ToolDefinition]) -> str:
    """Generate Hermes-style system prompt with <tools> XML block."""
    # Template from RESEARCH.md Hermes format section

def format_tool_call(name: str, arguments: dict) -> str:
    """Wrap tool call in <tool_call> XML tags."""
    # Returns: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>

def format_tool_response(response: dict | str) -> str:
    """Format tool execution response."""

def create_training_sample(
    sample_id: str,
    source: str,  # "seed" or "augmented"
    scenario: str,
    tool_name: str | None,
    user_message: str,
    assistant_response: str,
    tool_response: str | None = None,
    tools: list[ToolDefinition] = None,
) -> dict:
    """Create complete JSONL-ready training sample."""
```

**Acceptance Criteria**:
- [ ] Generates valid Hermes ChatML format
- [ ] Supports all 5 scenario types
- [ ] Output passes existing QC validation
- [ ] Unit tests cover all formatters

---

### Task 2: Implement GPT-5 Seed Generator

**File**: `src/mcp_forge/data/seed_generator.py`

**Purpose**: Generate diverse seed training samples using GPT-5 function calling.

**Implementation**:

```python
@dataclass
class SeedGeneratorConfig:
    model: str = "gpt-5"
    temperature: float = 0.8
    max_retries: int = 3
    batch_size: int = 10

class SeedGenerator:
    def __init__(self, config: SeedGeneratorConfig, tools: list[ToolDefinition]):
        self.client = openai.OpenAI()
        self.config = config
        self.tools = tools

    async def generate_seeds(
        self,
        plan: SynthesisPlan,
        output_path: Path,
        progress_callback: Callable | None = None,
    ) -> list[dict]:
        """Generate seed samples according to synthesis plan."""

    def _create_seed_prompt(self, scenario: str, tool: ToolDefinition | None) -> list[dict]:
        """Create prompt for GPT-5 to generate a training example."""

    def _parse_gpt_response(self, response, scenario: str, tool: ToolDefinition | None) -> dict:
        """Parse GPT-5 response into training sample format."""
```

**Scenario-Specific Prompts**:

| Scenario | Prompt Strategy |
|----------|-----------------|
| `standard` | "Generate a user query that requires using {tool_name}" |
| `no_tool` | "Generate a conversational query that does NOT require any tool" |
| `error` | "Generate a query with invalid/missing parameters for {tool_name}" |
| `ambiguous` | "Generate an unclear query where tool use is uncertain" |
| `edge` | "Generate an edge case query for {tool_name}" |

**GPT-5 Configuration**:
```python
# Use strict mode for reliable schema conformance
tools=[{
    "type": "function",
    "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        "strict": True  # Critical for schema conformance
    }
}]
```

**Acceptance Criteria**:
- [ ] Generates samples for all 5 scenario types
- [ ] Respects `SynthesisPlan.scenario_weights` distribution
- [ ] Uses GPT-5 with `strict: true` for function calling
- [ ] Handles API errors with retry logic
- [ ] Outputs valid JSONL format
- [ ] Progress reporting via callback

---

### Task 3: Implement Paraphrase Augmenter

**File**: `src/mcp_forge/data/augmenter.py`

**Purpose**: Expand seed data via LLM paraphrasing (target 10x expansion).

**Implementation**:

```python
@dataclass
class AugmenterConfig:
    model: str = "gpt-5"
    expansion_factor: int = 10
    temperature_range: tuple[float, float] = (0.7, 1.0)
    max_synthetic_ratio: float = 0.30  # Prevent model collapse

class DataAugmenter:
    def __init__(self, config: AugmenterConfig):
        self.client = openai.OpenAI()
        self.config = config

    async def augment_dataset(
        self,
        seed_samples: list[dict],
        target_total: int,
        output_path: Path,
        progress_callback: Callable | None = None,
    ) -> list[dict]:
        """Expand seed samples via paraphrasing."""

    def _paraphrase_query(self, original_query: str, temperature: float) -> str:
        """Generate paraphrased version of user query."""

    def _vary_parameters(self, tool_call: dict, tool: ToolDefinition) -> dict:
        """Create variations of tool call parameters."""
```

**Augmentation Strategies**:

1. **Query Paraphrasing** (primary)
   - Rephrase user message
   - Keep tool call identical
   - Vary temperature for diversity

2. **Parameter Variation** (secondary)
   - Same query structure
   - Different argument values
   - E.g., different cities for weather

3. **Quality Filtering**
   - Deduplicate near-identical outputs
   - Validate schema conformance
   - Remove low-quality paraphrases

**Model Collapse Prevention**:
```python
# From research: keep synthetic ratio ≤30%
final_synthetic_ratio = augmented_count / total_count
if final_synthetic_ratio > config.max_synthetic_ratio:
    # Warn and truncate augmented samples
```

**Acceptance Criteria**:
- [ ] Achieves 10x expansion from seeds
- [ ] Maintains scenario distribution
- [ ] Synthetic ratio stays under 30%
- [ ] Deduplicates augmented samples
- [ ] Validates all outputs against schema

---

### Task 4: Create Main Synthesizer Orchestrator

**File**: `src/mcp_forge/data/synthesizer.py`

**Purpose**: Orchestrate the full synthesis pipeline.

**Implementation**:

```python
@dataclass
class SynthesisResult:
    seed_count: int
    augmented_count: int
    total_count: int
    seed_path: Path
    training_path: Path
    qc_passed: bool
    qc_report: QCReport | None

class DataSynthesizer:
    def __init__(
        self,
        tools: list[ToolDefinition],
        plan: SynthesisPlan,
        output_dir: Path,
    ):
        self.tools = tools
        self.plan = plan
        self.output_dir = output_dir
        self.seed_generator = SeedGenerator(SeedGeneratorConfig(), tools)
        self.augmenter = DataAugmenter(AugmenterConfig())

    async def synthesize(
        self,
        progress_callback: Callable | None = None,
    ) -> SynthesisResult:
        """Run full synthesis pipeline."""
        # 1. Generate seeds
        # 2. Validate seeds
        # 3. Augment seeds
        # 4. Validate augmented
        # 5. Merge and dedupe
        # 6. Final QC validation
        # 7. Return result
```

**Pipeline Flow**:
```
Tools + Plan
     │
     ▼
┌─────────────┐
│ Generate    │ ──► seed.jsonl
│ Seeds       │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Validate    │ ──► Invalid samples logged
│ Seeds       │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Augment     │ ──► augmented.jsonl
│ (Paraphrase)│
└─────────────┘
     │
     ▼
┌─────────────┐
│ Merge &     │ ──► train.jsonl
│ Dedupe      │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Final QC    │ ──► QC Report
└─────────────┘
```

**Acceptance Criteria**:
- [ ] Orchestrates full synthesis flow
- [ ] Progress reporting at each stage
- [ ] Integrates with existing `DataQualityController`
- [ ] Persists intermediate files (seed.jsonl, augmented.jsonl)
- [ ] Returns `SynthesisResult` with metrics

---

### Task 5: Update Data Module Exports

**File**: `src/mcp_forge/data/__init__.py`

**Update exports**:
```python
from .qc import DataQualityController, QCConfig, QCReport, QCIssue
from .synthesizer import DataSynthesizer, SynthesisResult
from .seed_generator import SeedGenerator, SeedGeneratorConfig
from .augmenter import DataAugmenter, AugmenterConfig
from .formatter import (
    format_system_prompt,
    format_tool_call,
    create_training_sample,
)

__all__ = [
    # QC (existing)
    "DataQualityController",
    "QCConfig",
    "QCReport",
    "QCIssue",
    # Synthesis (new)
    "DataSynthesizer",
    "SynthesisResult",
    "SeedGenerator",
    "SeedGeneratorConfig",
    "DataAugmenter",
    "AugmenterConfig",
    "format_system_prompt",
    "format_tool_call",
    "create_training_sample",
]
```

**Acceptance Criteria**:
- [ ] All new classes exported
- [ ] `from mcp_forge.data import DataSynthesizer` works
- [ ] No circular imports

---

### Task 6: Integrate with CLI Pipeline

**File**: `src/mcp_forge/cli.py`

**Update**: Replace TODO at lines 162-170 with synthesis call.

```python
# Stage 2: Generate data
if state.stage in (PipelineStage.INSPECTING, PipelineStage.SYNTHESIZING):
    console.print("\n[bold]Stage 2: Generating training data...[/bold]")
    state.update_stage(PipelineStage.SYNTHESIZING)
    state_manager.save_state(state)

    # NEW: Actual synthesis implementation
    from mcp_forge.data import DataSynthesizer

    synthesizer = DataSynthesizer(
        tools=state.tools,
        plan=state.synthesis_plan,
        output_dir=state_manager.data_dir,
    )

    with console.status("[bold green]Generating training data..."):
        result = asyncio.run(synthesizer.synthesize(
            progress_callback=lambda msg: console.print(f"   {msg}")
        ))

    state.seed_data_path = str(result.seed_path)
    state.training_data_path = str(result.training_path)

    console.print(f"   [green]✓[/green] Generated {result.seed_count} seed samples")
    console.print(f"   [green]✓[/green] Augmented to {result.total_count} total samples")
```

**Also update `generate` command** (lines 439-446):
```python
@cli.command()
@click.option("--server", "-s", required=True, help="MCP server command")
@click.option("--samples", default=500, help="Total samples to generate")
@click.option("--output", "-o", required=True, type=click.Path())
def generate(server: str, samples: int, output: str):
    """Generate training data from MCP server tools."""
    # Implement standalone generation command
```

**Acceptance Criteria**:
- [ ] `mcp-forge run` executes synthesis stage
- [ ] `mcp-forge generate` works standalone
- [ ] Progress displayed in terminal
- [ ] State updated with data paths

---

### Task 7: Write Unit Tests

**File**: `tests/unit/test_synthesizer.py`

**Test Coverage**:

```python
# Formatter tests
class TestFormatter:
    def test_format_system_prompt_with_tools(self, sample_tools):
        """System prompt includes <tools> XML block."""

    def test_format_tool_call_structure(self):
        """Tool calls wrapped in <tool_call> tags."""

    def test_create_training_sample_standard(self, sample_tools):
        """Standard scenario sample has correct structure."""

    def test_create_training_sample_no_tool(self):
        """No-tool scenario has null tool_name."""

# Seed generator tests (mocked GPT-5)
class TestSeedGenerator:
    @pytest.fixture
    def mock_openai(self, mocker):
        """Mock OpenAI client."""

    def test_generate_seeds_respects_plan(self, mock_openai, sample_tools):
        """Seeds match SynthesisPlan distribution."""

    def test_scenario_prompt_standard(self, sample_tools):
        """Standard scenario prompts for tool use."""

    def test_scenario_prompt_no_tool(self):
        """No-tool scenario prompts for conversation."""

    def test_handles_api_error_with_retry(self, mock_openai):
        """Retries on transient API errors."""

# Augmenter tests
class TestAugmenter:
    def test_expansion_factor(self, mock_openai, valid_training_sample):
        """Augmentation achieves target expansion."""

    def test_prevents_model_collapse(self, mock_openai):
        """Synthetic ratio capped at 30%."""

    def test_deduplicates_similar_samples(self):
        """Near-duplicate augments removed."""

# Integration test
class TestDataSynthesizer:
    def test_full_pipeline(self, mock_openai, sample_tools, tmp_path):
        """Full synthesis pipeline produces valid output."""

    def test_output_passes_qc(self, mock_openai, sample_tools, tmp_path):
        """Synthesized data passes DataQualityController."""
```

**Acceptance Criteria**:
- [ ] 90%+ coverage on new modules
- [ ] All tests pass with mocked OpenAI
- [ ] Integration test validates full flow

---

### Task 8: Write Integration Tests

**File**: `tests/integration/test_synthesis_pipeline.py`

**Tests**:

```python
class TestSynthesisPipeline:
    @pytest.mark.integration
    def test_cli_generate_command(self, tmp_path, mock_mcp_server):
        """CLI generate command produces valid JSONL."""

    @pytest.mark.integration
    def test_pipeline_stage_2(self, tmp_path, mock_mcp_server, mock_openai):
        """Pipeline stage 2 runs synthesis correctly."""

    @pytest.mark.integration
    def test_synthesis_with_real_tools(self, tmp_path, mock_openai):
        """Synthesis works with realistic tool definitions."""
```

**Acceptance Criteria**:
- [ ] Integration tests cover CLI commands
- [ ] Tests use fixtures from Phase 2

---

### Task 9: Update Documentation

**Files to update**:
- `CLAUDE.md` - Update implementation status
- `.planning/STATE.md` - Update phase status

**CLAUDE.md updates**:
```markdown
**Implemented (v1.2):**
- CLI framework with all command stubs
- State management with checkpoint/resume
- MCP inspector module
- Data QC engine with schema validation
- **Data synthesis engine (seed + augmentation)** ← NEW
```

**Acceptance Criteria**:
- [ ] CLAUDE.md reflects new capabilities
- [ ] STATE.md shows Phase 3 complete

---

## Verification

### Test Commands

```bash
# Run all tests
pytest

# Run synthesis tests specifically
pytest tests/unit/test_synthesizer.py -v

# Run with coverage
pytest --cov=mcp_forge.data --cov-report=term-missing

# Type check
mypy src/mcp_forge/data/

# Lint
ruff check src/mcp_forge/data/
```

### Manual Verification

```bash
# Test standalone generate command (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
mcp-forge generate \
  --server "npx -y @modelcontextprotocol/server-filesystem ." \
  --samples 50 \
  --output ./test_data.jsonl

# Verify output with QC
mcp-forge qa --data ./test_data.jsonl --tools ./tools.json
```

---

## Success Criteria

- [ ] `mcp-forge generate --server <cmd> --samples 100` produces JSONL
- [ ] All 5 scenario types represented in output
- [ ] Augmentation expands seed data 10x minimum
- [ ] Synthesis plan persisted in state
- [ ] API call caching reduces redundant requests (bonus)
- [ ] Tests pass with 85%+ coverage on new code
- [ ] `ruff check` and `mypy` pass

---

## Output

**Files Created**:
- `src/mcp_forge/data/formatter.py` (~100 lines)
- `src/mcp_forge/data/seed_generator.py` (~200 lines)
- `src/mcp_forge/data/augmenter.py` (~200 lines)
- `src/mcp_forge/data/synthesizer.py` (~150 lines)
- `tests/unit/test_synthesizer.py` (~300 lines)
- `tests/integration/test_synthesis_pipeline.py` (~100 lines)

**Files Modified**:
- `src/mcp_forge/data/__init__.py`
- `src/mcp_forge/cli.py`
- `CLAUDE.md`
- `.planning/STATE.md`

---

## Dependencies

**Existing** (already in pyproject.toml):
- `openai>=1.0.0`
- `jsonschema`

**Optional** (for caching):
- `diskcache>=5.6.0` - Add if API caching needed

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPT-5 API costs | Use batch API, implement caching, test with small samples |
| Model collapse | Cap synthetic ratio at 30%, aggressive deduplication |
| Schema validation failures | Use `strict: true` in GPT-5, validate early |
| Rate limits | Implement exponential backoff, respect rate limits |

---

## Estimated Task Breakdown

| Task | Scope | Dependencies |
|------|-------|--------------|
| 1. Formatter | Small | None |
| 2. Seed Generator | Medium | Task 1 |
| 3. Augmenter | Medium | Task 1 |
| 4. Orchestrator | Medium | Tasks 2, 3 |
| 5. Module exports | Trivial | Tasks 1-4 |
| 6. CLI integration | Small | Task 4 |
| 7. Unit tests | Medium | Tasks 1-4 |
| 8. Integration tests | Small | Tasks 6, 7 |
| 9. Documentation | Trivial | All above |

**Recommended Execution Order**: 1 → 2 → 3 → 4 → 5 → 7 → 6 → 8 → 9

---

*Plan created: 2026-01-13*
