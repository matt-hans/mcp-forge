# Architecture

## System Overview

MCP-Forge is a CLI pipeline that fine-tunes local LLMs on MCP server tool schemas to produce distributable "Agent Bundles" containing a quantized model, tool definitions, and deployment configuration.

## Design Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| Pipeline/State Machine | `state.py:PipelineStage` | 8-stage pipeline with checkpoint/resume |
| Data Transfer Objects | `state.py` dataclasses | Immutable data containers between stages |
| Strategy Pattern | Training profiles | fast_dev, balanced, max_quality configurations |
| Factory Pattern | `StateManager.create_session()` | Session initialization |

## Key Abstractions

### PipelineStage (state.py:22-36)
Enum defining the 8 pipeline stages:
1. IDLE → INSPECTING → SYNTHESIZING → QC_VALIDATING
2. FORMATTING → TRAINING → VALIDATING → BENCHMARKING
3. EXPORTING → PACKAGING → COMPLETE/FAILED

### ToolDefinition (state.py:50-74)
Represents an MCP tool with:
- `name`, `description`, `input_schema`
- `source` tracking (mcp/file/openai)
- `schema_hash()` for drift detection

### PipelineState (state.py:198-337)
Full checkpoint state containing:
- Session metadata (id, stage, timestamps)
- Tool definitions and synthesis plan
- QC reports, validation results, benchmark results
- Training progress and paths

### StateManager (state.py:339-514)
Handles persistence to `.mcp-forge/` directory:
- Atomic save with temp file rename
- Checkpoint/resume support
- Report storage

## Data Flow

```
MCP Server/File → Inspector → ToolDefinitions
                                    ↓
                             Data Synthesizer (GPT-4o seeds + Unsloth augmentation)
                                    ↓
                             QC Validation (schema, dedup, coverage)
                                    ↓
                             Training Engine (Unsloth + LoRA)
                                    ↓
                             Looped Validation (real/stubbed MCP)
                                    ↓
                             Benchmark Suite
                                    ↓
                             GGUF Export + Bundle Packaging
```

## Component Relationships

| Component | Depends On | Provides To |
|-----------|------------|-------------|
| CLI (`cli.py`) | StateManager, Inspector | User interface |
| Inspector (`inspector.py`) | MCP SDK | ToolDefinitions |
| QC Engine (`qc.py`) | ToolDefinitions | Validated samples, QCReport |
| StateManager (`state.py`) | Filesystem | Persistence layer |

## Entry Points

| Entry Point | Location | Purpose |
|-------------|----------|---------|
| `mcp-forge` / `forge` | `cli.py:main()` | CLI entry point |
| `cli.cli()` | `cli.py:48` | Click command group |
| `_run_pipeline()` | `cli.py:141` | Main pipeline executor |

## Concurrency Model

- **Async I/O**: MCP server communication uses asyncio (`inspector.py`)
- **Sync Processing**: QC validation, training are synchronous
- **Timeouts**: 30-second default for MCP operations

## Quality Gates

| Gate | Location | Criteria |
|------|----------|----------|
| QC Validation | Before training | ≥98% schema pass, ≥10 samples/tool |
| Looped Validation | After training | ≥98% parse, ≥95% schema, ≥90% accuracy |
