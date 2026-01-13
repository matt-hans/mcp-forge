# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP-Forge is a CLI pipeline that fine-tunes local LLMs on Model Context Protocol (MCP) server tool schemas. It produces distributable "Agent Bundles" containing a quantized model, tool definitions, and deployment configuration.

## Build and Development Commands

```bash
# Install from source (requires CUDA 12.x and 16GB+ VRAM)
pip install -e ".[dev]"

# Run linting
ruff check .

# Run type checking
mypy .

# Run tests
pytest

# Run a single test
pytest tests/test_module.py::test_function -v
```

## CLI Commands

The entry point is `mcp-forge` (or `forge`). Key commands:

```bash
# Check environment (CUDA, VRAM, dependencies)
mcp-forge doctor

# Full pipeline
mcp-forge run --server "npx -y @mcp/server-weather" --model deepseek-r1 --output ./dist/agent

# Inspect MCP server tools
mcp-forge tools inspect --server "npx -y @mcp/server" --output tools.json

# Run dataset QA
mcp-forge qa --data train.jsonl --tools tools.json --fix

# Show pipeline state
mcp-forge status
```

## Architecture

### Current File Structure (Flat, Pre-Modularization)

The codebase is currently flat in the root directory, with planned migration to `src/mcp_forge/`:

- `cli.py` - Click-based CLI entry point with all commands defined
- `state.py` - Pipeline state management with checkpoint/resume support, dataclasses for `PipelineState`, `ToolDefinition`, `QCReport`, `ValidationResult`, `BenchmarkResult`
- `inspector.py` - MCP server connection and tool schema extraction using `mcp` SDK
- `qc.py` - Data quality controller with schema validation, deduplication, coverage analysis

### Pipeline Stages

The pipeline executes these stages in order (defined in `PipelineStage` enum):

1. **INSPECTING** - Extract tool schemas from MCP server
2. **SYNTHESIZING** - Generate training data (GPT-4o seeds + Unsloth augmentation)
3. **QC_VALIDATING** - Validate data quality with mandatory QA gate
4. **TRAINING** - Fine-tune with LoRA using Unsloth
5. **VALIDATING** - Looped validation against real/stubbed MCP server
6. **BENCHMARKING** - Run evaluation suite
7. **EXPORTING** - Convert to GGUF format
8. **PACKAGING** - Create agent bundle

### Key Abstractions

- **ToolDefinition** (`state.py:50`) - Represents an MCP tool with name, description, input_schema
- **PipelineState** (`state.py:198`) - Full checkpoint state with tools, QC reports, validation results
- **StateManager** (`state.py:339`) - Handles persistence to `.mcp-forge/` directory
- **DataQualityController** (`qc.py:89`) - Validates training samples against tool schemas

### Training Data Format

JSONL with conversation samples containing:
- `id`, `source` (seed/augmented), `scenario` (standard/no_tool/error/ambiguous/edge)
- `tool_name`, `messages` array with system/user/assistant/tool roles
- Scenarios ensure the model doesn't "over-call" tools (15% no-tool samples)

### State Directory Structure

```
.mcp-forge/
├── state.json      # Pipeline checkpoint
├── data/           # Training data files
├── logs/           # Pipeline logs
└── reports/        # QC and benchmark reports
```

## Implementation Status

**Implemented (v1.1):**
- CLI framework with all command stubs
- State management with checkpoint/resume
- MCP inspector module
- Data QC engine with schema validation

**Not Yet Implemented:**
- Data synthesis (generate command)
- Training engine
- Looped validation
- Benchmark suite
- GGUF export
- Bundle packaging
- File-based tool import (v1.2)

## Supported Models

| Model | VRAM (4-bit) | Template |
|-------|--------------|----------|
| DeepSeek-R1-Distill-8B | ~6GB | ChatML + `<think>` tokens |
| Qwen-2.5-14B-Instruct | ~9GB | ChatML |

## Quality Thresholds

- Tool-call parse rate: ≥98%
- Schema conformance: ≥95%
- Tool selection accuracy: ≥90%
- No-tool correctness: ≥85%
- Loop completion rate: ≥95%
