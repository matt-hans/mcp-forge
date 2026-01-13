# MCP-Forge

CLI pipeline that fine-tunes local LLMs on Model Context Protocol (MCP) server tool schemas to produce distributable Agent Bundles.

## Vision

Enable developers to create specialized AI agents that reliably interact with MCP servers by fine-tuning open-source models on tool schemas. The agent bundles run locally, require no cloud infrastructure, and produce high-quality tool calls.

## Requirements

### Validated

- State management with checkpoint/resume — `state.py`
- MCP server inspection via async stdio — `inspector.py`
- QC engine with JSON Schema validation and deduplication — `qc.py`
- CLI framework with command groups — `cli.py`
- Training data format (JSONL with 5 scenarios) — documented in architecture

### Active

- [ ] **Structural reorganization**: Migrate from flat files to `src/mcp_forge/` with proper `__init__.py` files
- [ ] **Data synthesis stage**: GPT-4o seed generation + Unsloth augmentation with 5 scenarios (standard, no_tool, error, ambiguous, edge)
- [ ] **Training engine**: Unsloth + LoRA fine-tuning for DeepSeek-R1-Distill-8B and Qwen-2.5-14B-Instruct
- [ ] **Looped validation stage**: Test model against real/stubbed MCP servers
- [ ] **Benchmark suite**: Tool accuracy, end-to-end loop success, and response latency metrics
- [ ] **GGUF export stage**: Convert fine-tuned models to quantized GGUF format
- [ ] **Bundle packaging stage**: Create Agent Bundles (GGUF model + tool definitions JSON + deployment config)
- [ ] **File-based tool import**: Load tool definitions from JSON files (not just MCP servers)
- [ ] **Test suite**: Full coverage (85%+) for all components
- [ ] **Auto-retry logic**: Exponential backoff for failed stages before checkpointing
- [ ] **Configurable QC gates**: CLI flags/config to override threshold defaults

### Out of Scope

- Multi-GPU training — keep single-GPU for v1 simplicity
- Cloud deployment — focus on local developer experience
- Full documentation site — README + CLI help sufficient for v1

## Architecture

### Pipeline Stages

```
1. INSPECTING    → Extract tool schemas from MCP server
2. SYNTHESIZING  → Generate training data (GPT-4o seeds + augmentation)
3. QC_VALIDATING → Validate data quality with mandatory QA gate
4. TRAINING      → Fine-tune with LoRA using Unsloth
5. VALIDATING    → Looped validation against real/stubbed MCP server
6. BENCHMARKING  → Run evaluation suite (accuracy, loop success, latency)
7. EXPORTING     → Convert to GGUF format
8. PACKAGING     → Create Agent Bundle
```

### Target Structure

```
src/
  mcp_forge/
    __init__.py
    cli.py
    state.py
    inspector.py
    data/
      __init__.py
      qc.py
      synthesis.py
    training/
      __init__.py
      engine.py
      profiles.py
    validation/
      __init__.py
      looped.py
      benchmark.py
    export/
      __init__.py
      gguf.py
      bundle.py
tests/
  unit/
  integration/
```

### Supported Models

| Model | VRAM (4-bit) | Template |
|-------|--------------|----------|
| DeepSeek-R1-Distill-8B | ~6GB | ChatML + `<think>` tokens |
| Qwen-2.5-14B-Instruct | ~9GB | ChatML |

### Quality Thresholds (Configurable)

| Metric | Default | Description |
|--------|---------|-------------|
| Schema pass rate | 98% | Tool calls must match JSON schema |
| Min samples/tool | 10 | Minimum training samples per tool |
| Tool selection accuracy | 90% | Correct tool chosen for task |
| No-tool correctness | 85% | Correctly declines when no tool fits |
| Loop completion rate | 95% | Full tool call loop succeeds |

### Agent Bundle Contents

- `model.gguf` — Quantized fine-tuned model
- `tools.json` — Tool definitions with schemas
- `config.yaml` — Deployment configuration

## Constraints

- **Hardware**: Requires NVIDIA GPU with CUDA 12.x, 16GB+ VRAM recommended
- **OS**: Linux (Ubuntu 22.04/24.04) for training
- **Dependencies**: Unsloth, TRL, Transformers, MCP SDK

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Clean break migration | Simpler than preserving git history for structural fix | Pending |
| Support both models equally | Different use cases (reasoning vs general) | Pending |
| Configurable QC thresholds | Different projects have different quality needs | Pending |
| Auto-retry + checkpoint | Resilient pipeline without losing progress | Pending |
| GGUF + tools + config bundle | Minimal viable bundle for local deployment | Pending |

## Success Criteria

1. **Structural soundness**: Package installs and imports work correctly
2. **Full pipeline**: All 8 stages execute end-to-end
3. **Quality output**: Agent Bundles produce accurate tool calls
4. **Test coverage**: 85%+ coverage across all components
5. **Developer UX**: Clear CLI with good error messages

## Risks

| Risk | Mitigation |
|------|------------|
| Training quality varies by model | Benchmark both models, document differences |
| GPT-4o API costs for synthesis | Implement caching, allow seed data reuse |
| VRAM limitations on smaller GPUs | Support 4-bit quantization, document requirements |

---
*Last updated: 2026-01-12 after initialization*
