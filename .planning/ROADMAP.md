# MCP-Forge Roadmap

## Milestone: v1.0 - Full Pipeline Implementation

Transform the existing prototype into a complete, production-ready CLI pipeline for fine-tuning LLMs on MCP server tool schemas.

### Current State

- **Implemented**: CLI framework, state management, MCP inspector (partial), QC engine (unwired)
- **Critical Issues**: Flat file structure (imports fail), no tests, 7/8 pipeline stages are TODO stubs
- **Ready**: pyproject.toml configured, dataclass models defined, checkpoint/resume infrastructure

---

## Phase 1: Foundation & Package Structure

**Goal**: Fix the structural issues so the package installs and imports work correctly.

**Scope**:
- Migrate flat files to `src/mcp_forge/` directory structure
- Create proper `__init__.py` files for all subpackages
- Fix import paths that currently cause `ModuleNotFoundError`
- Wire up existing QC module properly
- Verify `pip install -e .` works and CLI entry points resolve

**Research Needed**: None - structure is documented in TECHNICAL_ARCHITECTURE

**Acceptance Criteria**:
- [ ] `pip install -e .` succeeds without errors
- [ ] `mcp-forge --help` displays available commands
- [ ] `mcp-forge doctor` runs and reports environment status
- [ ] All existing code migrated with no functional changes
- [ ] ruff and mypy pass on migrated code

---

## Phase 2: Test Infrastructure

**Goal**: Establish test foundation before adding new features.

**Scope**:
- Create `tests/` directory structure (unit, integration, fixtures)
- Set up pytest configuration and coverage reporting
- Write tests for existing modules (state, inspector, qc)
- Add test fixtures for MCP server mocking
- Configure CI-ready test execution

**Research Needed**: MCP SDK test patterns, async testing with pytest-asyncio

**Acceptance Criteria**:
- [ ] `pytest` discovers and runs tests
- [ ] Coverage reporting functional (target: 85%+ for existing code)
- [ ] State management fully tested
- [ ] Inspector module tested with mocked MCP server
- [ ] QC engine tested with sample datasets

---

## Phase 3: Data Synthesis Engine

**Goal**: Implement the SYNTHESIZING stage to generate training data.

**Scope**:
- Create `data/synthesizer.py` for GPT-4o seed generation
- Implement 5 scenario types: standard, no_tool, error, ambiguous, edge
- Add Unsloth augmentation for data expansion
- Create synthesis plan configuration
- Implement caching to reduce API costs

**Research Needed**: GPT-4o function calling for seed generation, Unsloth data augmentation patterns

**Acceptance Criteria**:
- [ ] `mcp-forge generate --server <cmd> --samples 100` produces JSONL
- [ ] All 5 scenario types represented in output
- [ ] Augmentation expands seed data 10x minimum
- [ ] Synthesis plan persisted in state
- [ ] API call caching reduces redundant requests

---

## Phase 4: QC Gate Integration

**Goal**: Wire QC validation as mandatory gate before training.

**Scope**:
- Integrate existing `qc.py` into pipeline flow
- Add configurable thresholds (CLI flags + config file)
- Implement `--fix` mode for auto-corrections
- Create detailed QC reports (JSON + terminal output)
- Block training if thresholds not met

**Research Needed**: None - QC engine exists, needs wiring

**Acceptance Criteria**:
- [ ] `mcp-forge qa --data train.jsonl --tools tools.json` validates data
- [ ] Schema pass rate, dedup, coverage metrics calculated
- [ ] QC failure blocks pipeline progression
- [ ] `--fix` mode auto-corrects fixable issues
- [ ] Thresholds configurable via CLI and config file

---

## Phase 5: Training Engine

**Goal**: Implement the TRAINING stage with Unsloth + LoRA fine-tuning.

**Scope**:
- Create `training/engine.py` for Unsloth training wrapper
- Implement training profiles (fast_dev, balanced, max_quality)
- Add DeepSeek-R1-Distill-8B and Qwen-2.5-14B-Instruct support
- Implement checkpoint saving during training
- Add training progress reporting

**Research Needed**: Unsloth API, LoRA configuration for tool-calling, ChatML template handling

**Acceptance Criteria**:
- [ ] `mcp-forge train --data train.jsonl --model deepseek-r1` initiates training
- [ ] Training profiles adjust hyperparameters correctly
- [ ] Both supported models train without errors
- [ ] Training checkpoints saved at configurable intervals
- [ ] Progress displayed in terminal with loss/metrics

---

## Phase 6: Looped Validation

**Goal**: Implement the VALIDATING stage with real/stubbed MCP server testing.

**Scope**:
- Create `validation/looped.py` for validation executor
- Implement deterministic MCP stubs for reproducible testing
- Add validation against real MCP servers
- Create validation result aggregation
- Implement retry logic for flaky connections

**Research Needed**: MCP server stubbing patterns, tool call accuracy measurement

**Acceptance Criteria**:
- [ ] `mcp-forge validate --model ./trained --server <cmd>` runs validation
- [ ] Stubbed validation produces deterministic results
- [ ] Real server validation handles connection failures gracefully
- [ ] Validation results include accuracy, parse rate, schema conformance
- [ ] Results persisted in state for benchmarking

---

## Phase 7: Benchmark Suite

**Goal**: Implement the BENCHMARKING stage with comprehensive metrics.

**Scope**:
- Create `validation/benchmark.py` for evaluation suite
- Implement tool accuracy measurement
- Add end-to-end loop success tracking
- Measure response latency metrics
- Generate benchmark reports (JSON + summary)

**Research Needed**: LLM evaluation metrics for tool calling, latency measurement patterns

**Acceptance Criteria**:
- [ ] `mcp-forge benchmark --model ./trained --server <cmd>` runs benchmarks
- [ ] Tool selection accuracy calculated (target: 90%+)
- [ ] No-tool correctness measured (target: 85%+)
- [ ] Loop completion rate tracked (target: 95%+)
- [ ] Benchmark reports generated and persisted

---

## Phase 8: GGUF Export

**Goal**: Implement the EXPORTING stage to convert fine-tuned models to GGUF.

**Scope**:
- Create `export/gguf.py` for GGUF conversion
- Support multiple quantization levels (Q4_K_M default)
- Implement conversion verification
- Add model metadata embedding
- Create export progress reporting

**Research Needed**: llama.cpp GGUF conversion, quantization quality tradeoffs

**Acceptance Criteria**:
- [ ] `mcp-forge export --model ./trained --output model.gguf` creates GGUF
- [ ] Q4_K_M quantization by default, other levels configurable
- [ ] Exported model loads in llama.cpp/Ollama
- [ ] Model metadata includes training info
- [ ] Export size and quality metrics reported

---

## Phase 9: Bundle Packaging

**Goal**: Implement the PACKAGING stage to create distributable Agent Bundles.

**Scope**:
- Create `export/bundle.py` for bundle assembly
- Generate `tools.json` with schema definitions
- Create `config.yaml` deployment configuration
- Add `README.md` generation for bundle
- Optionally create Ollama `Modelfile`

**Research Needed**: Ollama Modelfile format, deployment configuration best practices

**Acceptance Criteria**:
- [ ] `mcp-forge package --model model.gguf --tools tools.json --output ./dist/agent` creates bundle
- [ ] Bundle contains: model.gguf, tools.json, config.yaml
- [ ] README.md documents usage instructions
- [ ] Modelfile enables `ollama create` workflow
- [ ] Bundle validates before packaging completes

---

## Summary

| Phase | Name | Research | Estimated Complexity |
|-------|------|----------|---------------------|
| 1 | Foundation & Package Structure | None | Medium |
| 2 | Test Infrastructure | Async testing, MCP mocking | Medium |
| 3 | Data Synthesis Engine | GPT-4o, Unsloth augmentation | High |
| 4 | QC Gate Integration | None | Low |
| 5 | Training Engine | Unsloth, LoRA, ChatML | High |
| 6 | Looped Validation | MCP stubbing | Medium |
| 7 | Benchmark Suite | LLM eval metrics | Medium |
| 8 | GGUF Export | llama.cpp conversion | Medium |
| 9 | Bundle Packaging | Ollama Modelfile | Low |

---

*Created: 2026-01-12*
