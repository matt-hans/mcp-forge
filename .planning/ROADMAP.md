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

## Phase 4: QC Gate Integration ✅

**Status**: Complete (2026-01-13)
**Plans**: 1 | **Commits**: 6 | **Tests**: +57

**Goal**: Wire QC validation as mandatory gate before training.

**Deliverables**:
- ForgeConfig system with YAML support and CLI override merging
- Enhanced qa command with 8 new CLI options
- QCFailedError exception with remediation suggestions
- Repair handlers (truncation, whitespace, encoding)
- Report output in JSON, Markdown, CSV formats
- QC_VALIDATING stage blocks pipeline on failure

**Acceptance Criteria**:
- [x] `mcp-forge qa --data train.jsonl --tools tools.json` validates data
- [x] Schema pass rate, dedup, coverage metrics calculated
- [x] QC failure blocks pipeline progression
- [x] `--fix` mode auto-corrects fixable issues
- [x] Thresholds configurable via CLI and config file

---

## Phase 5: Training Engine ✅

**Status**: Complete (2026-01-13)
**Plans**: 1 | **Commits**: 3 | **Tests**: +39

**Goal**: Implement the TRAINING stage with Unsloth + LoRA fine-tuning.

**Deliverables**:
- Training module structure (config.py, callbacks.py, engine.py)
- TrainingProfile with fast_dev, balanced, max_quality presets
- TrainingEngine with load_model, prepare_dataset, train methods
- ForgeProgressCallback for pipeline state updates
- CLI pipeline Stage 4 integration and standalone train command

**Acceptance Criteria**:
- [x] `mcp-forge train --data train.jsonl --model deepseek-r1` initiates training
- [x] Training profiles adjust hyperparameters correctly
- [x] Both supported models train without errors
- [x] Training checkpoints saved at configurable intervals
- [x] Progress displayed in terminal with loss/metrics

---

## Phase 6: Looped Validation ✅

**Status**: Complete (2026-01-13)
**Plans**: 1 | **Commits**: 8 | **Tests**: +73

**Goal**: Implement the VALIDATING stage with real/stubbed MCP server testing.

**Deliverables**:
- Validation module structure (config.py, stubs.py, runner.py)
- InferenceConfig, StubConfig, ValidationConfig dataclasses
- WeatherStub, FilesystemStub with deterministic responses
- ValidationRunner with model loading and metric aggregation
- CLI validate command and pipeline Stage 5 integration

**Acceptance Criteria**:
- [x] `mcp-forge validate --model ./trained --server <cmd>` runs validation
- [x] Stubbed validation produces deterministic results
- [x] Real server validation handles connection failures gracefully
- [x] Validation results include accuracy, parse rate, schema conformance
- [x] Results persisted in state for benchmarking

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

| Phase | Name | Status | Tests |
|-------|------|--------|-------|
| 1 | Foundation & Package Structure | ✅ Complete | - |
| 2 | Test Infrastructure | ✅ Complete | 103 |
| 3 | Data Synthesis Engine | ✅ Complete | 146 |
| 4 | QC Gate Integration | ✅ Complete | 203 |
| 5 | Training Engine | ✅ Complete | 242 |
| 6 | Looped Validation | ✅ Complete | 315 |
| 7 | Benchmark Suite | Pending | - |
| 8 | GGUF Export | Pending | - |
| 9 | Bundle Packaging | Pending | - |

---

*Created: 2026-01-12 | Updated: 2026-01-13*
