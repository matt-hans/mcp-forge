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

## Phase 7: Benchmark Suite ✅

**Status**: Complete (2026-01-13)
**Plans**: 1 | **Commits**: 7 | **Tests**: +49

**Goal**: Implement the BENCHMARKING stage with comprehensive metrics.

**Deliverables**:
- BenchmarkConfig with model, samples, thresholds, and baseline comparison
- LatencyStats with min/max/mean/p50/p95/p99 calculations
- BenchmarkRunner with per-tool and per-scenario sample generation
- Latency tracking with warmup sample exclusion
- CLI benchmark command with stub/baseline options
- Pipeline Stage 6 integration with state persistence
- Reports in JSON + Markdown via StateManager

**Acceptance Criteria**:
- [x] `mcp-forge benchmark --model ./trained --server <cmd>` runs benchmarks
- [x] Tool selection accuracy calculated (target: 90%+)
- [x] No-tool correctness measured (target: 85%+)
- [x] Loop completion rate tracked (target: 95%+)
- [x] Benchmark reports generated and persisted

---

## Phase 8: GGUF Export ✅

**Status**: Complete (2026-01-13)
**Plans**: 1 | **Commits**: 7 | **Tests**: +30

**Goal**: Implement the EXPORTING stage to convert fine-tuned models to GGUF.

**Deliverables**:
- Export module structure (config.py, metadata.py, engine.py)
- QuantizationType enum with 5 formats (Q8_0, Q4_K_M, Q4_K_S, Q5_K_M, F16)
- ExportConfig with adapter path, output path, and verification settings
- ExportResult with size/timing metrics and JSON serialization
- GGUFMetadata with MCP-Forge custom fields (tools, training, quality)
- ExportEngine with merge_adapter, convert_to_gguf, verify_gguf methods
- CLI export command with format selection and verification skip option
- Pipeline Stage 7 integration with state persistence

**Acceptance Criteria**:
- [x] `mcp-forge export --model ./trained --output model.gguf` creates GGUF
- [x] Q8_0 quantization by default, 5 levels configurable
- [x] Exported model verified to load via llama-cpp-python
- [x] Model metadata includes training info and tool signatures
- [x] Export size and quality metrics reported

---

## Phase 9: Bundle Packaging ✅

**Status**: Complete (2026-01-13)
**Plans**: 1 | **Commits**: 5 | **Tests**: +63

**Goal**: Implement the PACKAGING stage to create distributable Agent Bundles.

**Deliverables**:
- BundleConfig dataclass with GGUF path, tools, output, and optional metrics
- BundleResult dataclass with success status, files, size, and validation
- BundleEngine with package(), _generate_tools_json(), _generate_config_yaml()
- README.md generation with tool table and quality metrics
- Ollama Modelfile generation with ChatML template
- Bundle validation with file integrity and cross-validation checks
- CLI pack command with tools from file or pipeline state
- CLI verify-bundle command with optional smoke test
- Pipeline Stage 8 integration with state persistence

**Acceptance Criteria**:
- [x] `mcp-forge pack --model model.gguf --tools tools.json --output ./dist/agent` creates bundle
- [x] Bundle contains: model.gguf, tools.json, config.yaml
- [x] README.md documents usage instructions
- [x] Modelfile enables `ollama create` workflow
- [x] Bundle validates before packaging completes

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
| 7 | Benchmark Suite | ✅ Complete | 364 |
| 8 | GGUF Export | ✅ Complete | 394 |
| 9 | Bundle Packaging | ✅ Complete | 466 |

**🎉 Milestone v1.0 Complete!**

All 9 phases implemented. The full pipeline is now functional:
1. INSPECTING → 2. SYNTHESIZING → 3. QC_VALIDATING → 4. TRAINING →
5. VALIDATING → 6. BENCHMARKING → 7. EXPORTING → 8. PACKAGING → COMPLETE

---

*Created: 2026-01-12 | Updated: 2026-01-13 | v1.0 Complete: 2026-01-13*
