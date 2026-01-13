# Phase 4: QC Gate Integration

## Objective

Wire the existing `DataQualityController` into the pipeline as a mandatory gate that blocks training when quality thresholds are not met. Implement configurable thresholds via CLI flags and config file, and add a `--fix` mode for auto-corrections.

## Execution Context

**Prerequisites**:
- Phase 3 complete (Data Synthesis Engine implemented)
- QC engine exists at `src/mcp_forge/data/qc.py` with full validation logic
- Unit tests for QC exist at `tests/unit/test_qc.py`

**Key Files**:
- `src/mcp_forge/data/qc.py` - DataQualityController (existing, needs minor enhancements)
- `src/mcp_forge/cli.py` - CLI commands (qa command exists, pipeline integration needed)
- `src/mcp_forge/state.py` - QCReport and related state structures (existing)
- `pyproject.toml` - No changes needed

**Constraints**:
- No breaking changes to existing QC API
- Maintain backward compatibility with `mcp-forge qa` standalone command
- Thresholds must be configurable at runtime (CLI > config > defaults)

---

## Tasks

### Task 1: Add QC Configuration File Support

**Goal**: Enable loading QC thresholds from `.mcp-forge/config.yaml`

**Implementation**:
1. Create `src/mcp_forge/config.py` module
2. Define `ForgeConfig` dataclass with QC threshold fields:
   - `qc_schema_pass_threshold: float = 0.98`
   - `qc_min_samples_per_tool: int = 10`
   - `qc_dedup_enabled: bool = True`
   - `qc_auto_repair: bool = True`
   - `qc_require_scenario_coverage: bool = True`
3. Implement `load_config(path: Path | None = None) -> ForgeConfig`
4. Support YAML format with sensible defaults
5. Add config file discovery (check `.mcp-forge/config.yaml`, then `~/.config/mcp-forge/config.yaml`)

**Tests**:
- `tests/unit/test_config.py`:
  - Test default config values
  - Test loading from YAML file
  - Test config override hierarchy
  - Test missing file returns defaults

**Verification**: `pytest tests/unit/test_config.py -v`

---

### Task 2: Add CLI Threshold Options to QA Command

**Goal**: Allow CLI override of QC thresholds

**Implementation**:
1. Add options to `qa` command in `cli.py`:
   - `--threshold` / `-T` (schema pass rate, default 0.98)
   - `--min-samples` (minimum samples per tool, default 10)
   - `--no-dedup` (disable deduplication)
   - `--no-auto-repair` (disable auto-repair)
   - `--strict` (fail on any warning, not just errors)
2. Wire CLI options to `QCConfig` creation
3. Implement threshold precedence: CLI > config file > defaults

**Tests**:
- Update `tests/integration/test_cli.py`:
  - Test `qa` command with custom thresholds
  - Test `--strict` mode behavior
  - Test threshold override hierarchy

**Verification**: `mcp-forge qa --help` shows new options

---

### Task 3: Implement Blocking Gate in Pipeline

**Goal**: QC validation blocks pipeline progression when thresholds not met

**Implementation**:
1. Update `_run_pipeline()` in `cli.py` to implement QC_VALIDATING stage:
   - Load synthesized data from `state.training_data_path`
   - Run `DataQualityController.validate_dataset()`
   - Store `QCReport` in pipeline state
   - Check `report.passes_threshold()` - if fails, raise `QCFailedError`
2. Create custom `QCFailedError` exception in `data/qc.py`
3. Update pipeline error handling to provide actionable remediation:
   - Print which thresholds failed
   - Suggest `--fix` rerun or threshold adjustment
   - Show sample of issues

**Tests**:
- `tests/integration/test_qc_gate.py`:
  - Test pipeline stops on QC failure
  - Test pipeline continues on QC pass
  - Test error message contains remediation guidance
  - Test QC report is saved to state

**Verification**: Pipeline with bad data stops at QC stage

---

### Task 4: Enhance Fix Mode with Issue-Specific Repairs

**Goal**: Expand `--fix` mode to handle more issue types

**Implementation**:
1. Add repair handlers to `DataQualityController`:
   - `_repair_truncated_response()` - truncate overly long responses
   - `_repair_whitespace()` - normalize whitespace in messages
   - `_repair_encoding()` - fix common encoding issues
2. Track repair statistics in QCReport:
   - Add `repairs_attempted: int`
   - Add `repairs_successful: int`
   - Add `repair_details: list[dict]`
3. Update `validate_dataset()` to optionally rewrite repaired samples
4. Add `--dry-run` option to preview repairs without writing

**Tests**:
- `tests/unit/test_qc.py`:
  - Test whitespace repair
  - Test truncation repair
  - Test repair statistics tracking
  - Test `--dry-run` doesn't modify files

**Verification**: `mcp-forge qa --data bad.jsonl --tools tools.json --fix` repairs issues

---

### Task 5: Add JSON Report Output

**Goal**: Generate machine-readable QC report alongside terminal output

**Implementation**:
1. Add `--report` / `-r` option to `qa` command (output path)
2. Enhance `StateManager.save_qc_report()` to support custom paths
3. Include in report:
   - Timestamp and version
   - All metrics (pass rates, coverage, etc.)
   - Full issue list with sample IDs
   - Repair summary (if fix mode)
   - Pass/fail verdict with threshold comparison
4. Add `--format` option: `json` (default), `markdown`, `csv`

**Tests**:
- `tests/unit/test_qc.py`:
  - Test JSON report structure
  - Test markdown report generation
  - Test CSV format for issue export

**Verification**: `mcp-forge qa --data train.jsonl --tools tools.json --report qc.json`

---

### Task 6: Integration Test for Full QC Flow

**Goal**: End-to-end test of QC gate in pipeline

**Implementation**:
1. Create `tests/integration/test_qc_gate.py` with:
   - Test fixture generating intentionally bad data
   - Test fixture generating good data at threshold boundary
   - Test full pipeline run with QC gate (mocked training)
   - Test `--fix` mode in pipeline context
2. Verify state persistence after QC stage
3. Verify QC report saved to `.mcp-forge/reports/`

**Tests**:
- `test_qc_blocks_bad_data` - Pipeline stops on validation failure
- `test_qc_passes_good_data` - Pipeline continues on valid data
- `test_qc_fix_mode_repairs` - Fix mode repairs and continues
- `test_qc_report_persisted` - Report saved to correct location

**Verification**: `pytest tests/integration/test_qc_gate.py -v`

---

## Verification

Run full test suite:
```bash
pytest tests/ -v --cov=mcp_forge --cov-report=term-missing
```

Manual verification:
```bash
# Test standalone QA command
mcp-forge qa --data tests/fixtures/sample_data.jsonl --tools tests/fixtures/sample_tools.json

# Test with custom threshold
mcp-forge qa --data tests/fixtures/sample_data.jsonl --tools tests/fixtures/sample_tools.json --threshold 0.90

# Test fix mode
mcp-forge qa --data tests/fixtures/sample_data.jsonl --tools tests/fixtures/sample_tools.json --fix

# Test JSON report output
mcp-forge qa --data tests/fixtures/sample_data.jsonl --tools tests/fixtures/sample_tools.json --report qc_report.json
```

---

## Success Criteria

- [ ] `mcp-forge qa --data train.jsonl --tools tools.json` validates data and prints report
- [ ] Schema pass rate, dedup, coverage metrics calculated correctly
- [ ] QC failure blocks pipeline progression with clear error message
- [ ] `--fix` mode auto-corrects fixable issues and rewrites cleaned data
- [ ] Thresholds configurable via CLI flags and config file
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage maintained at 72%+ (deferred issue from Phase 2)

---

## Output

**New Files**:
- `src/mcp_forge/config.py` - Configuration loading
- `tests/unit/test_config.py` - Config tests
- `tests/integration/test_qc_gate.py` - QC gate integration tests

**Modified Files**:
- `src/mcp_forge/data/qc.py` - Enhanced repair logic, QCFailedError
- `src/mcp_forge/cli.py` - CLI options, pipeline QC integration
- `tests/unit/test_qc.py` - Additional repair tests

---

## Scope Estimate

**Complexity**: Low (roadmap estimate confirmed)
**Tasks**: 6
**Test coverage**: Unit + Integration

This phase wires existing functionality rather than building from scratch. The DataQualityController is already implemented with validation logic, deduplication, and schema checking. The work is primarily integration (CLI, pipeline, config).
