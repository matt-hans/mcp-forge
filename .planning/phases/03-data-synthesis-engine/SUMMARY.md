# Phase 3: Data Synthesis Engine - Summary

**Completed**: 2026-01-13
**Duration**: Single session
**Outcome**: SUCCESS

---

## What Was Built

### New Modules

1. **formatter.py** (~210 lines)
   - `format_system_prompt()` - Hermes-style system prompt with `<tools>` XML block
   - `format_tool_call()` - Wrap tool calls in `<tool_call>` XML tags
   - `format_tool_response()` - Format tool responses
   - `create_training_sample()` - Create JSONL-ready samples
   - `parse_tool_call()` - Extract tool calls from content
   - `validate_sample_format()` - Validate sample structure

2. **seed_generator.py** (~380 lines)
   - `SeedGeneratorConfig` - Model, temperature, retry settings
   - `SeedGenerator` - GPT-based seed sample generation
   - Scenario-specific prompts for all 5 scenario types
   - Async generation with retry logic and rate limit handling
   - JSON parsing with fallback for malformed responses

3. **augmenter.py** (~480 lines)
   - `AugmenterConfig` - Expansion factor, temperature range, synthetic ratio cap
   - `DataAugmenter` - LLM paraphrasing and parameter variation
   - Model collapse prevention (max 30% synthetic ratio)
   - Content-based deduplication

4. **synthesizer.py** (~280 lines)
   - `DataSynthesizer` - Main orchestrator class
   - `SynthesisResult` - Result dataclass with metrics
   - 5-stage pipeline: generate → validate → augment → merge → final QC
   - Progress callbacks at each stage

### Updated Modules

- **data/__init__.py** - Added exports for all new classes and functions
- **data/qc.py** - Added `<tool_call>` XML tag support for Hermes format
- **cli.py** - Integrated synthesis into pipeline Stage 2 and `generate` command
- **pyproject.toml** - Added `allow-direct-references = true` for hatch build

### Test Coverage

- **test_synthesizer.py** - 33 unit tests
  - 17 formatter tests
  - 7 seed generator tests (mocked)
  - 5 augmenter tests (mocked)
  - 4 synthesizer tests

- **test_synthesis_pipeline.py** - 10 integration tests
  - CLI generate command tests
  - Pipeline integration tests
  - QA command with synthesized data
  - End-to-end flow tests

**Total tests**: 146 passing

---

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 982d443 | feat | Add Hermes ChatML formatter |
| 02cfe0d | feat | Add GPT-5 seed generator |
| 2febb84 | feat | Add paraphrase augmenter |
| 6494eb7 | feat | Add main synthesizer orchestrator |
| c3c71b0 | chore | Update data module exports |
| cd7a8a0 | feat | Integrate synthesis with CLI pipeline |
| 8561274 | test | Add synthesizer unit tests |
| 826e621 | test | Add synthesis integration tests |

---

## Key Decisions

1. **Hermes ChatML Format**
   - Chose Hermes function-calling format over OpenAI format
   - Tool calls in `<tool_call>` XML tags
   - System prompt with `<tools>` XML block

2. **GPT-4o for Seeds** (GPT-5 placeholder)
   - Using `gpt-4o` model ID (GPT-5 when available)
   - Strict mode enabled for schema conformance
   - Temperature 0.8 for diversity

3. **Model Collapse Prevention**
   - Max 30% synthetic ratio enforced
   - Content-based deduplication via SHA256 hashing
   - Quality filtering at each stage

4. **Scenario Distribution**
   - standard: 60%
   - no_tool: 15%
   - error: 10%
   - ambiguous: 10%
   - edge: 5%

---

## API Dependencies

- **OpenAI API** - Required for seed generation and augmentation
  - Requires `OPENAI_API_KEY` environment variable
  - Uses Chat Completions API with function calling

---

## Usage

### CLI Generate Command

```bash
# Generate from MCP server
mcp-forge generate --server "npx -y @mcp/server-weather" --samples 500 --output train.jsonl

# Generate from tools file
mcp-forge generate --tools tools.json --samples 500 --output train.jsonl
```

### Pipeline Integration

Stage 2 of `mcp-forge run` now automatically:
1. Generates seed samples using GPT-5
2. Validates seeds with QC
3. Augments via paraphrasing (10x expansion target)
4. Merges and deduplicates
5. Runs final QC validation

---

## Next Phase

Phase 4: QC Gate Integration
- Integrate QC as mandatory gate in pipeline
- Fail pipeline if thresholds not met
- Generate detailed QC reports

---

*Summary created: 2026-01-13*
