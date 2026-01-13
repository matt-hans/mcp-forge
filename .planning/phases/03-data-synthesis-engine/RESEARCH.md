# Phase 3 Research: Data Synthesis Engine

**Research Date**: 2026-01-13
**Phase Goal**: Implement SYNTHESIZING stage to generate training data for tool-calling LLMs

---

## Executive Summary

This research covers three domains critical to Phase 3:
1. **GPT-5 Function Calling** - Seed data generation via OpenAI API
2. **Data Augmentation** - Expanding seed data without model collapse
3. **Tool-Calling Training Formats** - Industry-standard data structures

**Key Decisions**:
- Use **GPT-5** (not GPT-4o) - 60% cheaper, better quality
- Use **Hermes ChatML format** - proven for tool-calling fine-tuning
- Augment via **paraphrasing** (not back-translation) - simpler, effective
- Target **~30% synthetic** in final mix to avoid model collapse

---

## 1. GPT-5 for Seed Generation

### Model Selection

| Model | Input Cost | Output Cost | Context | Notes |
|-------|-----------|-------------|---------|-------|
| GPT-5 | $1.25/1M | $10/1M | 400K | **Recommended** - 60% cheaper than 4o |
| GPT-5.2 Pro | $1.75/1M | $14/1M | 400K | Latest, reasoning tokens add cost |
| GPT-4o | $5/1M | $20/1M | 128K | Legacy, more expensive |

**Recommendation**: Use `gpt-5` model ID for seed generation.

### Function Calling Features

GPT-5 includes robust function calling support:

- **Strict Mode** (`strict: true`): Guarantees outputs match JSON schema exactly
- **Structured Outputs**: Built-in JSON schema validation
- **Custom Tools**: Can send raw text (SQL, code) not just JSON
- **Parallel Tool Calling**: Multiple function calls in single response
- **Prompt Caching**: 90% discount on cached tokens ($0.18/1M)

### API Configuration

```python
# Recommended configuration for seed generation
client.chat.completions.create(
    model="gpt-5",
    messages=[...],
    tools=[{
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "...",
            "parameters": {...},
            "strict": True  # ALWAYS enable
        }
    }],
    tool_choice="auto"  # or {"type": "function", "function": {"name": "specific_tool"}}
)
```

### Cost Optimization Strategies

1. **Prompt Caching**: Structure prompts with static system message + dynamic user content
2. **Batch API**: Use for bulk seed generation (lower cost, higher latency)
3. **Temperature Variation**: Generate diverse seeds by varying temperature (0.7-1.0)
4. **Schema Reuse**: Cache tool schemas, don't regenerate per request

**Sources**:
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [GPT-5 Pricing](https://pricepertoken.com/pricing-page/model/openai-gpt-5)
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs)

---

## 2. Training Data Format

### Industry Standard: Hermes ChatML

The **Hermes Function Calling format** is the proven standard for tool-calling fine-tuning. Used by NousResearch, compatible with Unsloth.

#### System Prompt Template

```
<|im_start|>system
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.

<tools>
[{"name": "tool_name", "description": "...", "parameters": {...}}]
</tools>

For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "<function-name>", "arguments": <args-dict>}
</tool_call><|im_end|>
```

#### Conversation Structure

```
<|im_start|>user
What's the weather in San Francisco?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}
</tool_call><|im_end|}
<|im_start|>tool
{"temperature": 18, "condition": "partly cloudy"}<|im_end|>
<|im_start|>assistant
The weather in San Francisco is 18°C and partly cloudy.<|im_end|>
```

#### JSONL Training Format

```json
{
  "id": "sample_001",
  "source": "seed",
  "scenario": "standard",
  "tool_name": "get_weather",
  "messages": [
    {"role": "system", "content": "You are a function calling AI model..."},
    {"role": "user", "content": "What's the weather in San Francisco?"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"San Francisco\"}}\n</tool_call>"},
    {"role": "tool", "content": "{\"temperature\": 18, \"condition\": \"partly cloudy\"}"},
    {"role": "assistant", "content": "The weather in San Francisco is 18°C and partly cloudy."}
  ]
}
```

### Scenario Distribution

Based on ToolACE and BFCL research, recommended distribution:

| Scenario | % | Description |
|----------|---|-------------|
| `standard` | 55% | Normal tool calls with clear intent |
| `no_tool` | 15% | Queries that don't need tools (prevents over-calling) |
| `parallel` | 10% | Multiple tool calls in one response |
| `error` | 10% | Tool errors and graceful handling |
| `ambiguous` | 5% | Unclear intent requiring clarification |
| `edge` | 5% | Edge cases, unusual parameters |

**Critical**: The 15% `no_tool` samples prevent the model from "over-calling" tools on every query.

**Sources**:
- [Hermes Function Calling](https://github.com/NousResearch/Hermes-Function-Calling)
- [ToolACE Paper (ICLR 2025)](https://openreview.net/forum?id=8EB8k6DdCU)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [Unsloth Chat Templates](https://unsloth.ai/docs/basics/chat-templates)

---

## 3. Data Augmentation Strategy

### DO NOT Use Unsloth for Augmentation

**Important Finding**: Unsloth is a **training framework**, not a data augmentation library. It provides:
- Optimized LoRA training (2-5x faster, 80% less VRAM)
- Synthetic data notebooks (for generating from documents)
- Chat template support

Augmentation must be done **separately** before training.

### Recommended: LLM-Based Paraphrasing

Research shows LLM paraphrasing matches or exceeds back-translation quality while being simpler to implement.

#### Paraphrasing Strategy

```python
# Use GPT-5 to paraphrase user queries while preserving tool call intent
PARAPHRASE_PROMPT = """
Rephrase the following user query in a different way while:
1. Preserving the exact intent and required information
2. Using different vocabulary and sentence structure
3. Maintaining natural conversational tone

Original: {original_query}
Rephrased:
"""
```

#### Augmentation Approaches (Ranked)

1. **Query Paraphrasing** - Rephrase user messages, keep tool calls identical
2. **Parameter Variation** - Vary tool arguments (different cities, values, etc.)
3. **Context Addition** - Add conversational context before tool-requiring query
4. **Multi-turn Expansion** - Extend single-turn to multi-turn conversations

### Avoiding Model Collapse

**Critical Research Finding**: Training on >30% synthetic data risks model collapse.

| Synthetic % | Risk | Recommendation |
|-------------|------|----------------|
| 0-20% | Low | Safe zone |
| 20-30% | Medium | Optimal for augmentation |
| 30-50% | High | Monitor quality closely |
| >50% | Very High | Model collapse likely |

**Mitigation Strategies**:
1. **Mix with seed data**: Never train on 100% augmented
2. **Source diversity**: Use multiple paraphrasing strategies
3. **Quality filtering**: Use LLM-as-judge to filter low-quality augments
4. **Deduplication**: Remove near-duplicates aggressively

**Sources**:
- [Synthetic Data Diversity Impact (2025)](https://arxiv.org/abs/2511.01490)
- [Model Collapse in AI (Nature 2024)](https://www.nature.com/articles/s41586-024-07566-y)
- [Data Augmentation Survey 2025](https://link.springer.com/article/10.1007/s10462-025-11405-5)
- [Unsloth Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)

---

## 4. Quality Assurance

### ToolACE Dual-Layer Verification

The ToolACE approach (ICLR 2025) achieved SOTA results using:

1. **Rule-Based Checks**:
   - JSON syntax validation
   - Schema conformance (required fields, types)
   - Tool name existence
   - Parameter completeness

2. **Model-Based Checks**:
   - LLM-as-judge for semantic correctness
   - Intent preservation verification
   - Response coherence scoring

### Recommended QC Pipeline

```
Seed Generation → Schema Validation → Deduplication → Augmentation →
Re-validation → LLM Quality Score → Final Dataset
```

### Quality Thresholds (from existing QC module)

| Metric | Threshold | Source |
|--------|-----------|--------|
| Tool-call parse rate | ≥98% | MCP-Forge spec |
| Schema conformance | ≥95% | MCP-Forge spec |
| Deduplication ratio | <5% duplicate | Best practice |
| LLM quality score | ≥0.8 | ToolACE approach |

**Sources**:
- [ToolACE Verification System](https://openreview.net/forum?id=8EB8k6DdCU)
- [BFCL Evaluation Methodology](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)

---

## 5. Implementation Architecture

### Recommended Module Structure

```
src/mcp_forge/data/
├── __init__.py
├── synthesizer.py     # Main synthesis orchestrator
├── seed_generator.py  # GPT-5 seed generation
├── augmenter.py       # Paraphrasing & augmentation
├── formatter.py       # Hermes ChatML formatting
└── cache.py           # API response caching
```

### Key Dependencies

```toml
# Already in pyproject.toml
openai = ">=1.0.0"

# May need to add
diskcache = ">=5.6.0"  # For API response caching
```

### Synthesis Flow

```
1. Load MCP tool schemas (from inspector)
2. Generate seed samples via GPT-5
   - One prompt per scenario type
   - Batch API for cost efficiency
3. Validate seeds against schemas
4. Augment via paraphrasing (target 10x expansion)
5. Re-validate augmented samples
6. Apply LLM quality scoring
7. Filter and deduplicate
8. Output final JSONL dataset
```

---

## 6. What NOT to Hand-Roll

### Use Existing Solutions

| Component | Use This | Don't Build |
|-----------|----------|-------------|
| API client | `openai` SDK | Custom HTTP |
| Response caching | `diskcache` | File-based cache |
| JSON validation | `jsonschema` | Custom validators |
| Chat templates | Unsloth's `get_chat_template` | Manual formatting |
| Deduplication | MinHash/SimHash | Exact matching only |

### Leverage Existing MCP-Forge Code

- `qc.py` - Already has schema validation, deduplication
- `state.py` - Has `ToolDefinition` dataclass for schemas
- `inspector.py` - Extracts tool schemas from MCP servers

---

## 7. Common Pitfalls to Avoid

### From Research

1. **Over-generating tool calls**: Without 15% no-tool samples, model will call tools unnecessarily
2. **Repetitive augmentation**: Use diverse paraphrasing to avoid training on near-duplicates
3. **Ignoring model collapse**: Keep synthetic data under 30% of total
4. **Skipping validation**: Every sample must pass schema validation before training
5. **Single-turn only**: Include multi-turn conversations for realistic tool use

### Implementation Pitfalls

1. **Not using `strict: true`**: GPT-5 outputs may not match schema without it
2. **Forgetting prompt caching**: Static system prompts should be cached
3. **Serial generation**: Use batch API for large seed generation
4. **No quality filtering**: Use LLM-as-judge to remove low-quality samples

---

## 8. Success Metrics

### Phase 3 Acceptance Criteria Mapping

| Criteria | Research Finding |
|----------|------------------|
| `mcp-forge generate --server <cmd> --samples 100` | Use GPT-5 batch API with Hermes format |
| All 5 scenario types represented | 55/15/10/10/5/5 distribution |
| Augmentation expands seed data 10x | Paraphrasing + parameter variation |
| Synthesis plan persisted in state | Extend `PipelineState` dataclass |
| API call caching reduces costs | Use `diskcache` + prompt caching |

---

## 9. Recommended Next Steps

1. **Plan Phase 3** - Create detailed implementation plan using this research
2. **Prototype seed generation** - Test GPT-5 function calling with one tool
3. **Validate Hermes format** - Ensure Unsloth accepts the training format
4. **Benchmark augmentation** - Compare paraphrasing quality at different temperatures

---

## Sources Summary

### Primary Sources
- [OpenAI Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI GPT-5 Model Documentation](https://platform.openai.com/docs/models/gpt-5)
- [Unsloth Documentation](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Hermes Function Calling Repository](https://github.com/NousResearch/Hermes-Function-Calling)
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)

### Research Papers
- [ToolACE: Winning the Points of LLM Function Calling (ICLR 2025)](https://openreview.net/forum?id=8EB8k6DdCU)
- [Synthetic Eggs in Many Baskets: Synthetic Data Diversity Impact](https://arxiv.org/abs/2511.01490)
- [Model Collapse in AI (Nature 2024)](https://www.nature.com/articles/s41586-024-07566-y)
- [Data Augmentation for LLMs: Comprehensive Survey 2025](https://link.springer.com/article/10.1007/s10462-025-11405-5)
- [Backtranslation and Paraphrasing in the LLM Era](https://arxiv.org/abs/2507.14590)

### Datasets & Benchmarks
- [ToolACE Dataset (HuggingFace)](https://huggingface.co/Team-ACE)
- [BFCL Dataset](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)

---

*Research completed: 2026-01-13*
