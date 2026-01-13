# Integrations

## External Services

### MCP Servers (Primary)

| Aspect | Details |
|--------|---------|
| Protocol | Model Context Protocol (stdio JSON-RPC) |
| SDK | `mcp` package (>=1.0.0) |
| Connection | `mcp.client.stdio.stdio_client()` |
| Operations | `tools/list`, tool execution |
| Timeout | 30 seconds (configurable) |

Implementation in `inspector.py:41-96`:
```python
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools_response = await session.list_tools()
```

### OpenAI API

| Aspect | Details |
|--------|---------|
| Model | GPT-4o for seed generation |
| SDK | `openai` package (>=1.0.0) |
| Auth | `OPENAI_API_KEY` environment variable |
| Usage | Generate 100 seed samples, ~$0.50-2 total cost |
| Status | **Not yet implemented** |

### Hugging Face Hub

| Aspect | Details |
|--------|---------|
| Purpose | Base model downloads |
| Auth | `HF_TOKEN` (optional, for gated models) |
| Models | DeepSeek-R1-Distill-8B, Qwen-2.5-14B |
| Status | **Not yet implemented** (training not implemented) |

## File I/O Patterns

### State Persistence

Location: `.mcp-forge/state.json`

```python
# Atomic write pattern (state.py:389-398)
temp_file = self.state_file.with_suffix(".tmp")
with open(temp_file, "w") as f:
    json.dump(state.to_dict(), f, indent=2)
temp_file.rename(self.state_file)  # Atomic on POSIX
```

### Training Data

| File | Format | Purpose |
|------|--------|---------|
| `seed_raw.jsonl` | JSONL | GPT-4o generated seeds |
| `train.jsonl` | JSONL | Validated training data |
| `tools.json` | JSON | Tool definitions |

### Reports

| File | Format | Purpose |
|------|--------|---------|
| `qc_*.json` | JSON | QC analysis results |
| `benchmark_*.json` | JSON | Benchmark metrics |
| `benchmark_*.md` | Markdown | Human-readable report |

## Third-Party Tool Integrations

### Unsloth

| Aspect | Details |
|--------|---------|
| Purpose | Fast LoRA fine-tuning (2x faster, 70% less VRAM) |
| Features | 4-bit quant, `add_new_tokens()`, SyntheticDataKit |
| Status | **Not yet implemented** |

### llama.cpp / Ollama

| Aspect | Details |
|--------|---------|
| Purpose | GGUF export and deployment |
| Formats | Q8_0, Q4_K_M |
| Output | Modelfile for Ollama deployment |
| Status | **Not yet implemented** |

## Network Protocols

| Protocol | Usage | Implementation |
|----------|-------|----------------|
| stdio JSON-RPC | MCP communication | `mcp` SDK |
| HTTPS REST | OpenAI API, HF downloads | `openai`, `transformers` |

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes* | Seed data generation |
| `HF_TOKEN` | No | Gated model access |
| `MCP_FORGE_CONFIG` | No | Custom config path |
| `MCP_FORGE_CACHE_DIR` | No | Cache directory |
| `CUDA_VISIBLE_DEVICES` | No | GPU selection |

## Integration Status

| Integration | Status | Location |
|-------------|--------|----------|
| MCP Inspector | Implemented | `inspector.py` |
| OpenAI Synthesis | Not implemented | — |
| Unsloth Training | Not implemented | — |
| GGUF Export | Not implemented | — |
| Ollama Deployment | Not implemented | — |
| File-based Tools | Not implemented | v1.2 planned |
