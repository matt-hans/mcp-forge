# MCP-Forge

**Fine-tune local LLMs on MCP server tool schemas**

MCP-Forge transforms any [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server into a specialized, locally-runnable language model. The pipeline inspects tool schemas, synthesizes high-quality training data, fine-tunes quantized models, validates them against real tool execution, and packages everything for deployment.

## ✨ Key Features

- **🔍 Automatic Tool Discovery**: Connect to any MCP server and extract tool schemas
- **📝 Hybrid Data Synthesis**: GPT-4o seeds + Unsloth augmentation for quality at reasonable cost
- **🔬 Mandatory QA Gate**: Schema validation, deduplication, and coverage analysis
- **🔄 Looped Validation**: Test models against real or stubbed MCP servers
- **📈 Benchmark Suite**: Track metrics across tools, scenarios, and over time
- **📦 Agent Bundles**: Distributable packages with model, tools, and Ollama Modelfile
- **🚀 Local-First**: All training runs on consumer hardware (RTX 3090 or equivalent)

## 🎯 Supported Models

| Model | VRAM (4-bit) | Use Case |
|-------|--------------|----------|
| DeepSeek-R1-Distill-8B | ~6GB | Reasoning-first tool use with `<think>` tokens |
| Qwen-2.5-14B-Instruct | ~9GB | High accuracy, general purpose |

## 📦 Installation

```bash
# Prerequisites
# - NVIDIA GPU with 16GB+ VRAM (24GB recommended)
# - CUDA 12.x
# - Python 3.10+

# Install from source
git clone https://github.com/yourorg/mcp-forge
cd mcp-forge
pip install -e ".[dev]"

# Set OpenAI API key for seed generation
export OPENAI_API_KEY="sk-..."
```

## 🚀 Quick Start

### Full Pipeline (Recommended)

```bash
mcp-forge run \
  --server "npx -y @modelcontextprotocol/server-weather" \
  --model deepseek-r1 \
  --profile balanced \
  --output ./dist/weather-expert
```

This runs the complete pipeline:
1. **Inspect**: Extract tool schemas from MCP server
2. **Generate**: Create 500 training samples (100 seed + 400 augmented)
3. **QA**: Validate data quality and coverage
4. **Train**: Fine-tune with LoRA on your GPU
5. **Validate**: Test against real tool execution
6. **Benchmark**: Measure quality metrics
7. **Export**: Convert to GGUF (Q8_0)
8. **Pack**: Create distributable agent bundle

### Step-by-Step (For Debugging)

```bash
# Check environment
mcp-forge doctor

# Inspect MCP server
mcp-forge tools inspect \
  --server "npx -y @modelcontextprotocol/server-weather" \
  --output tools.json

# Generate training data
mcp-forge generate \
  --tools tools.json \
  --samples 500 \
  --output train.jsonl

# Validate data quality
mcp-forge qa \
  --data train.jsonl \
  --tools tools.json \
  --fix

# Train model
mcp-forge train \
  --data train.jsonl \
  --model deepseek-r1 \
  --profile balanced \
  --output ./output/lora

# Validate with looped execution
mcp-forge validate \
  --model ./output/lora \
  --server "npx -y @modelcontextprotocol/server-weather" \
  --samples 20

# Export to GGUF
mcp-forge export \
  --model ./output/lora \
  --format q8_0 \
  --output ./weather-expert.gguf

# Create agent bundle
mcp-forge pack \
  --model ./weather-expert.gguf \
  --tools tools.json \
  --output ./dist/weather-expert
```

### Deploy with Ollama

```bash
cd ./dist/weather-expert
ollama create weather-expert -f Modelfile
ollama run weather-expert
```

## 📊 Quality Metrics

MCP-Forge measures model quality across multiple dimensions:

| Metric | Target | Description |
|--------|--------|-------------|
| Tool-call validity | ≥98% | Outputs that parse into tool calls |
| Schema conformance | ≥95% | Arguments passing JSON Schema validation |
| Tool selection accuracy | ≥90% | Correct tool chosen for prompt |
| No-tool correctness | ≥85% | Correctly avoids tools when not needed |
| Loop completion rate | ≥95% | Full tool loops completing successfully |

## 🗂️ Agent Bundle Structure

```
dist/weather-expert/
├── model.gguf              # Quantized model
├── tools.json              # Tool definitions
├── prompt_template.json    # System prompt + format
├── agent.yaml              # Metadata manifest
├── tests/validation.json   # Benchmark results
├── README.md               # Usage instructions
└── Modelfile               # Ollama-ready config
```

## 📋 Training Profiles

| Profile | Use Case | Epochs | Batch | LR |
|---------|----------|--------|-------|-----|
| `fast_dev` | Quick iteration | 1 | 4 | 5e-4 |
| `balanced` | Default | 1 | 2 | 2e-4 |
| `max_quality` | Production | 2 | 1 | 1e-4 |

## 🔧 CLI Reference

```
mcp-forge
├── run             # Full pipeline
├── doctor          # Environment check
├── status          # Show pipeline state
│
├── tools
│   ├── inspect     # Extract from MCP server
│   └── import      # Import from file
│
├── generate        # Create training data
├── qa              # Dataset quality analysis
├── train           # Fine-tune model
├── validate        # Looped validation
├── benchmark       # Evaluation suite
│
├── export          # Convert to GGUF
├── pack            # Create agent bundle
└── verify-bundle   # Verify bundle integrity
```

## 🛣️ Roadmap

### v1.1 (Current)
- [x] CLI framework
- [x] State management with checkpointing
- [x] MCP inspector
- [x] Data QC engine
- [ ] Complete data synthesis
- [ ] Looped validation
- [ ] Basic benchmark

### v1.2
- [ ] ToolProvider abstraction
- [ ] File-based tool import
- [ ] OpenAI function format adapter
- [ ] Agent bundle packaging
- [ ] Deterministic stubs for CI

### v1.3
- [ ] Multi-tool chains (optional)
- [ ] Judge scoring
- [ ] GLM-4 support

## 📄 License

MIT

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for fast fine-tuning
- [Model Context Protocol](https://modelcontextprotocol.io) for the standard
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF format
