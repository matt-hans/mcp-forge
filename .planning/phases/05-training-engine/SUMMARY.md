# Phase 5: Training Engine - Summary

**Status**: Complete
**Date**: 2026-01-13

---

## Overview

This phase implemented the TRAINING pipeline stage using Unsloth with LoRA/QLoRA fine-tuning. The training module provides:
1. Profile-based hyperparameter configuration (fast_dev, balanced, max_quality)
2. Model loading with 4-bit quantization via Unsloth
3. SFTTrainer integration with progress callbacks
4. LoRA adapter saving for downstream export

---

## Tasks Completed

| # | Task | Commit |
|---|------|--------|
| 1-5 | Create training module with config, callbacks, engine | `36f37d4` |
| 6 | Wire TrainingEngine into CLI pipeline | `d1cab8d` |
| 7 | Add comprehensive training tests | `fcb69c0` |

---

## Key Deliverables

### 1. Training Module Structure (`src/mcp_forge/training/`)

```
training/
├── __init__.py      # Module exports
├── config.py        # TrainingConfig, TrainingProfile, PROFILES, MODEL_IDS
├── callbacks.py     # ForgeProgressCallback for progress reporting
└── engine.py        # TrainingEngine with load_model, prepare_dataset, train
```

### 2. TrainingConfig and Profiles

**TrainingProfile** dataclass with:
- LoRA parameters: rank, alpha, dropout
- Training parameters: batch size, accumulation, learning rate
- Schedule: epochs vs max_steps, warmup, save frequency

**Predefined Profiles:**

| Profile | LoRA Rank | Alpha | Max Steps | Epochs | Use Case |
|---------|-----------|-------|-----------|--------|----------|
| fast_dev | 8 | 16 | 60 | - | Quick iteration |
| balanced | 16 | 16 | - | 1 | Production training |
| max_quality | 128 | 256 | - | 3 | Maximum accuracy |

**Model Mappings:**

| CLI Value | Unsloth Model ID |
|-----------|------------------|
| deepseek-r1 | unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit |
| qwen-2.5 | unsloth/Qwen2.5-14B-Instruct-bnb-4bit |

### 3. ForgeProgressCallback

- Integrates with HuggingFace TrainerCallback
- Reports step, total steps, progress percentage, loss, epoch
- Updates pipeline state during training
- Handles edge cases (missing loss, non-main process)

### 4. TrainingEngine

**Methods:**
- `load_model()` - Load model with Unsloth 4-bit quantization, apply LoRA
- `prepare_dataset()` - Load JSONL, apply chat template, create HF Dataset
- `train()` - Configure SFTTrainer, run training, save adapter

**Key Configuration:**
- `packing=False` - Critical for tool-calling to preserve conversation boundaries
- `optim="adamw_8bit"` - Memory-efficient optimizer
- `use_gradient_checkpointing="unsloth"` - Memory optimization

### 5. CLI Integration

**Pipeline Stage 4:**
- Loads TrainingConfig from pipeline state
- Creates TrainingEngine with progress callback
- Saves LoRA adapter to `.mcp-forge/adapters/`
- Updates state with adapter path

**Standalone `train` Command:**
```bash
mcp-forge train --data train.jsonl --model deepseek-r1 --profile balanced --output ./adapter
```

---

## Files Changed

**New Files:**
- `src/mcp_forge/training/__init__.py` - Module exports
- `src/mcp_forge/training/config.py` - Configuration and profiles
- `src/mcp_forge/training/callbacks.py` - Progress callback
- `src/mcp_forge/training/engine.py` - Training engine
- `tests/unit/test_training_config.py` - Config unit tests (15)
- `tests/unit/test_training_callbacks.py` - Callback tests (8)
- `tests/integration/test_training_engine.py` - Engine tests (16)

**Modified Files:**
- `src/mcp_forge/cli.py` - Pipeline Stage 4, standalone train command

---

## Test Coverage

| Metric | Before | After |
|--------|--------|-------|
| Tests | 203 | 242 |
| Unit (config) | 0 | 15 |
| Unit (callbacks) | 0 | 8 |
| Integration (engine) | 0 | 16 |

All tests passing. Tests use mocked Unsloth to run without GPU.

---

## Usage Examples

```bash
# Standalone training (requires GPU)
mcp-forge train --data train.jsonl --model deepseek-r1 --profile fast_dev --output ./adapter

# Full pipeline (training happens automatically after QC)
mcp-forge run --server "npx -y @mcp/server" --model deepseek-r1 --output ./bundle

# Check train command options
mcp-forge train --help
```

---

## Technical Notes

### GPU Requirements
- DeepSeek-R1-Distill-8B: ~6GB VRAM
- Qwen-2.5-14B-Instruct: ~9GB VRAM
- Tests mock Unsloth to run without GPU

### LoRA Target Modules
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP/FFN
]
```

### Chat Template
Training uses model's native chat template via `tokenizer.apply_chat_template()`.
This preserves Hermes format from formatter.py while using model-native templates.

---

## Next Phase

Phase 6: Looped Validation
- Validation against real/stubbed MCP servers
- Tool call accuracy measurement
- Loop completion tracking
