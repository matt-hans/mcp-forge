# Phase 5: Training Engine - Research

**Prepared**: 2026-01-13
**Domain**: LLM Fine-Tuning with Unsloth + LoRA for Tool Calling

---

## Executive Summary

Phase 5 implements the TRAINING stage using Unsloth with LoRA/QLoRA fine-tuning. The research confirms:

1. **Unsloth is the right choice** - 2x faster training, 70% less VRAM, fully HuggingFace compatible
2. **SFTTrainer integration is straightforward** - Pass Unsloth model directly to TRL's SFTTrainer
3. **Hermes format is already implemented** - Our `formatter.py` produces correct `<tool_call>` XML format
4. **Training profiles map to hyperparameters** - Standard LoRA ranks (8/16/128) for fast_dev/balanced/max_quality

**What NOT to hand-roll**:
- Model quantization (use Unsloth's `load_in_4bit`)
- LoRA injection (use `FastLanguageModel.get_peft_model()`)
- Training loop (use SFTTrainer from TRL)
- Chat template formatting (use `tokenizer.apply_chat_template()`)

---

## 1. Unsloth Library Architecture

### 1.1 Core API

```python
from unsloth import FastLanguageModel

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
    max_seq_length=2048,
    dtype=None,  # Auto-detect (bfloat16 on Ampere+)
    load_in_4bit=True,  # QLoRA mode
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Memory optimization
    random_state=3407,
)
```

### 1.2 Key Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `max_seq_length` | Context window | 2048 (tool calls are verbose) |
| `load_in_4bit` | QLoRA quantization | `True` for consumer GPUs |
| `dtype` | Compute precision | `None` (auto-detect) |
| `use_gradient_checkpointing` | Memory optimization | `"unsloth"` |

### 1.3 Supported Models for MCP-Forge

| Model | Unsloth ID | VRAM (4-bit) | Notes |
|-------|------------|--------------|-------|
| DeepSeek-R1-Distill-8B | `unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit` | ~6GB | Has `<think>` reasoning tokens |
| Qwen-2.5-14B-Instruct | `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` | ~9GB | Native tool_call support |

**Sources**:
- [Unsloth Documentation](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Unsloth DeepSeek R1 Blog](https://unsloth.ai/blog/deepseek-r1)

---

## 2. LoRA Hyperparameter Configuration

### 2.1 Recommended Configurations by Profile

| Profile | Rank (r) | Alpha | Dropout | Target Modules | Use Case |
|---------|----------|-------|---------|----------------|----------|
| `fast_dev` | 8 | 16 | 0 | All linear | Quick iteration, testing |
| `balanced` | 16 | 16 | 0 | All linear | Production-ready training |
| `max_quality` | 128 | 256 | 0.05 | All linear | Maximum accuracy, longer train |

### 2.2 Target Modules

Research confirms targeting **all linear layers** is crucial for matching full fine-tuning performance:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP/FFN
]
```

### 2.3 Training Arguments by Profile

```python
TRAINING_PROFILES = {
    "fast_dev": {
        "r": 8,
        "lora_alpha": 16,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_steps": 60,
        "learning_rate": 2e-4,
        "warmup_steps": 5,
    },
    "balanced": {
        "r": 16,
        "lora_alpha": 16,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 1,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
    },
    "max_quality": {
        "r": 128,
        "lora_alpha": 256,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.05,
    },
}
```

### 2.4 Overfitting Indicators

- **Training loss 0.5-1.0**: Healthy learning
- **Training loss → 0**: Overfitting (memorizing data)
- **Validation loss increasing**: Stop training (early stopping)

**Sources**:
- [Unsloth LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [HuggingFace TRL Blog](https://huggingface.co/blog/unsloth-trl)

---

## 3. Training Data Format (Hermes ChatML)

### 3.1 Current Implementation

Our `formatter.py` already produces the correct Hermes format:

```json
{
  "id": "sample_001",
  "source": "seed",
  "scenario": "standard",
  "tool_name": "get_weather",
  "messages": [
    {"role": "system", "content": "<tools>...</tools>"},
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n</tool_call>"},
    {"role": "tool", "content": "<tool_response>{\"temp\": 22}</tool_response>"},
    {"role": "assistant", "content": "The weather in Tokyo is 22°C."}
  ]
}
```

### 3.2 Chat Template Integration

Unsloth supports multiple templates. For Qwen models:

```python
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5",  # or "chatml" for Hermes-style
    mapping={"role": "role", "content": "content"},
)
```

### 3.3 Dataset Formatting Function

```python
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}
```

### 3.4 Compatibility Note

**Qwen 2.5 models have native Hermes-style tool_call support** in their tokenizer_config.json. The `hermes` parser in vLLM works directly with Qwen models.

**Sources**:
- [Hermes-3 Model Card](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)
- [Hermes Function Calling Dataset](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- [Unsloth Chat Templates](https://docs.unsloth.ai/basics/chat-templates)

---

## 4. SFTTrainer Integration

### 4.1 Basic Setup

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,  # Don't pack for tool-calling (preserve conversation boundaries)
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="steps",
        save_steps=50,
    ),
)
```

### 4.2 Important SFTTrainer Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `packing` | `False` | Preserve conversation boundaries for tool calls |
| `dataset_text_field` | `"text"` | Field containing formatted conversation |
| `max_seq_length` | 2048 | Tool schemas are verbose |
| `optim` | `"adamw_8bit"` | Memory efficient optimizer |

**Sources**:
- [TRL SFTTrainer Documentation](https://huggingface.co/docs/trl/sft_trainer)
- [HuggingFace Unsloth Integration](https://huggingface.co/blog/unsloth-trl)

---

## 5. Checkpoint Management

### 5.1 Saving Checkpoints

```python
# In TrainingArguments
args = TrainingArguments(
    output_dir="outputs",
    save_strategy="steps",  # or "epoch"
    save_steps=50,
    save_total_limit=3,  # Keep only 3 most recent
)
```

### 5.2 Saving LoRA Adapter After Training

```python
# Save LoRA adapter only (small ~100MB file)
model.save_pretrained("lora_adapter")
tokenizer.save_pretrained("lora_adapter")

# Or save merged model (full size)
model.save_pretrained_merged("merged_model", tokenizer)
```

### 5.3 Resuming Training

```python
# Load existing LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./lora_adapter",  # Path to saved adapter
    max_seq_length=2048,
    load_in_4bit=True,
)

# Resume training with SFTTrainer
trainer.train(resume_from_checkpoint="./outputs/checkpoint-100")
```

**Sources**:
- [Unsloth Checkpoint Documentation](https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint)
- [GitHub Issue #591](https://github.com/unslothai/unsloth/issues/591)

---

## 6. Progress Reporting

### 6.1 Custom TrainerCallback

```python
from transformers import TrainerCallback

class ForgeProgressCallback(TrainerCallback):
    def __init__(self, progress_callback):
        self.progress_callback = progress_callback

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_local_process_zero:
            loss = logs.get("loss", None)
            step = state.global_step
            total = state.max_steps
            progress = step / total if total > 0 else 0

            self.progress_callback(
                step=step,
                total_steps=total,
                progress=progress,
                loss=loss,
                epoch=state.epoch,
            )
```

### 6.2 TrainerState Fields

Key fields available in callbacks:

- `state.global_step`: Current training step
- `state.max_steps`: Total steps
- `state.epoch`: Current epoch (float)
- `state.log_history`: List of all logged metrics
- `logs["loss"]`: Current training loss (in `on_log`)

### 6.3 Integration with Pipeline State

```python
# Update pipeline state during training
def update_forge_state(step, total_steps, progress, loss, epoch):
    state.training_progress = progress
    state.training_loss = loss
    state_manager.save_state(state)
```

**Sources**:
- [HuggingFace TrainerCallback Docs](https://huggingface.co/docs/transformers/main_classes/callback)

---

## 7. Model-Specific Considerations

### 7.1 DeepSeek-R1-Distill-8B

**Special Features**:
- Uses `<think>` tokens for chain-of-thought reasoning
- Chat template: `<｜User｜>` and `<｜Assistant｜>` tokens
- Based on Llama architecture

**Training Considerations**:
- Preserve `<think>` reasoning in training data for reasoning tasks
- For pure tool-calling, can omit `<think>` blocks

```python
# DeepSeek R1 uses standard Llama tokenizer
model_id = "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit"
```

### 7.2 Qwen-2.5-14B-Instruct

**Special Features**:
- Native tool_call support in tokenizer_config.json
- ChatML format with `<|im_start|>` and `<|im_end|>` tokens
- Hermes-compatible tool calling format

**Training Considerations**:
- Use `qwen-2.5` chat template in Unsloth
- Works directly with our Hermes-style training data

```python
# Qwen 2.5 with quantization
model_id = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
```

**Sources**:
- [Qwen Tool Planning Model](https://huggingface.co/Vikhrmodels/Qwen2.5-7B-Instruct-Tool-Planning-v0.1)
- [vLLM Tool Calling Docs](https://docs.vllm.ai/en/latest/features/tool_calling/)

---

## 8. Common Pitfalls & Solutions

### 8.1 Overfitting

**Problem**: Model memorizes training data, performs poorly on new inputs.

**Solutions**:
- Use validation set (80/20 split)
- Implement early stopping
- Keep LoRA rank moderate (16 for most cases)
- Use dropout for `max_quality` profile (0.05)

### 8.2 Catastrophic Forgetting

**Problem**: Model forgets general knowledge during fine-tuning.

**Solutions**:
- Use LoRA/QLoRA (preserves base weights)
- Keep training short (1-3 epochs)
- Include diverse training samples

### 8.3 Wrong Chat Template

**Problem**: Model generates malformed output at inference.

**Solutions**:
- Use same template for training and inference
- Verify template matches base model
- Test inference before deployment

### 8.4 VRAM Exhaustion

**Problem**: Out of memory during training.

**Solutions**:
- Use `gradient_accumulation_steps` instead of larger batch size
- Enable `use_gradient_checkpointing="unsloth"`
- Reduce `max_seq_length` if possible
- Use `optim="adamw_8bit"`

### 8.5 Tool Call Parsing Failures

**Problem**: Model generates invalid JSON in `<tool_call>` tags.

**Solutions**:
- Ensure training data has consistent JSON formatting
- QC validation catches schema violations before training
- Use packing=False to preserve conversation boundaries

**Sources**:
- [Machine Learning Mastery - LLM Fine-tuning Problems](https://machinelearningmastery.com/5-problems-encountered-fine-tuning-llms-with-solutions/)
- [Lakera LLM Fine-tuning Guide](https://www.lakera.ai/blog/llm-fine-tuning-guide)

---

## 9. Architecture Recommendations

### 9.1 Proposed Module Structure

```
src/mcp_forge/training/
├── __init__.py
├── engine.py       # Main TrainingEngine class
├── config.py       # TrainingProfile dataclass
├── callbacks.py    # ForgeProgressCallback
└── utils.py        # Dataset loading, formatting helpers
```

### 9.2 TrainingEngine Interface

```python
@dataclass
class TrainingConfig:
    model_family: str  # "deepseek-r1" or "qwen-2.5"
    profile: str       # "fast_dev", "balanced", "max_quality"
    output_dir: Path
    max_seq_length: int = 2048
    save_steps: int = 50

class TrainingEngine:
    def __init__(self, config: TrainingConfig):
        ...

    def load_model(self) -> tuple[Model, Tokenizer]:
        """Load model with Unsloth QLoRA."""
        ...

    def prepare_dataset(self, data_path: Path) -> Dataset:
        """Format JSONL to HuggingFace Dataset."""
        ...

    def train(self, progress_callback: Callable) -> LoraAdapterPath:
        """Run training with progress reporting."""
        ...

    def save_adapter(self, path: Path) -> None:
        """Save LoRA adapter."""
        ...
```

### 9.3 Integration with Pipeline

```python
# In cli.py _run_pipeline()
if state.stage in (PipelineStage.QC_VALIDATING, PipelineStage.TRAINING):
    from mcp_forge.training import TrainingEngine, TrainingConfig

    config = TrainingConfig(
        model_family=state.model_family,
        profile=state.profile,
        output_dir=state_manager.state_dir / "adapters",
    )

    engine = TrainingEngine(config)

    def on_progress(step, total_steps, progress, loss, epoch):
        state.training_progress = progress
        state.training_loss = loss
        state_manager.save_state(state)
        console.print(f"Step {step}/{total_steps} | Loss: {loss:.4f}")

    adapter_path = engine.train(
        data_path=Path(state.training_data_path),
        progress_callback=on_progress,
    )

    state.lora_adapter_path = str(adapter_path)
```

---

## 10. Dependencies to Add

```toml
# Already in pyproject.toml, but verify versions:
dependencies = [
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    "torch>=2.1.0",
    "transformers>=4.40.0",
    "trl>=0.8.0",
    "bitsandbytes>=0.43.0",
    "datasets>=2.18.0",
    "peft>=0.10.0",  # May need to add explicitly
]
```

**Note**: Unsloth has specific version compatibility. Testing was done with `unsloth==2025.10.1`. Versions 2025.10.2 and 2025.10.3 have reported issues.

---

## 11. Testing Strategy

### 11.1 Unit Tests

- Mock Unsloth model loading (test config parsing)
- Test dataset formatting function
- Test progress callback integration
- Test profile hyperparameter resolution

### 11.2 Integration Tests

- End-to-end training with tiny dataset (5 samples)
- Verify LoRA adapter saves correctly
- Test checkpoint resume functionality
- Validate output format

### 11.3 Test Fixtures

```python
@pytest.fixture
def mock_unsloth_model():
    """Mock FastLanguageModel for unit tests."""
    ...

@pytest.fixture
def tiny_training_dataset(tmp_path):
    """5-sample JSONL for integration tests."""
    ...
```

---

## 12. Research Sources

### Official Documentation
- [Unsloth Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Unsloth LoRA Hyperparameters](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [Unsloth Chat Templates](https://docs.unsloth.ai/basics/chat-templates)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [HuggingFace TrainerCallback](https://huggingface.co/docs/transformers/main_classes/callback)

### Model Resources
- [DeepSeek R1 on Unsloth](https://unsloth.ai/blog/deepseek-r1)
- [Hermes-3 Model Card](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)
- [Hermes Function Calling Dataset](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- [Qwen Tool Planning Model](https://huggingface.co/Vikhrmodels/Qwen2.5-7B-Instruct-Tool-Planning-v0.1)

### Best Practices
- [HuggingFace Unsloth TRL Blog](https://huggingface.co/blog/unsloth-trl)
- [Machine Learning Mastery - LLM Fine-tuning Problems](https://machinelearningmastery.com/5-problems-encountered-fine-tuning-llms-with-solutions/)
- [Lakera LLM Fine-tuning Guide](https://www.lakera.ai/blog/llm-fine-tuning-guide)
- [vLLM Tool Calling](https://docs.vllm.ai/en/latest/features/tool_calling/)

---

## 13. Key Decisions for Planning

1. **Use Unsloth's FastLanguageModel** - Don't hand-roll model loading
2. **Use SFTTrainer from TRL** - Don't hand-roll training loop
3. **Target all linear layers** - Standard for tool-calling fine-tuning
4. **packing=False** - Critical for tool-calling conversations
5. **Implement custom callback** - For pipeline state updates
6. **Save adapters only** - Not merged models (100MB vs 16GB)
7. **Profile-based config** - Map CLI profiles to hyperparameters
8. **Test with tiny dataset** - Integration tests with 5 samples

---

*Research completed: 2026-01-13*
*Ready for: gsd:plan-phase 5*
