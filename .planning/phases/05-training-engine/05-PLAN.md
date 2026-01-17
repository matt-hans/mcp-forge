# Phase 5: Training Engine - Execution Plan

**Phase**: 5 of 9
**Milestone**: v1.0 - Full Pipeline Implementation
**Created**: 2026-01-13
**Estimated Scope**: Medium (7 tasks)

---

## Objective

Implement the TRAINING stage using Unsloth with LoRA/QLoRA fine-tuning. Create a `training/` module that:
1. Loads models with 4-bit quantization via Unsloth
2. Applies LoRA adapters with profile-based hyperparameters
3. Trains using TRL's SFTTrainer with progress callbacks
4. Saves LoRA adapters and integrates with pipeline state

---

## Execution Context

**Research**: `05-RESEARCH.md` - Comprehensive Unsloth/LoRA documentation
**Architecture**: Follow patterns in `qc.py`, `synthesizer.py`
**Conventions**: `CONVENTIONS.md` - Dataclasses, type hints, docstrings
**Testing**: 203 existing tests, maintain 85%+ coverage

---

## Context

### Existing Infrastructure

| Component | Path | Integration Point |
|-----------|------|-------------------|
| Pipeline State | `state.py:236-237` | `training_progress`, `training_loss`, `lora_adapter_path` |
| Training Stub | `cli.py:244-252` | Stage 4 transition to TRAINING |
| Config System | `config.py` | `training_profile`, `training_quantization` |
| Formatter | `formatter.py` | Hermes ChatML format already correct |
| Test Fixtures | `conftest.py` | `sample_state`, `synthesis_plan`, `temp_jsonl_file` |

### Model Mapping

| CLI Value | Unsloth Model ID |
|-----------|------------------|
| `deepseek-r1` | `unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit` |
| `qwen-2.5` | `unsloth/Qwen2.5-14B-Instruct-bnb-4bit` |

### Profile Mapping

| Profile | LoRA Rank | Alpha | Max Steps | Learning Rate |
|---------|-----------|-------|-----------|---------------|
| `fast_dev` | 8 | 16 | 60 | 2e-4 |
| `balanced` | 16 | 16 | 1 epoch | 2e-4 |
| `max_quality` | 128 | 256 | 3 epochs | 1e-4 |

---

## Tasks

### Task 1: Create training module structure

**Action**: Create `src/mcp_forge/training/` package with module files.

**Files**:
- `src/mcp_forge/training/__init__.py` - Module exports
- `src/mcp_forge/training/config.py` - TrainingConfig, TrainingProfile dataclasses
- `src/mcp_forge/training/callbacks.py` - ForgeProgressCallback
- `src/mcp_forge/training/engine.py` - TrainingEngine class

**Details**:
```
src/mcp_forge/training/
├── __init__.py         # Export TrainingEngine, TrainingConfig
├── config.py           # TrainingConfig, TrainingProfile, PROFILES dict
├── callbacks.py        # ForgeProgressCallback(TrainerCallback)
└── engine.py           # TrainingEngine class
```

**Verification**: `python -c "from mcp_forge.training import TrainingEngine"`

---

### Task 2: Implement TrainingConfig and profiles

**Action**: Create configuration dataclasses and profile definitions.

**File**: `src/mcp_forge/training/config.py`

**Implementation**:
```python
@dataclass
class TrainingProfile:
    """Hyperparameters for a training profile."""
    name: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    num_train_epochs: int | None  # None = use max_steps
    max_steps: int | None  # None = use epochs
    save_steps: int

@dataclass
class TrainingConfig:
    """Configuration for training run."""
    model_family: str  # "deepseek-r1" or "qwen-2.5"
    profile: str  # "fast_dev", "balanced", "max_quality"
    data_path: Path
    output_dir: Path
    max_seq_length: int = 2048
    seed: int = 3407

PROFILES: dict[str, TrainingProfile] = { ... }
MODEL_IDS: dict[str, str] = { ... }
```

**Verification**: Unit test for profile resolution

---

### Task 3: Implement ForgeProgressCallback

**Action**: Create TrainerCallback for progress reporting and state updates.

**File**: `src/mcp_forge/training/callbacks.py`

**Implementation**:
```python
from transformers import TrainerCallback, TrainerState, TrainerControl

class ForgeProgressCallback(TrainerCallback):
    """Callback for pipeline progress updates."""

    def __init__(
        self,
        progress_callback: Callable[[int, int, float, float | None, float], None] | None = None,
    ):
        self.progress_callback = progress_callback

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if logs and state.is_local_process_zero and self.progress_callback:
            loss = logs.get("loss")
            step = state.global_step
            total = state.max_steps or 1
            progress = step / total
            epoch = state.epoch or 0.0
            self.progress_callback(step, total, progress, loss, epoch)
```

**Verification**: Unit test with mock callback

---

### Task 4: Implement TrainingEngine core

**Action**: Create the main training engine with model loading and dataset preparation.

**File**: `src/mcp_forge/training/engine.py`

**Implementation**:
```python
class TrainingEngine:
    """Unsloth-based training engine with LoRA fine-tuning."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.profile = get_profile(config.profile)
        self.model = None
        self.tokenizer = None

    def load_model(self) -> tuple[Any, Any]:
        """Load model with Unsloth and apply LoRA adapters."""
        from unsloth import FastLanguageModel

        model_id = MODEL_IDS[self.config.model_family]

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.profile.lora_rank,
            lora_alpha=self.profile.lora_alpha,
            lora_dropout=self.profile.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
        )

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def prepare_dataset(self) -> Dataset:
        """Load JSONL and format for SFTTrainer."""
        ...
```

**Key Methods**:
- `load_model()` - Unsloth model + LoRA
- `prepare_dataset()` - JSONL → HuggingFace Dataset with chat template
- `_formatting_func()` - Apply chat template to messages

**Verification**: Unit test with mocked Unsloth imports

---

### Task 5: Implement training execution

**Action**: Add train() method with SFTTrainer integration and adapter saving.

**File**: `src/mcp_forge/training/engine.py` (continued)

**Implementation**:
```python
def train(
    self,
    progress_callback: Callable | None = None,
) -> Path:
    """Run training and return adapter path."""
    from trl import SFTTrainer
    from transformers import TrainingArguments

    if self.model is None:
        self.load_model()

    dataset = self.prepare_dataset()

    training_args = TrainingArguments(
        output_dir=str(self.config.output_dir / "checkpoints"),
        per_device_train_batch_size=self.profile.per_device_train_batch_size,
        gradient_accumulation_steps=self.profile.gradient_accumulation_steps,
        learning_rate=self.profile.learning_rate,
        warmup_ratio=self.profile.warmup_ratio,
        num_train_epochs=self.profile.num_train_epochs,
        max_steps=self.profile.max_steps or -1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=self.config.seed,
        save_strategy="steps",
        save_steps=self.profile.save_steps,
        save_total_limit=3,
    )

    callbacks = []
    if progress_callback:
        callbacks.append(ForgeProgressCallback(progress_callback))

    trainer = SFTTrainer(
        model=self.model,
        tokenizer=self.tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=self.config.max_seq_length,
        packing=False,  # Critical for tool-calling
        args=training_args,
        callbacks=callbacks,
    )

    trainer.train()

    # Save adapter
    adapter_path = self.config.output_dir / "adapter"
    self.model.save_pretrained(adapter_path)
    self.tokenizer.save_pretrained(adapter_path)

    return adapter_path
```

**Verification**: Integration test with tiny dataset (5 samples)

---

### Task 6: Integrate with CLI pipeline

**Action**: Wire TrainingEngine into `cli.py` `_run_pipeline()`.

**File**: `src/mcp_forge/cli.py`

**Replace** (lines 244-252):
```python
# Stage 4: Training
if state.stage in (PipelineStage.QC_VALIDATING, PipelineStage.TRAINING):
    console.print("\n[bold]Stage 4: Training model...[/bold]")
    state.update_stage(PipelineStage.TRAINING)
    state_manager.save_state(state)

    console.print(f"   Model: {state.model_family}")
    console.print(f"   Profile: {state.profile}")
    # TODO: Implement training
    console.print("   [yellow]Training not yet implemented[/yellow]")
```

**With**:
```python
# Stage 4: Training
if state.stage in (PipelineStage.QC_VALIDATING, PipelineStage.TRAINING):
    console.print("\n[bold]Stage 4: Training model...[/bold]")
    state.update_stage(PipelineStage.TRAINING)
    state_manager.save_state(state)

    from mcp_forge.training import TrainingEngine, TrainingConfig

    console.print(f"   Model: {state.model_family}")
    console.print(f"   Profile: {state.profile}")

    training_config = TrainingConfig(
        model_family=state.model_family,
        profile=state.profile,
        data_path=Path(state.training_data_path),
        output_dir=state_manager.state_dir / "adapters",
    )

    engine = TrainingEngine(training_config)

    def on_training_progress(step, total, progress, loss, epoch):
        state.training_progress = progress
        state.training_loss = loss
        state_manager.save_state(state)
        loss_str = f"{loss:.4f}" if loss else "N/A"
        console.print(f"   Step {step}/{total} | Loss: {loss_str}")

    with console.status("[bold green]Training..."):
        adapter_path = engine.train(progress_callback=on_training_progress)

    state.lora_adapter_path = str(adapter_path)
    console.print(f"   [green]✓[/green] Adapter saved to {adapter_path}")
```

**Also update** standalone `train` command (lines 771-773).

**Verification**: CLI smoke test with `mcp-forge train --help`

---

### Task 7: Add tests

**Action**: Create unit and integration tests for training module.

**Files**:
- `tests/unit/test_training_config.py` - Profile resolution, config validation
- `tests/unit/test_training_callbacks.py` - ForgeProgressCallback
- `tests/integration/test_training_engine.py` - End-to-end with mocked Unsloth

**Test Cases**:

```python
# test_training_config.py
def test_profile_resolution():
    """Profile name resolves to correct hyperparameters."""

def test_model_id_mapping():
    """Model family maps to correct Unsloth ID."""

def test_config_validation():
    """Invalid profile raises ValueError."""

# test_training_callbacks.py
def test_progress_callback_called():
    """Callback receives step, progress, loss."""

def test_callback_handles_missing_loss():
    """Callback handles logs without loss key."""

# test_training_engine.py
@pytest.fixture
def mock_unsloth(monkeypatch):
    """Mock FastLanguageModel for tests without GPU."""

def test_engine_loads_model(mock_unsloth):
    """Engine loads model with correct config."""

def test_engine_prepares_dataset(tiny_training_dataset):
    """Dataset formatted with chat template."""

@pytest.mark.integration
def test_training_end_to_end(mock_unsloth, tiny_training_dataset):
    """Full training flow with mocked model."""
```

**Fixture** for tiny dataset:
```python
@pytest.fixture
def tiny_training_dataset(tmp_path, sample_tools):
    """5-sample JSONL for training tests."""
    # Create minimal valid training data
```

**Verification**: `pytest tests/unit/test_training*.py tests/integration/test_training*.py -v`

---

## Verification

After all tasks complete:

```bash
# 1. Lint and type check
ruff check src/mcp_forge/training/
mypy src/mcp_forge/training/

# 2. Run training tests
pytest tests/unit/test_training*.py -v
pytest tests/integration/test_training*.py -v

# 3. Check coverage
pytest --cov=src/mcp_forge/training --cov-report=term-missing

# 4. CLI smoke test
mcp-forge train --help

# 5. Verify import
python -c "from mcp_forge.training import TrainingEngine, TrainingConfig; print('OK')"
```

---

## Success Criteria

- [ ] `src/mcp_forge/training/` module created with 4 files
- [ ] TrainingConfig and TrainingProfile dataclasses defined
- [ ] Three profiles implemented: fast_dev, balanced, max_quality
- [ ] ForgeProgressCallback updates pipeline state during training
- [ ] TrainingEngine loads models via Unsloth with 4-bit quantization
- [ ] TrainingEngine trains with SFTTrainer (packing=False)
- [ ] LoRA adapters saved to `.mcp-forge/adapters/`
- [ ] CLI `train` command wired to TrainingEngine
- [ ] Pipeline stage transitions correctly (QC_VALIDATING → TRAINING)
- [ ] All tests pass with mocked Unsloth (no GPU required)
- [ ] Coverage maintained at 85%+
- [ ] ruff and mypy pass on new code

---

## Output

| Artifact | Path |
|----------|------|
| Training module | `src/mcp_forge/training/` |
| Config tests | `tests/unit/test_training_config.py` |
| Callback tests | `tests/unit/test_training_callbacks.py` |
| Engine tests | `tests/integration/test_training_engine.py` |
| LoRA adapters (runtime) | `.mcp-forge/adapters/` |

---

## Notes

### GPU Requirements

- Training requires CUDA GPU with 6GB+ VRAM (8B model) or 9GB+ (14B model)
- Tests use mocked Unsloth to run without GPU
- Integration tests marked with `@pytest.mark.integration` can be skipped in CI

### Dependencies

The following are already in `pyproject.toml`:
- `unsloth[colab-new]`
- `torch>=2.1.0`
- `transformers>=4.40.0`
- `trl>=0.8.0`
- `datasets>=2.18.0`
- `bitsandbytes>=0.43.0`

May need to add `peft>=0.10.0` explicitly.

### Chat Template Handling

The engine will:
1. Load JSONL with `messages` array
2. Apply model's native chat template via `tokenizer.apply_chat_template()`
3. Store formatted text in `"text"` field for SFTTrainer

This preserves the Hermes format from `formatter.py` while using model-native templates.

---

*Plan created: 2026-01-13*
*Execute with: /gsd:execute-plan*
