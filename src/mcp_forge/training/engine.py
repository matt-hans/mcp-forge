"""Training engine for Unsloth-based LoRA fine-tuning.

Provides the main TrainingEngine class for loading models,
preparing datasets, and running training.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset

from mcp_forge.training.callbacks import ForgeProgressCallback
from mcp_forge.training.config import (
    TrainingConfig,
    TrainingProfile,
    get_model_id,
    get_profile,
)


class TrainingEngine:
    """Unsloth-based training engine with LoRA fine-tuning.

    Handles model loading, dataset preparation, and training execution
    using Unsloth for efficient 4-bit quantization and TRL's SFTTrainer.
    """

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the training engine.

        Args:
            config: Training configuration with model, profile, and paths
        """
        self.config = config
        self.profile: TrainingProfile = get_profile(config.profile)
        self.model: Any = None
        self.tokenizer: Any = None

    def load_model(self) -> tuple[Any, Any]:
        """Load model with Unsloth and apply LoRA adapters.

        Returns:
            Tuple of (model, tokenizer)
        """
        from unsloth import FastLanguageModel

        model_id = get_model_id(self.config.model_family)

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=self.config.max_seq_length,
            dtype=None,  # Auto-detect (bfloat16 on Ampere+)
            load_in_4bit=True,  # QLoRA mode
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.profile.lora_rank,
            lora_alpha=self.profile.lora_alpha,
            lora_dropout=self.profile.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
        )

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def prepare_dataset(self) -> Dataset:
        """Load JSONL and format for SFTTrainer.

        Reads the training data JSONL file and applies the chat template
        to format messages for the model.

        Returns:
            HuggingFace Dataset with 'text' field for training
        """
        import json

        if self.tokenizer is None:
            raise RuntimeError("Model must be loaded before preparing dataset")

        # Load JSONL data
        samples = []
        with open(self.config.data_path) as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        # Format with chat template
        def format_sample(sample: dict[str, Any]) -> str:
            messages = sample.get("messages", [])
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        texts = [format_sample(s) for s in samples]

        return Dataset.from_dict({"text": texts})

    def train(
        self,
        progress_callback: Callable[[int, int, float, float | None, float], None] | None = None,
    ) -> Path:
        """Run training and return adapter path.

        Args:
            progress_callback: Function called with (step, total, progress, loss, epoch)

        Returns:
            Path to saved LoRA adapter directory
        """
        import torch
        from transformers import TrainingArguments
        from transformers.trainer_callback import TrainerCallback
        from trl import SFTTrainer

        if self.model is None:
            self.load_model()

        dataset = self.prepare_dataset()

        # Create output directory
        checkpoint_dir = self.config.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build training arguments
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            per_device_train_batch_size=self.profile.per_device_train_batch_size,
            gradient_accumulation_steps=self.profile.gradient_accumulation_steps,
            learning_rate=self.profile.learning_rate,
            warmup_ratio=self.profile.warmup_ratio,
            num_train_epochs=self.profile.num_train_epochs or 1,
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

        # Wrap callback to conform to TrainerCallback interface
        callbacks: list[TrainerCallback] = []
        if progress_callback:
            forge_callback = ForgeProgressCallback(progress_callback)

            class _CallbackWrapper(TrainerCallback):
                def on_log(
                    self,
                    args: Any,
                    state: Any,
                    control: Any,
                    logs: dict[str, Any] | None = None,
                    **kwargs: Any,
                ) -> None:
                    forge_callback.on_log(args, state, control, logs, **kwargs)

            callbacks.append(_CallbackWrapper())

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.config.max_seq_length,
            packing=False,  # Critical for tool-calling (preserve conversation boundaries)
            args=training_args,
            callbacks=callbacks if callbacks else None,
        )

        trainer.train()

        # Save adapter
        adapter_path = self.config.output_dir / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        return adapter_path
