"""Training configuration and profiles for MCP-Forge.

Defines hyperparameter profiles and model mappings for Unsloth training.
"""

from dataclasses import dataclass
from pathlib import Path


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
    """Configuration for a training run."""

    model_family: str  # "deepseek-r1" or "qwen-2.5"
    profile: str  # "fast_dev", "balanced", "max_quality"
    data_path: Path
    output_dir: Path
    max_seq_length: int = 2048
    seed: int = 3407


# Mapping from CLI model names to Unsloth model IDs
MODEL_IDS: dict[str, str] = {
    "deepseek-r1": "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
    "qwen-2.5": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    "qwen-2.5-7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
}


# Training profiles with LoRA and training hyperparameters
PROFILES: dict[str, TrainingProfile] = {
    "fast_dev": TrainingProfile(
        name="fast_dev",
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        num_train_epochs=None,
        max_steps=60,
        save_steps=20,
    ),
    "balanced": TrainingProfile(
        name="balanced",
        lora_rank=16,
        lora_alpha=16,
        lora_dropout=0.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        num_train_epochs=1,
        max_steps=None,
        save_steps=50,
    ),
    "max_quality": TrainingProfile(
        name="max_quality",
        lora_rank=128,
        lora_alpha=256,
        lora_dropout=0.05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        num_train_epochs=3,
        max_steps=None,
        save_steps=100,
    ),
}


def get_profile(name: str) -> TrainingProfile:
    """Get a training profile by name.

    Args:
        name: Profile name ("fast_dev", "balanced", or "max_quality")

    Returns:
        The training profile

    Raises:
        ValueError: If profile name is not recognized
    """
    if name not in PROFILES:
        valid = ", ".join(sorted(PROFILES.keys()))
        raise ValueError(f"Unknown profile '{name}'. Valid profiles: {valid}")
    return PROFILES[name]


def get_model_id(model_family: str) -> str:
    """Get Unsloth model ID for a model family.

    Args:
        model_family: CLI model name ("deepseek-r1" or "qwen-2.5")

    Returns:
        The Unsloth model ID

    Raises:
        ValueError: If model family is not supported
    """
    if model_family not in MODEL_IDS:
        valid = ", ".join(sorted(MODEL_IDS.keys()))
        raise ValueError(f"Unknown model family '{model_family}'. Valid models: {valid}")
    return MODEL_IDS[model_family]
