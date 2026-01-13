"""Unit tests for training configuration and profiles."""

from pathlib import Path

import pytest

from mcp_forge.training.config import (
    MODEL_IDS,
    PROFILES,
    TrainingConfig,
    TrainingProfile,
    get_model_id,
    get_profile,
)


class TestTrainingProfile:
    """Tests for TrainingProfile dataclass."""

    def test_profile_attributes(self):
        """Profile has all required attributes."""
        profile = TrainingProfile(
            name="test",
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            num_train_epochs=1,
            max_steps=None,
            save_steps=50,
        )
        assert profile.name == "test"
        assert profile.lora_rank == 16
        assert profile.lora_alpha == 32
        assert profile.lora_dropout == 0.05
        assert profile.per_device_train_batch_size == 2
        assert profile.learning_rate == 2e-4
        assert profile.num_train_epochs == 1
        assert profile.max_steps is None


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_config_defaults(self, tmp_path: Path):
        """Config has sensible defaults."""
        config = TrainingConfig(
            model_family="deepseek-r1",
            profile="balanced",
            data_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
        )
        assert config.max_seq_length == 2048
        assert config.seed == 3407

    def test_config_custom_values(self, tmp_path: Path):
        """Config accepts custom values."""
        config = TrainingConfig(
            model_family="qwen-2.5",
            profile="fast_dev",
            data_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "output",
            max_seq_length=4096,
            seed=42,
        )
        assert config.model_family == "qwen-2.5"
        assert config.profile == "fast_dev"
        assert config.max_seq_length == 4096
        assert config.seed == 42


class TestProfiles:
    """Tests for predefined training profiles."""

    def test_fast_dev_profile_exists(self):
        """fast_dev profile is defined."""
        assert "fast_dev" in PROFILES
        profile = PROFILES["fast_dev"]
        assert profile.name == "fast_dev"
        assert profile.lora_rank == 8
        assert profile.max_steps == 60

    def test_balanced_profile_exists(self):
        """balanced profile is defined."""
        assert "balanced" in PROFILES
        profile = PROFILES["balanced"]
        assert profile.name == "balanced"
        assert profile.lora_rank == 16
        assert profile.num_train_epochs == 1

    def test_max_quality_profile_exists(self):
        """max_quality profile is defined."""
        assert "max_quality" in PROFILES
        profile = PROFILES["max_quality"]
        assert profile.name == "max_quality"
        assert profile.lora_rank == 128
        assert profile.lora_alpha == 256
        assert profile.num_train_epochs == 3

    def test_all_profiles_have_valid_lora_config(self):
        """All profiles have valid LoRA configurations."""
        for name, profile in PROFILES.items():
            assert profile.lora_rank > 0, f"{name} has invalid lora_rank"
            assert profile.lora_alpha >= profile.lora_rank, f"{name} alpha < rank"
            assert 0 <= profile.lora_dropout <= 1, f"{name} invalid dropout"


class TestModelIDs:
    """Tests for model ID mappings."""

    def test_deepseek_r1_mapping(self):
        """deepseek-r1 maps to correct Unsloth ID."""
        assert "deepseek-r1" in MODEL_IDS
        assert "DeepSeek-R1-Distill-Llama-8B" in MODEL_IDS["deepseek-r1"]
        assert "bnb-4bit" in MODEL_IDS["deepseek-r1"]

    def test_qwen_mapping(self):
        """qwen-2.5 maps to correct Unsloth ID."""
        assert "qwen-2.5" in MODEL_IDS
        assert "Qwen2.5-14B-Instruct" in MODEL_IDS["qwen-2.5"]
        assert "bnb-4bit" in MODEL_IDS["qwen-2.5"]


class TestGetProfile:
    """Tests for get_profile function."""

    def test_get_valid_profile(self):
        """get_profile returns correct profile."""
        profile = get_profile("balanced")
        assert profile.name == "balanced"
        assert profile == PROFILES["balanced"]

    def test_get_invalid_profile_raises(self):
        """get_profile raises ValueError for unknown profile."""
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent")

    def test_error_message_lists_valid_profiles(self):
        """Error message includes list of valid profiles."""
        with pytest.raises(ValueError) as exc_info:
            get_profile("bad_profile")
        error_msg = str(exc_info.value)
        assert "balanced" in error_msg
        assert "fast_dev" in error_msg
        assert "max_quality" in error_msg


class TestGetModelId:
    """Tests for get_model_id function."""

    def test_get_valid_model_id(self):
        """get_model_id returns correct Unsloth ID."""
        model_id = get_model_id("deepseek-r1")
        assert model_id == MODEL_IDS["deepseek-r1"]

    def test_get_invalid_model_raises(self):
        """get_model_id raises ValueError for unknown model."""
        with pytest.raises(ValueError, match="Unknown model family"):
            get_model_id("gpt-4")

    def test_error_message_lists_valid_models(self):
        """Error message includes list of valid models."""
        with pytest.raises(ValueError) as exc_info:
            get_model_id("llama")
        error_msg = str(exc_info.value)
        assert "deepseek-r1" in error_msg
        assert "qwen-2.5" in error_msg
