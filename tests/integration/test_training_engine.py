"""Integration tests for TrainingEngine with mocked Unsloth.

These tests mock the heavy ML dependencies (unsloth, transformers, trl)
at the sys.modules level to allow testing without GPU or heavy imports.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcp_forge.training import TrainingConfig, TrainingEngine
from mcp_forge.training.config import get_profile


@pytest.fixture
def tiny_training_dataset(tmp_path: Path) -> Path:
    """Create a tiny JSONL training file for tests."""
    data_path = tmp_path / "train.jsonl"
    samples = []

    for i in range(5):
        sample = {
            "id": f"sample_{i:03d}",
            "source": "seed",
            "scenario": "standard",
            "tool_name": "get_weather",
            "messages": [
                {"role": "system", "content": "<tools>[...]</tools>"},
                {"role": "user", "content": f"What's the weather in City{i}?"},
                {
                    "role": "assistant",
                    "content": f'<tool_call>\n{{"name": "get_weather", "arguments": {{"location": "City{i}"}}}}\n</tool_call>',
                },
                {"role": "tool", "content": '{"temp": 22}'},
                {"role": "assistant", "content": f"The weather in City{i} is 22 degrees."},
            ],
        }
        samples.append(sample)

    with open(data_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return data_path


@pytest.fixture
def training_config(tmp_path: Path, tiny_training_dataset: Path) -> TrainingConfig:
    """TrainingConfig for tests."""
    return TrainingConfig(
        model_family="deepseek-r1",
        profile="fast_dev",
        data_path=tiny_training_dataset,
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def mock_unsloth():
    """Mock unsloth module with FastLanguageModel."""
    # Create mock FastLanguageModel class
    mock_flm = MagicMock()

    # Mock model
    mock_model = MagicMock()
    mock_model.save_pretrained = MagicMock()

    # Mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(
        side_effect=lambda msgs, **kwargs: f"formatted_{len(msgs)}_messages"
    )
    mock_tokenizer.save_pretrained = MagicMock()

    # Configure from_pretrained
    mock_flm.from_pretrained = MagicMock(return_value=(mock_model, mock_tokenizer))
    mock_flm.get_peft_model = MagicMock(return_value=mock_model)

    # Create a mock unsloth module
    mock_unsloth_module = MagicMock()
    mock_unsloth_module.FastLanguageModel = mock_flm

    # Patch sys.modules to inject our mock
    with patch.dict(sys.modules, {"unsloth": mock_unsloth_module}):
        yield {
            "FastLanguageModel": mock_flm,
            "model": mock_model,
            "tokenizer": mock_tokenizer,
        }


class TestTrainingEngineInit:
    """Tests for TrainingEngine initialization."""

    def test_init_stores_config(self, training_config: TrainingConfig):
        """Engine stores config correctly."""
        engine = TrainingEngine(training_config)
        assert engine.config == training_config
        assert engine.model is None
        assert engine.tokenizer is None

    def test_init_resolves_profile(self, training_config: TrainingConfig):
        """Engine resolves profile from config."""
        engine = TrainingEngine(training_config)
        expected_profile = get_profile("fast_dev")
        assert engine.profile.name == expected_profile.name
        assert engine.profile.lora_rank == expected_profile.lora_rank


class TestTrainingEngineLoadModel:
    """Tests for model loading."""

    def test_load_model_calls_unsloth(
        self, training_config: TrainingConfig, mock_unsloth: dict
    ):
        """load_model uses Unsloth correctly."""
        engine = TrainingEngine(training_config)
        model, tokenizer = engine.load_model()

        # Verify from_pretrained called
        mock_unsloth["FastLanguageModel"].from_pretrained.assert_called_once()
        call_kwargs = mock_unsloth["FastLanguageModel"].from_pretrained.call_args[1]
        assert "DeepSeek-R1" in call_kwargs["model_name"]
        assert call_kwargs["max_seq_length"] == 2048
        assert call_kwargs["load_in_4bit"] is True

        # Verify get_peft_model called
        mock_unsloth["FastLanguageModel"].get_peft_model.assert_called_once()
        peft_kwargs = mock_unsloth["FastLanguageModel"].get_peft_model.call_args[1]
        assert peft_kwargs["r"] == 8  # fast_dev rank
        assert peft_kwargs["lora_alpha"] == 16
        assert "q_proj" in peft_kwargs["target_modules"]

        # Verify model/tokenizer stored
        assert engine.model is not None
        assert engine.tokenizer is not None

    def test_load_model_uses_correct_model_id(
        self, tmp_path: Path, tiny_training_dataset: Path, mock_unsloth: dict
    ):
        """load_model uses correct model ID for each family."""
        for model_family, expected_substring in [
            ("deepseek-r1", "DeepSeek-R1"),
            ("qwen-2.5", "Qwen2.5"),
        ]:
            mock_unsloth["FastLanguageModel"].from_pretrained.reset_mock()

            config = TrainingConfig(
                model_family=model_family,
                profile="fast_dev",
                data_path=tiny_training_dataset,
                output_dir=tmp_path / "output",
            )
            engine = TrainingEngine(config)
            engine.load_model()

            call_kwargs = mock_unsloth["FastLanguageModel"].from_pretrained.call_args[1]
            assert expected_substring in call_kwargs["model_name"]

    @pytest.mark.parametrize("profile_name,expected_rank,expected_alpha", [
        ("fast_dev", 8, 16),
        ("balanced", 16, 16),
        ("max_quality", 128, 256),
    ])
    def test_load_model_applies_profile_lora_config(
        self,
        tmp_path: Path,
        tiny_training_dataset: Path,
        mock_unsloth: dict,
        profile_name: str,
        expected_rank: int,
        expected_alpha: int,
    ):
        """load_model applies correct LoRA config from profile."""
        config = TrainingConfig(
            model_family="deepseek-r1",
            profile=profile_name,
            data_path=tiny_training_dataset,
            output_dir=tmp_path / "output",
        )
        engine = TrainingEngine(config)
        engine.load_model()

        peft_kwargs = mock_unsloth["FastLanguageModel"].get_peft_model.call_args[1]
        assert peft_kwargs["r"] == expected_rank
        assert peft_kwargs["lora_alpha"] == expected_alpha


class TestTrainingEnginePrepareDataset:
    """Tests for dataset preparation."""

    def test_prepare_dataset_requires_loaded_model(
        self, training_config: TrainingConfig
    ):
        """prepare_dataset raises if model not loaded."""
        engine = TrainingEngine(training_config)
        with pytest.raises(RuntimeError, match="Model must be loaded"):
            engine.prepare_dataset()

    def test_prepare_dataset_loads_jsonl(
        self, training_config: TrainingConfig, mock_unsloth: dict
    ):
        """prepare_dataset loads and formats JSONL correctly."""
        engine = TrainingEngine(training_config)
        engine.load_model()

        dataset = engine.prepare_dataset()

        assert len(dataset) == 5  # 5 samples in tiny dataset
        assert "text" in dataset.column_names

    def test_prepare_dataset_applies_chat_template(
        self, training_config: TrainingConfig, mock_unsloth: dict
    ):
        """prepare_dataset applies tokenizer chat template."""
        engine = TrainingEngine(training_config)
        engine.load_model()

        engine.prepare_dataset()

        # Verify chat template was called for each sample
        assert mock_unsloth["tokenizer"].apply_chat_template.call_count == 5

    def test_prepare_dataset_formats_messages(
        self, training_config: TrainingConfig, mock_unsloth: dict
    ):
        """prepare_dataset correctly formats message content."""
        engine = TrainingEngine(training_config)
        engine.load_model()

        dataset = engine.prepare_dataset()

        # Check that the formatted text contains expected pattern
        # Our mock returns "formatted_X_messages" where X is number of messages
        assert dataset[0]["text"] == "formatted_5_messages"


class TestTrainingEngineTrainConfiguration:
    """Tests for train() configuration without running full training.

    These tests verify the engine correctly configures training parameters
    without actually invoking the training loop, which requires GPU.
    """

    def test_train_method_exists(self, training_config: TrainingConfig):
        """Engine has a train method."""
        engine = TrainingEngine(training_config)
        assert hasattr(engine, "train")
        assert callable(engine.train)

    def test_profile_determines_training_hyperparameters(
        self, training_config: TrainingConfig
    ):
        """Profile resolves to correct hyperparameters for training."""
        engine = TrainingEngine(training_config)

        # fast_dev profile
        assert engine.profile.max_steps == 60
        assert engine.profile.per_device_train_batch_size == 2
        assert engine.profile.gradient_accumulation_steps == 4
        assert engine.profile.learning_rate == 2e-4

    def test_different_profiles_have_different_configs(
        self, tmp_path: Path, tiny_training_dataset: Path
    ):
        """Different profiles produce different training configurations."""
        profiles = {}
        for profile_name in ["fast_dev", "balanced", "max_quality"]:
            config = TrainingConfig(
                model_family="deepseek-r1",
                profile=profile_name,
                data_path=tiny_training_dataset,
                output_dir=tmp_path / "output",
            )
            engine = TrainingEngine(config)
            profiles[profile_name] = engine.profile

        # fast_dev uses max_steps, others use epochs
        assert profiles["fast_dev"].max_steps == 60
        assert profiles["fast_dev"].num_train_epochs is None
        assert profiles["balanced"].max_steps is None
        assert profiles["balanced"].num_train_epochs == 1
        assert profiles["max_quality"].max_steps is None
        assert profiles["max_quality"].num_train_epochs == 3

        # LoRA ranks differ
        assert profiles["fast_dev"].lora_rank < profiles["balanced"].lora_rank
        assert profiles["balanced"].lora_rank < profiles["max_quality"].lora_rank

    def test_config_seed_propagates(self, tmp_path: Path, tiny_training_dataset: Path):
        """Custom seed from config is accessible."""
        config = TrainingConfig(
            model_family="deepseek-r1",
            profile="fast_dev",
            data_path=tiny_training_dataset,
            output_dir=tmp_path / "output",
            seed=42,
        )
        engine = TrainingEngine(config)
        assert engine.config.seed == 42

    def test_config_max_seq_length_propagates(
        self, tmp_path: Path, tiny_training_dataset: Path
    ):
        """Custom max_seq_length from config is accessible."""
        config = TrainingConfig(
            model_family="deepseek-r1",
            profile="fast_dev",
            data_path=tiny_training_dataset,
            output_dir=tmp_path / "output",
            max_seq_length=4096,
        )
        engine = TrainingEngine(config)
        assert engine.config.max_seq_length == 4096
