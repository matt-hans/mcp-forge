"""Unit tests for training callbacks."""

from unittest.mock import MagicMock

import pytest

from mcp_forge.training.callbacks import ForgeProgressCallback


class TestForgeProgressCallback:
    """Tests for ForgeProgressCallback."""

    def test_callback_initialization(self):
        """Callback initializes with optional progress_callback."""
        callback = ForgeProgressCallback()
        assert callback.progress_callback is None

        mock_fn = MagicMock()
        callback = ForgeProgressCallback(progress_callback=mock_fn)
        assert callback.progress_callback is mock_fn

    def test_on_log_calls_progress_callback(self):
        """on_log invokes progress_callback with correct values."""
        mock_fn = MagicMock()
        callback = ForgeProgressCallback(progress_callback=mock_fn)

        # Create mock trainer state
        mock_state = MagicMock()
        mock_state.is_local_process_zero = True
        mock_state.global_step = 10
        mock_state.max_steps = 100
        mock_state.epoch = 0.5

        logs = {"loss": 0.5432}

        callback.on_log(
            args=MagicMock(),
            state=mock_state,
            control=MagicMock(),
            logs=logs,
        )

        mock_fn.assert_called_once()
        call_args = mock_fn.call_args[0]
        assert call_args[0] == 10  # step
        assert call_args[1] == 100  # total
        assert call_args[2] == pytest.approx(0.1)  # progress
        assert call_args[3] == pytest.approx(0.5432)  # loss
        assert call_args[4] == pytest.approx(0.5)  # epoch

    def test_on_log_handles_missing_loss(self):
        """on_log handles logs without loss key."""
        mock_fn = MagicMock()
        callback = ForgeProgressCallback(progress_callback=mock_fn)

        mock_state = MagicMock()
        mock_state.is_local_process_zero = True
        mock_state.global_step = 5
        mock_state.max_steps = 50
        mock_state.epoch = 0.1

        logs = {"learning_rate": 2e-4}  # No loss

        callback.on_log(
            args=MagicMock(),
            state=mock_state,
            control=MagicMock(),
            logs=logs,
        )

        mock_fn.assert_called_once()
        call_args = mock_fn.call_args[0]
        assert call_args[3] is None  # loss is None

    def test_on_log_skips_non_main_process(self):
        """on_log doesn't call callback on non-main processes."""
        mock_fn = MagicMock()
        callback = ForgeProgressCallback(progress_callback=mock_fn)

        mock_state = MagicMock()
        mock_state.is_local_process_zero = False  # Not main process

        callback.on_log(
            args=MagicMock(),
            state=mock_state,
            control=MagicMock(),
            logs={"loss": 0.5},
        )

        mock_fn.assert_not_called()

    def test_on_log_skips_empty_logs(self):
        """on_log doesn't call callback when logs is None."""
        mock_fn = MagicMock()
        callback = ForgeProgressCallback(progress_callback=mock_fn)

        mock_state = MagicMock()
        mock_state.is_local_process_zero = True

        callback.on_log(
            args=MagicMock(),
            state=mock_state,
            control=MagicMock(),
            logs=None,
        )

        mock_fn.assert_not_called()

    def test_on_log_no_callback_set(self):
        """on_log handles case where no callback is set."""
        callback = ForgeProgressCallback()  # No callback

        mock_state = MagicMock()
        mock_state.is_local_process_zero = True

        # Should not raise
        callback.on_log(
            args=MagicMock(),
            state=mock_state,
            control=MagicMock(),
            logs={"loss": 0.5},
        )

    def test_on_log_handles_zero_max_steps(self):
        """on_log handles edge case of max_steps=0."""
        mock_fn = MagicMock()
        callback = ForgeProgressCallback(progress_callback=mock_fn)

        mock_state = MagicMock()
        mock_state.is_local_process_zero = True
        mock_state.global_step = 0
        mock_state.max_steps = 0  # Edge case
        mock_state.epoch = 0.0

        callback.on_log(
            args=MagicMock(),
            state=mock_state,
            control=MagicMock(),
            logs={"loss": 1.0},
        )

        call_args = mock_fn.call_args[0]
        assert call_args[2] == 0.0  # progress should be 0, not NaN

    def test_on_log_handles_none_epoch(self):
        """on_log handles case where epoch is None."""
        mock_fn = MagicMock()
        callback = ForgeProgressCallback(progress_callback=mock_fn)

        mock_state = MagicMock()
        mock_state.is_local_process_zero = True
        mock_state.global_step = 5
        mock_state.max_steps = 100
        mock_state.epoch = None

        callback.on_log(
            args=MagicMock(),
            state=mock_state,
            control=MagicMock(),
            logs={"loss": 0.8},
        )

        call_args = mock_fn.call_args[0]
        assert call_args[4] == 0.0  # epoch defaults to 0.0
