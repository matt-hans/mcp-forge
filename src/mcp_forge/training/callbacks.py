"""Training callbacks for progress reporting.

Provides callbacks for HuggingFace Trainer to report progress
and update pipeline state during training.
"""

from collections.abc import Callable
from typing import Any


class ForgeProgressCallback:
    """Callback for pipeline progress updates during training.

    This callback integrates with HuggingFace's Trainer and reports
    training progress to the pipeline state manager.
    """

    def __init__(
        self,
        progress_callback: Callable[[int, int, float, float | None, float], None] | None = None,
    ) -> None:
        """Initialize the progress callback.

        Args:
            progress_callback: Function called with (step, total, progress, loss, epoch)
        """
        self.progress_callback = progress_callback

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Handle logging events from the trainer.

        Args:
            args: TrainingArguments
            state: TrainerState with global_step, max_steps, epoch
            control: TrainerControl
            logs: Dict with loss and other metrics
            **kwargs: Additional arguments
        """
        if logs and state.is_local_process_zero and self.progress_callback:
            loss = logs.get("loss")
            step = state.global_step
            total = state.max_steps or 1
            progress = step / total if total > 0 else 0.0
            epoch = state.epoch or 0.0
            self.progress_callback(step, total, progress, loss, epoch)
