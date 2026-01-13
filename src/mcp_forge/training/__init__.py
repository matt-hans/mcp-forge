"""Training module for MCP-Forge.

Provides Unsloth-based LoRA fine-tuning for LLMs on MCP tool schemas.
"""

from mcp_forge.training.callbacks import ForgeProgressCallback
from mcp_forge.training.config import MODEL_IDS, PROFILES, TrainingConfig, TrainingProfile
from mcp_forge.training.engine import TrainingEngine

__all__ = [
    "TrainingConfig",
    "TrainingProfile",
    "TrainingEngine",
    "ForgeProgressCallback",
    "PROFILES",
    "MODEL_IDS",
]
