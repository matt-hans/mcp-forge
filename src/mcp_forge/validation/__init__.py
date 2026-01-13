"""Looped validation module for trained model verification.

Provides validation against real and stubbed MCP servers with
deterministic testing and metric aggregation.
"""

from mcp_forge.validation.config import (
    InferenceConfig,
    StubConfig,
    ValidationConfig,
)
from mcp_forge.validation.runner import (
    ValidationRunner,
    ValidationSample,
    generate_validation_samples,
)
from mcp_forge.validation.stubs import (
    FilesystemStub,
    MCPStub,
    StubRegistry,
    WeatherStub,
)

__all__ = [
    # Config
    "InferenceConfig",
    "StubConfig",
    "ValidationConfig",
    # Runner
    "ValidationRunner",
    "ValidationSample",
    "generate_validation_samples",
    # Stubs
    "MCPStub",
    "WeatherStub",
    "FilesystemStub",
    "StubRegistry",
]
