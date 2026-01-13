"""Deterministic MCP stub implementations for reproducible testing."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any


class MCPStub(ABC):
    """Base class for deterministic MCP stubs."""

    def __init__(self, seed: int = 42) -> None:
        """Initialize stub with random seed for determinism.

        Args:
            seed: Random seed for reproducible behavior
        """
        self.rng = random.Random(seed)
        self._call_count = 0

    @abstractmethod
    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions for this stub.

        Returns:
            List of tool definition dictionaries with name, description, inputSchema
        """
        pass

    @abstractmethod
    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool call and return deterministic result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Dictionary containing the tool execution result
        """
        pass

    @property
    def call_count(self) -> int:
        """Return number of times execute() has been called."""
        return self._call_count


class WeatherStub(MCPStub):
    """Deterministic weather service stub."""

    CITIES: dict[str, dict[str, Any]] = {
        "Paris": {"temp": 18, "condition": "cloudy"},
        "London": {"temp": 14, "condition": "rainy"},
        "Tokyo": {"temp": 22, "condition": "sunny"},
        "New York": {"temp": 16, "condition": "partly_cloudy"},
    }

    def get_tools(self) -> list[dict[str, Any]]:
        """Return weather tool definition."""
        return [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            }
        ]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute weather tool call with deterministic results.

        Args:
            tool_name: Must be "get_weather"
            arguments: Dict with "location" (required) and "units" (optional)

        Returns:
            Weather data dictionary or error
        """
        self._call_count += 1
        if tool_name != "get_weather":
            return {"error": f"Unknown tool: {tool_name}"}

        location = arguments.get("location", "Unknown")
        units = arguments.get("units", "celsius")

        weather = self.CITIES.get(location, {"temp": 20, "condition": "unknown"})
        temp = weather["temp"]

        if units == "fahrenheit":
            temp = int(temp * 9 / 5 + 32)

        return {
            "location": location,
            "temperature": temp,
            "units": units,
            "condition": weather["condition"],
        }


class FilesystemStub(MCPStub):
    """Deterministic filesystem operations stub."""

    VIRTUAL_FS: dict[str, list[str]] = {
        "/home/user/documents": ["report.pdf", "notes.txt"],
        "/home/user/projects": ["app.py", "test.py", "README.md"],
    }

    def get_tools(self) -> list[dict[str, Any]]:
        """Return filesystem tool definition."""
        return [
            {
                "name": "list_files",
                "description": "List files in a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"},
                    },
                    "required": ["path"],
                },
            }
        ]

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute filesystem tool call with deterministic results.

        Args:
            tool_name: Must be "list_files"
            arguments: Dict with "path" (required)

        Returns:
            Filesystem data dictionary or error
        """
        self._call_count += 1
        if tool_name != "list_files":
            return {"error": f"Unknown tool: {tool_name}"}

        path = arguments.get("path", "/")
        files = self.VIRTUAL_FS.get(path, [])
        return {"path": path, "files": files}


class StubRegistry:
    """Registry of available MCP stubs."""

    STUBS: dict[str, type[MCPStub]] = {
        "weather": WeatherStub,
        "filesystem": FilesystemStub,
    }

    @classmethod
    def get(cls, stub_type: str, seed: int = 42) -> MCPStub:
        """Get a stub instance by type.

        Args:
            stub_type: Type of stub ("weather" or "filesystem")
            seed: Random seed for deterministic behavior

        Returns:
            MCPStub instance

        Raises:
            ValueError: If stub_type is not registered
        """
        if stub_type not in cls.STUBS:
            raise ValueError(
                f"Unknown stub: {stub_type}. Available: {list(cls.STUBS.keys())}"
            )
        return cls.STUBS[stub_type](seed=seed)

    @classmethod
    def available(cls) -> list[str]:
        """Return list of available stub types."""
        return list(cls.STUBS.keys())
