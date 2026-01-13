"""Tests for mcp_forge.tools.inspector module."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_forge.state import ToolDefinition
from mcp_forge.tools.inspector import (
    MCPInspectorError,
    format_tool_for_display,
    generate_tool_use_prompt,
    parse_command,
    validate_tool_call,
)


class TestParseCommand:
    """Tests for parse_command function."""

    def test_simple_command(self) -> None:
        """Test parsing a simple command."""
        exe, args = parse_command("python script.py")
        assert exe == "python"
        assert args == ["script.py"]

    def test_command_with_multiple_args(self) -> None:
        """Test parsing command with multiple arguments."""
        exe, args = parse_command("npx -y @mcp/server-weather")
        assert exe == "npx"
        assert args == ["-y", "@mcp/server-weather"]

    def test_command_no_args(self) -> None:
        """Test parsing command with no arguments."""
        exe, args = parse_command("node")
        assert exe == "node"
        assert args == []

    def test_command_with_quotes(self) -> None:
        """Test parsing command with quoted arguments."""
        exe, args = parse_command('python -c "print(1)"')
        assert exe == "python"
        assert args == ["-c", "print(1)"]

    def test_empty_command_raises(self) -> None:
        """Test empty command raises MCPInspectorError."""
        with pytest.raises(MCPInspectorError, match="Empty command"):
            parse_command("")

    def test_uv_run_command(self) -> None:
        """Test parsing uv run command."""
        exe, args = parse_command("uv run my_server --port 8080")
        assert exe == "uv"
        assert args == ["run", "my_server", "--port", "8080"]


class TestFormatToolForDisplay:
    """Tests for format_tool_for_display function."""

    def test_basic_formatting(self, weather_tool: ToolDefinition) -> None:
        """Test basic tool formatting."""
        result = format_tool_for_display(weather_tool)
        assert "get_weather" in result
        assert "location" in result

    def test_truncates_long_description(self) -> None:
        """Test long descriptions are truncated."""
        tool = ToolDefinition(
            name="test",
            description="A" * 150,  # Very long description
            input_schema={},
        )
        result = format_tool_for_display(tool)
        assert "..." in result

    def test_shows_required_marker(self, weather_tool: ToolDefinition) -> None:
        """Test required parameters are marked."""
        result = format_tool_for_display(weather_tool)
        assert "location*" in result  # Required param has asterisk

    def test_empty_schema(self) -> None:
        """Test formatting tool with empty schema."""
        tool = ToolDefinition(name="empty", description="Empty tool", input_schema={})
        result = format_tool_for_display(tool)
        assert "empty" in result


class TestGenerateToolUsePrompt:
    """Tests for generate_tool_use_prompt function."""

    def test_generates_prompt_with_tools(self, sample_tools: list[ToolDefinition]) -> None:
        """Test prompt generation with multiple tools."""
        prompt = generate_tool_use_prompt(sample_tools)

        assert "get_weather" in prompt
        assert "search_files" in prompt
        assert "calculate" in prompt

    def test_empty_tools_message(self) -> None:
        """Test message when no tools available."""
        prompt = generate_tool_use_prompt([])
        assert "No tools are available" in prompt

    def test_includes_tool_descriptions(self, sample_tools: list[ToolDefinition]) -> None:
        """Test tool descriptions are included."""
        prompt = generate_tool_use_prompt(sample_tools)
        assert "current weather" in prompt.lower()
        assert "Search for files" in prompt


class TestValidateToolCall:
    """Tests for validate_tool_call function."""

    def test_valid_tool_call(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation of correct tool call."""
        tool_call = {
            "name": "get_weather",
            "arguments": {"location": "Paris"},
        }
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert is_valid
        assert error is None

    def test_missing_name_field(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation fails when name is missing."""
        tool_call = {"arguments": {"location": "Paris"}}
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert not is_valid
        assert "name" in error.lower()

    def test_unknown_tool_name(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation fails for unknown tool."""
        tool_call = {
            "name": "unknown_tool",
            "arguments": {},
        }
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert not is_valid
        assert "Unknown tool" in error

    def test_missing_required_parameter(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation fails when required param missing."""
        tool_call = {
            "name": "get_weather",
            "arguments": {},  # Missing required 'location'
        }
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert not is_valid
        assert "Missing required parameter" in error
        assert "location" in error

    def test_optional_parameter_not_required(self, sample_tools: list[ToolDefinition]) -> None:
        """Test optional parameters don't cause validation failure."""
        tool_call = {
            "name": "search_files",
            "arguments": {"pattern": "*.py"},  # 'directory' is optional
        }
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert is_valid

    def test_non_dict_tool_call(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation fails for non-dict input."""
        is_valid, error = validate_tool_call("not a dict", sample_tools)
        assert not is_valid
        assert "dictionary" in error

    def test_non_dict_arguments(self, sample_tools: list[ToolDefinition]) -> None:
        """Test validation fails for non-dict arguments."""
        tool_call = {
            "name": "get_weather",
            "arguments": "not a dict",
        }
        is_valid, error = validate_tool_call(tool_call, sample_tools)
        assert not is_valid
        assert "arguments" in error.lower()

    def test_tool_with_no_required_params(
        self,
        tool_with_no_required: ToolDefinition,
    ) -> None:
        """Test tool with no required params validates with empty args."""
        tool_call = {
            "name": "optional_tool",
            "arguments": {},
        }
        is_valid, error = validate_tool_call(tool_call, [tool_with_no_required])
        assert is_valid


class TestInspectMCPServer:
    """Tests for inspect_mcp_server async function."""

    @pytest.mark.asyncio
    async def test_inspect_returns_tools(self, mock_mcp_tools: list[dict[str, Any]]) -> None:
        """Test successful inspection returns ToolDefinitions."""
        from mcp_forge.tools.inspector import inspect_mcp_server

        # Create mock tool objects
        mock_tools = []
        for tool_data in mock_mcp_tools:
            mock_tool = MagicMock()
            mock_tool.name = tool_data["name"]
            mock_tool.description = tool_data["description"]
            mock_tool.inputSchema = tool_data["inputSchema"]
            mock_tools.append(mock_tool)

        # Mock the tools response
        mock_response = MagicMock()
        mock_response.tools = mock_tools

        # Mock session
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        # Mock stdio_client context manager
        mock_stdio = MagicMock()
        mock_stdio.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.__aexit__ = AsyncMock()

        with patch("mcp_forge.tools.inspector.stdio_client", return_value=mock_stdio):
            with patch("mcp_forge.tools.inspector.ClientSession", return_value=mock_session):
                tools = await inspect_mcp_server("test_command")

        assert len(tools) == 2
        assert tools[0].name == "get_weather"
        assert tools[1].name == "search_files"
        assert all(isinstance(t, ToolDefinition) for t in tools)

    @pytest.mark.asyncio
    async def test_inspect_timeout_error(self) -> None:
        """Test timeout raises MCPInspectorError."""
        import asyncio

        from mcp_forge.tools.inspector import inspect_mcp_server

        async def slow_operation(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(10)

        with patch("mcp_forge.tools.inspector.stdio_client") as mock_stdio:
            mock_stdio.return_value.__aenter__ = slow_operation

            with pytest.raises(MCPInspectorError, match="Timeout"):
                await inspect_mcp_server("slow_command", timeout=0.01)

    @pytest.mark.asyncio
    async def test_inspect_command_not_found(self) -> None:
        """Test FileNotFoundError is converted to MCPInspectorError."""
        from mcp_forge.tools.inspector import inspect_mcp_server

        with patch("mcp_forge.tools.inspector.stdio_client") as mock_stdio:
            mock_stdio.return_value.__aenter__ = AsyncMock(
                side_effect=FileNotFoundError("Command not found")
            )

            with pytest.raises(MCPInspectorError, match="Command not found"):
                await inspect_mcp_server("nonexistent_command")
