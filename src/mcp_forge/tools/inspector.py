"""MCP server inspection and tool schema extraction."""

from __future__ import annotations

import asyncio
import shlex
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console

from mcp_forge.state import ToolDefinition

console = Console()

# Timeout for MCP server operations
MCP_TIMEOUT_SECONDS = 30


class MCPInspectorError(Exception):
    """Error during MCP server inspection."""
    pass


def parse_command(command: str) -> tuple[str, list[str]]:
    """Parse a command string into executable and arguments.

    Handles common patterns:
    - "npx -y @modelcontextprotocol/server-weather"
    - "python server.py"
    - "uv run my_server"
    """
    parts = shlex.split(command)
    if not parts:
        raise MCPInspectorError("Empty command string")

    return parts[0], parts[1:] if len(parts) > 1 else []


async def inspect_mcp_server(
    command: str,
    timeout: float = MCP_TIMEOUT_SECONDS
) -> list[ToolDefinition]:
    """Connect to an MCP server and extract tool definitions.

    Args:
        command: Full command to start the MCP server (e.g., "npx -y @mcp/server-weather")
        timeout: Maximum time to wait for server response

    Returns:
        List of ToolDefinition objects for each tool the server exposes

    Raises:
        MCPInspectorError: If connection fails or server doesn't respond
    """
    executable, args = parse_command(command)

    server_params = StdioServerParameters(
        command=executable,
        args=args,
    )

    async def _inspect() -> list[ToolDefinition]:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize connection
                await session.initialize()

                # Fetch tool list
                tools_response = await session.list_tools()

                # Convert to our ToolDefinition format
                tools = []
                for tool in tools_response.tools:
                    tools.append(ToolDefinition(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema if hasattr(tool, "inputSchema") else {},
                    ))

                return tools

    try:
        if hasattr(asyncio, "timeout"):
            async with asyncio.timeout(timeout):
                return await _inspect()
        return await asyncio.wait_for(_inspect(), timeout=timeout)

    except asyncio.TimeoutError as e:
        raise MCPInspectorError(
            f"Timeout after {timeout}s waiting for MCP server. "
            "Check that the server command is correct and the server starts quickly."
        ) from e
    except FileNotFoundError as e:
        raise MCPInspectorError(
            f"Command not found: '{executable}'. "
            "Ensure the command is installed and in your PATH."
        ) from e
    except Exception as e:
        raise MCPInspectorError(f"Failed to connect to MCP server: {e}") from e


def format_tool_for_display(tool: ToolDefinition) -> str:
    """Format a tool definition for console display."""
    lines = [f"  [bold]{tool.name}[/bold]"]
    if tool.description:
        # Truncate long descriptions
        desc = tool.description
        if len(desc) > 100:
            desc = desc[:97] + "..."
        lines.append(f"    {desc}")

    # Show parameters if available
    if tool.input_schema and "properties" in tool.input_schema:
        props = tool.input_schema["properties"]
        required = tool.input_schema.get("required", [])

        params = []
        for name, schema in props.items():
            param_type = schema.get("type", "any")
            is_required = name in required
            marker = "*" if is_required else ""
            params.append(f"{name}{marker}: {param_type}")

        if params:
            lines.append(f"    Parameters: {', '.join(params)}")

    return "\n".join(lines)


def generate_tool_use_prompt(tools: list[ToolDefinition]) -> str:
    """Generate a system prompt section describing available tools.

    This is used as part of the system prompt during data synthesis
    and model inference.
    """
    if not tools:
        return "No tools are available."

    lines = ["You have access to the following tools:\n"]

    for tool in tools:
        lines.append(f"### {tool.name}")
        if tool.description:
            lines.append(tool.description)

        if tool.input_schema:
            import json
            lines.append(f"Input schema: {json.dumps(tool.input_schema, indent=2)}")

        lines.append("")

    lines.append("To use a tool, respond with a tool call in the appropriate format.")

    return "\n".join(lines)


def validate_tool_call(
    tool_call: dict[str, Any],
    tools: list[ToolDefinition]
) -> tuple[bool, str | None]:
    """Validate that a tool call matches available tools.

    Args:
        tool_call: Dictionary with 'name' and 'arguments' keys
        tools: List of available tool definitions

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(tool_call, dict):
        return False, "Tool call must be a dictionary"

    if "name" not in tool_call:
        return False, "Tool call missing 'name' field"

    tool_name = tool_call["name"]
    matching_tool = None
    for tool in tools:
        if tool.name == tool_name:
            matching_tool = tool
            break

    if matching_tool is None:
        available = [t.name for t in tools]
        return False, f"Unknown tool '{tool_name}'. Available: {available}"

    arguments = tool_call.get("arguments", {})
    if not isinstance(arguments, dict):
        return False, "Tool call 'arguments' must be a dictionary"

    # Validate required parameters
    schema = matching_tool.input_schema
    if schema and "required" in schema:
        for required_param in schema["required"]:
            if required_param not in arguments:
                return False, f"Missing required parameter: {required_param}"

    return True, None
