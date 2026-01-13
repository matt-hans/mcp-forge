"""Hermes ChatML formatter for tool-calling training data.

Formats training samples according to the Hermes function-calling format
with <tools> XML blocks and <tool_call> tags.
"""

from __future__ import annotations

import json
from typing import Any

from mcp_forge.state import ToolDefinition


def format_system_prompt(tools: list[ToolDefinition]) -> str:
    """Generate Hermes-style system prompt with <tools> XML block.

    Args:
        tools: List of tool definitions to include in the prompt

    Returns:
        Formatted system prompt string with tool descriptions
    """
    if not tools:
        return "You are a helpful assistant."

    # Build JSON tool definitions for Hermes format
    tool_defs = []
    for tool in tools:
        tool_def = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        tool_defs.append(tool_def)

    tools_json = json.dumps(tool_defs, indent=2)

    return f"""You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. Here are the available tools:

<tools>
{tools_json}
</tools>

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg1>": "<value1>"}}}}
</tool_call>"""


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Wrap tool call in <tool_call> XML tags.

    Args:
        name: Name of the tool/function being called
        arguments: Dictionary of arguments to pass to the tool

    Returns:
        Formatted tool call string with XML tags
    """
    tool_call = {"name": name, "arguments": arguments}
    return f"<tool_call>\n{json.dumps(tool_call, indent=2)}\n</tool_call>"


def format_tool_response(response: dict[str, Any] | str) -> str:
    """Format tool execution response.

    Args:
        response: Tool response as dict or string

    Returns:
        Formatted tool response string with XML tags
    """
    if isinstance(response, str):
        content = response
    else:
        content = json.dumps(response, indent=2)

    return f"<tool_response>\n{content}\n</tool_response>"


def create_training_sample(
    sample_id: str,
    source: str,
    scenario: str,
    tool_name: str | None,
    user_message: str,
    assistant_response: str,
    tool_response: str | None = None,
    tools: list[ToolDefinition] | None = None,
) -> dict[str, Any]:
    """Create complete JSONL-ready training sample.

    Args:
        sample_id: Unique identifier for this sample
        source: Origin of sample ("seed" or "augmented")
        scenario: Scenario type (standard, no_tool, error, ambiguous, edge)
        tool_name: Name of tool used (None for no_tool scenarios)
        user_message: The user's input message
        assistant_response: The assistant's response (may include tool call)
        tool_response: Optional tool response for multi-turn conversations
        tools: Optional list of tools for system prompt (uses default if None)

    Returns:
        Dictionary ready to be serialized as JSONL line
    """
    messages: list[dict[str, Any]] = []

    # System message with tools
    if tools:
        system_prompt = format_system_prompt(tools)
        messages.append({"role": "system", "content": system_prompt})

    # User message
    messages.append({"role": "user", "content": user_message})

    # Assistant response
    messages.append({"role": "assistant", "content": assistant_response})

    # Optional tool response and follow-up
    if tool_response:
        messages.append({"role": "tool", "content": tool_response})

    return {
        "id": sample_id,
        "source": source,
        "scenario": scenario,
        "tool_name": tool_name,
        "messages": messages,
    }


def parse_tool_call(content: str) -> dict[str, Any] | None:
    """Extract tool call from assistant response content.

    Args:
        content: Assistant response that may contain <tool_call> tags

    Returns:
        Parsed tool call dict with 'name' and 'arguments', or None if not found
    """
    import re

    # Try to find <tool_call> XML tags
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return None


def validate_sample_format(sample: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate that a sample has correct format.

    Args:
        sample: Training sample dictionary

    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["id", "source", "scenario", "messages"]

    for field in required_fields:
        if field not in sample:
            return False, f"Missing required field: {field}"

    # Validate source
    if sample["source"] not in ("seed", "augmented"):
        return False, f"Invalid source: {sample['source']} (must be 'seed' or 'augmented')"

    # Validate scenario
    valid_scenarios = {"standard", "no_tool", "error", "ambiguous", "edge"}
    if sample["scenario"] not in valid_scenarios:
        return False, f"Invalid scenario: {sample['scenario']}"

    # Validate messages structure
    messages = sample.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        return False, "Messages must be a list with at least 2 entries"

    # Check message roles
    valid_roles = {"system", "user", "assistant", "tool"}
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"Message {i} is not a dictionary"
        if "role" not in msg:
            return False, f"Message {i} missing 'role'"
        if msg["role"] not in valid_roles:
            return False, f"Message {i} has invalid role: {msg['role']}"
        if "content" not in msg:
            return False, f"Message {i} missing 'content'"

    # For non-no_tool scenarios, verify tool call exists
    if sample["scenario"] != "no_tool":
        has_tool_call = False
        for msg in messages:
            if msg["role"] == "assistant":
                if "<tool_call>" in msg.get("content", ""):
                    has_tool_call = True
                    break

        if not has_tool_call:
            return False, "Non-no_tool scenario missing tool call in assistant response"

    return True, None
