"""GPT-5 seed generation for training data.

Uses OpenAI GPT-5 with function calling to generate diverse seed training
samples across all scenario types.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import openai
from openai import APIError, RateLimitError

from mcp_forge.data.formatter import create_training_sample, format_tool_call
from mcp_forge.state import SynthesisPlan, ToolDefinition

logger = logging.getLogger(__name__)


class SeedGenerationError(Exception):
    """Error during seed generation."""

    pass


@dataclass
class SeedGeneratorConfig:
    """Configuration for seed generation."""

    model: str = "gpt-4o"  # GPT-5 when available, fallback to gpt-4o
    temperature: float = 0.8
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10
    request_timeout: float = 60.0


@dataclass
class SeedGenerationResult:
    """Result from seed generation."""

    samples: list[dict[str, Any]]
    total_generated: int
    failed_attempts: int
    scenario_counts: dict[str, int] = field(default_factory=dict)


class SeedGenerator:
    """Generate seed training samples using GPT-5 function calling."""

    # Scenario-specific prompt templates
    SCENARIO_PROMPTS = {
        "standard": (
            "Generate a realistic user query that requires using the {tool_name} tool. "
            "The query should be clear and specific, requiring the tool to answer."
        ),
        "no_tool": (
            "Generate a conversational query that does NOT require any tool to answer. "
            "This could be a greeting, general knowledge question, or simple chat. "
            "The assistant should respond directly without calling any tool."
        ),
        "error": (
            "Generate a user query for {tool_name} that has missing or invalid parameters. "
            "The user might be unclear about required fields or provide wrong types. "
            "The assistant should still attempt the tool call but handle the error gracefully."
        ),
        "ambiguous": (
            "Generate an unclear or ambiguous query where it's uncertain whether "
            "{tool_name} should be used. The assistant might need to ask for clarification "
            "or make a reasonable assumption."
        ),
        "edge": (
            "Generate an edge case query for {tool_name} that tests boundary conditions. "
            "This could involve unusual parameter values, empty inputs, or extreme cases."
        ),
    }

    def __init__(
        self,
        config: SeedGeneratorConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ):
        """Initialize seed generator.

        Args:
            config: Generator configuration
            tools: List of available tool definitions
        """
        self.config = config or SeedGeneratorConfig()
        self.tools = tools or []
        self._client: openai.AsyncOpenAI | None = None

    @property
    def client(self) -> openai.AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise SeedGenerationError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Set it with: export OPENAI_API_KEY=sk-..."
                )
            self._client = openai.AsyncOpenAI(api_key=api_key)
        return self._client

    async def generate_seeds(
        self,
        plan: SynthesisPlan,
        output_path: Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> SeedGenerationResult:
        """Generate seed samples according to synthesis plan.

        Args:
            plan: Synthesis plan with target counts and weights
            output_path: Path to write seed samples
            progress_callback: Optional callback for progress updates

        Returns:
            SeedGenerationResult with generated samples
        """
        samples: list[dict[str, Any]] = []
        failed_attempts = 0
        scenario_counts: dict[str, int] = {s: 0 for s in plan.scenario_weights}

        # Calculate target samples per scenario
        targets = plan.get_samples_per_scenario()

        # Adjust targets to match seed_samples total
        total_target = sum(targets.values())
        if total_target > 0:
            scale = plan.seed_samples / total_target
            targets = {k: max(1, int(v * scale)) for k, v in targets.items()}

        if progress_callback:
            progress_callback(f"Generating {plan.seed_samples} seed samples...")

        # Generate samples for each scenario
        for scenario, target_count in targets.items():
            if progress_callback:
                progress_callback(f"  Generating {target_count} {scenario} samples...")

            for _ in range(target_count):
                try:
                    sample = await self._generate_single_seed(scenario)
                    if sample:
                        samples.append(sample)
                        scenario_counts[scenario] += 1
                except Exception as e:
                    logger.warning(f"Failed to generate {scenario} sample: {e}")
                    failed_attempts += 1

        # Write samples to output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        if progress_callback:
            progress_callback(f"Generated {len(samples)} seed samples")

        return SeedGenerationResult(
            samples=samples,
            total_generated=len(samples),
            failed_attempts=failed_attempts,
            scenario_counts=scenario_counts,
        )

    async def _generate_single_seed(self, scenario: str) -> dict[str, Any] | None:
        """Generate a single seed sample for the given scenario.

        Args:
            scenario: The scenario type to generate

        Returns:
            Generated sample dictionary or None on failure
        """
        # Select a tool for this sample (random for standard scenarios)
        tool: ToolDefinition | None = None
        if scenario != "no_tool" and self.tools:
            import random

            tool = random.choice(self.tools)

        # Build the generation prompt
        prompt = self._create_seed_prompt(scenario, tool)

        # Call GPT with retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = await self._call_gpt(prompt, tool)
                return self._parse_gpt_response(response, scenario, tool)
            except (APIError, RateLimitError) as e:
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    logger.warning(f"API error, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise

        return None

    def _create_seed_prompt(
        self, scenario: str, tool: ToolDefinition | None
    ) -> list[dict[str, str]]:
        """Create prompt for GPT to generate a training example.

        Args:
            scenario: Scenario type to generate
            tool: Optional tool for tool-based scenarios

        Returns:
            List of message dicts for GPT API
        """
        # Get scenario-specific instruction
        template = self.SCENARIO_PROMPTS.get(scenario, self.SCENARIO_PROMPTS["standard"])
        if tool:
            instruction = template.format(tool_name=tool.name)
        else:
            instruction = template.format(tool_name="any available tool")

        # Build tool description for context
        tool_context = ""
        if tool:
            tool_context = f"""
The tool to use is:
- Name: {tool.name}
- Description: {tool.description}
- Parameters: {json.dumps(tool.input_schema, indent=2)}
"""

        system_message = f"""You are generating training data for an AI assistant that can use tools.

{instruction}

{tool_context}

Generate a realistic conversation with:
1. A user message (the query)
2. An assistant response that appropriately handles the query

For tool-calling scenarios, the assistant response should include a tool call in this format:
<tool_call>
{{"name": "tool_name", "arguments": {{"param": "value"}}}}
</tool_call>

For no_tool scenarios, the assistant should respond directly without any tool call.

Return ONLY valid JSON in this format:
{{
  "user_message": "the user's query",
  "assistant_response": "the assistant's response (with tool_call if needed)"
}}"""

        return [{"role": "system", "content": system_message}]

    async def _call_gpt(
        self, messages: list[dict[str, str]], tool: ToolDefinition | None
    ) -> Any:
        """Call GPT API with optional function calling.

        Args:
            messages: Message list for the API
            tool: Optional tool definition for function calling

        Returns:
            GPT response object
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "timeout": self.config.request_timeout,
        }

        # Add function definitions if we have tools
        if tool and self.tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                        "strict": True,
                    },
                }
                for t in self.tools
            ]
            kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**kwargs)
        return response

    def _parse_gpt_response(
        self,
        response: Any,
        scenario: str,
        tool: ToolDefinition | None,
    ) -> dict[str, Any]:
        """Parse GPT response into training sample format.

        Args:
            response: GPT API response
            scenario: Scenario type being generated
            tool: Tool used for this sample

        Returns:
            Formatted training sample dictionary
        """
        sample_id = f"seed_{uuid.uuid4().hex[:8]}"
        tool_name = tool.name if tool else None

        # Extract content from response
        message = response.choices[0].message
        content = message.content or ""

        # Try to parse as JSON
        try:
            # Clean up potential markdown code blocks
            clean_content = content
            if "```json" in content:
                clean_content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                clean_content = content.split("```")[1].split("```")[0]

            data = json.loads(clean_content.strip())
            user_message = data.get("user_message", "")
            assistant_response = data.get("assistant_response", "")
        except json.JSONDecodeError:
            # Fallback: use content directly
            user_message = f"Help me with {tool.name}" if tool else "Hello"
            assistant_response = content

        # If GPT made a function call, convert it to our format
        if message.tool_calls:
            tc = message.tool_calls[0]
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            assistant_response = format_tool_call(tool_name, args)

        return create_training_sample(
            sample_id=sample_id,
            source="seed",
            scenario=scenario,
            tool_name=tool_name,
            user_message=user_message,
            assistant_response=assistant_response,
            tools=self.tools,
        )


async def generate_seeds_standalone(
    tools: list[ToolDefinition],
    plan: SynthesisPlan,
    output_path: Path,
    config: SeedGeneratorConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> SeedGenerationResult:
    """Standalone function to generate seeds.

    Args:
        tools: List of tool definitions
        plan: Synthesis plan
        output_path: Output file path
        config: Optional generator config
        progress_callback: Optional progress callback

    Returns:
        Generation result
    """
    generator = SeedGenerator(config=config, tools=tools)
    return await generator.generate_seeds(plan, output_path, progress_callback)
