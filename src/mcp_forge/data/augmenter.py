"""Data augmentation via paraphrasing for training data expansion.

Expands seed data through LLM-based paraphrasing with diversity controls
and model collapse prevention.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai
from openai import APIError, RateLimitError

from mcp_forge.data.formatter import (
    create_training_sample,
    format_tool_call,
    parse_tool_call,
)
from mcp_forge.state import ToolDefinition

logger = logging.getLogger(__name__)


class AugmentationError(Exception):
    """Error during data augmentation."""

    pass


@dataclass
class AugmenterConfig:
    """Configuration for data augmentation."""

    model: str = "gpt-4o"  # GPT-5 when available
    expansion_factor: int = 10
    temperature_range: tuple[float, float] = (0.7, 1.0)
    max_synthetic_ratio: float = 0.30  # Prevent model collapse
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 5
    request_timeout: float = 60.0
    similarity_threshold: float = 0.85  # For deduplication


@dataclass
class AugmentationResult:
    """Result from data augmentation."""

    samples: list[dict[str, Any]]
    total_augmented: int
    duplicates_removed: int
    failed_attempts: int
    synthetic_ratio: float
    scenario_counts: dict[str, int] = field(default_factory=dict)


class DataAugmenter:
    """Expand training data through paraphrasing and parameter variation."""

    PARAPHRASE_PROMPT = """You are helping to create training data for an AI assistant.

Paraphrase the following user message while keeping the same intent.
The paraphrase should be natural and varied in wording but request the same action.

Original message: "{original}"

Rules:
1. Keep the same meaning and required parameters
2. Use different words and sentence structure
3. Vary the formality level slightly
4. Do NOT change the tool or action being requested
5. Return ONLY the paraphrased message, nothing else"""

    PARAMETER_VARIATION_PROMPT = """You are helping to create training data for an AI assistant.

Create a variation of this tool call with different parameter values but the same structure.
The original tool call is for: {tool_name}

Original arguments: {original_args}

Parameter schema: {schema}

Generate new realistic parameter values that are different from the original.
Return ONLY valid JSON with the new arguments, nothing else."""

    def __init__(
        self,
        config: AugmenterConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ):
        """Initialize augmenter.

        Args:
            config: Augmenter configuration
            tools: List of available tool definitions for schema access
        """
        self.config = config or AugmenterConfig()
        self.tools = tools or []
        self.tool_schemas = {t.name: t.input_schema for t in self.tools}
        self._client: openai.AsyncOpenAI | None = None
        self._seen_hashes: set[str] = set()

    @property
    def client(self) -> openai.AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise AugmentationError("OPENAI_API_KEY environment variable not set")
            self._client = openai.AsyncOpenAI(api_key=api_key)
        return self._client

    async def augment_dataset(
        self,
        seed_samples: list[dict[str, Any]],
        target_total: int,
        output_path: Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AugmentationResult:
        """Expand seed samples via paraphrasing.

        Args:
            seed_samples: Original seed samples to augment
            target_total: Target total samples (seeds + augmented)
            output_path: Path to write augmented samples
            progress_callback: Optional callback for progress updates

        Returns:
            AugmentationResult with generated samples and metrics
        """
        if not seed_samples:
            raise AugmentationError("No seed samples provided for augmentation")

        # Calculate how many augmented samples we need
        augmented_needed = target_total - len(seed_samples)

        # Check synthetic ratio limit
        max_augmented = int(target_total * self.config.max_synthetic_ratio)
        if augmented_needed > max_augmented:
            logger.warning(
                f"Requested {augmented_needed} augmented samples exceeds "
                f"max synthetic ratio ({self.config.max_synthetic_ratio}). "
                f"Capping at {max_augmented}."
            )
            augmented_needed = max_augmented

        if progress_callback:
            progress_callback(f"Augmenting {len(seed_samples)} seeds to {target_total} total...")

        # Initialize tracking
        augmented_samples: list[dict[str, Any]] = []
        failed_attempts = 0
        duplicates_removed = 0
        scenario_counts: dict[str, int] = {}

        # Initialize seen hashes with seed samples
        self._seen_hashes = set()
        for sample in seed_samples:
            self._seen_hashes.add(self._compute_hash(sample))
            scenario = sample.get("scenario", "standard")
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        # Calculate per-sample expansion
        samples_per_seed = max(1, augmented_needed // len(seed_samples))
        remaining = augmented_needed

        # Process seeds in batches
        for i, seed in enumerate(seed_samples):
            if remaining <= 0:
                break

            # How many augmented samples for this seed
            batch_size = min(samples_per_seed, remaining)

            if progress_callback and i % 10 == 0:
                progress_callback(
                    f"  Processing seed {i+1}/{len(seed_samples)} "
                    f"({len(augmented_samples)} augmented so far)"
                )

            # Generate augmented versions
            for _ in range(batch_size):
                try:
                    augmented = await self._augment_single(seed)
                    if augmented:
                        # Check for duplicates
                        sample_hash = self._compute_hash(augmented)
                        if sample_hash in self._seen_hashes:
                            duplicates_removed += 1
                            continue

                        self._seen_hashes.add(sample_hash)
                        augmented_samples.append(augmented)
                        remaining -= 1

                        # Track scenario
                        scenario = augmented.get("scenario", "standard")
                        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

                except Exception as e:
                    logger.warning(f"Failed to augment sample: {e}")
                    failed_attempts += 1

        # Calculate synthetic ratio
        total_samples = len(seed_samples) + len(augmented_samples)
        synthetic_ratio = len(augmented_samples) / total_samples if total_samples > 0 else 0

        # Write augmented samples to output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for sample in augmented_samples:
                f.write(json.dumps(sample) + "\n")

        if progress_callback:
            progress_callback(
                f"Generated {len(augmented_samples)} augmented samples "
                f"(synthetic ratio: {synthetic_ratio:.1%})"
            )

        return AugmentationResult(
            samples=augmented_samples,
            total_augmented=len(augmented_samples),
            duplicates_removed=duplicates_removed,
            failed_attempts=failed_attempts,
            synthetic_ratio=synthetic_ratio,
            scenario_counts=scenario_counts,
        )

    async def _augment_single(
        self, seed_sample: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Augment a single seed sample.

        Args:
            seed_sample: Original seed sample to augment

        Returns:
            Augmented sample or None on failure
        """
        # Determine augmentation strategy
        scenario = seed_sample.get("scenario", "standard")

        if scenario == "no_tool":
            # For no-tool samples, just paraphrase the user message
            return await self._paraphrase_query(seed_sample)
        else:
            # For tool samples, randomly choose between strategies
            strategy = random.choice(["paraphrase", "parameter_variation"])
            if strategy == "paraphrase":
                return await self._paraphrase_query(seed_sample)
            else:
                return await self._vary_parameters(seed_sample)

    async def _paraphrase_query(
        self, seed_sample: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Generate paraphrased version of user query.

        Args:
            seed_sample: Original sample

        Returns:
            New sample with paraphrased query
        """
        # Extract user message from seed
        messages = seed_sample.get("messages", [])
        user_message = ""
        assistant_response = ""

        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
            elif msg.get("role") == "assistant":
                assistant_response = msg.get("content", "")

        if not user_message:
            return None

        # Get paraphrased query from GPT
        prompt = self.PARAPHRASE_PROMPT.format(original=user_message)

        temperature = random.uniform(*self.config.temperature_range)

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=self.config.request_timeout,
                )

                paraphrased = response.choices[0].message.content
                if not paraphrased:
                    continue

                # Clean up response
                paraphrased = paraphrased.strip().strip('"').strip("'")

                # Create new sample with paraphrased query
                sample_id = f"aug_{uuid.uuid4().hex[:8]}"

                return create_training_sample(
                    sample_id=sample_id,
                    source="augmented",
                    scenario=seed_sample.get("scenario", "standard"),
                    tool_name=seed_sample.get("tool_name"),
                    user_message=paraphrased,
                    assistant_response=assistant_response,
                    tools=[ToolDefinition.from_dict(t) for t in self._get_tools_from_sample(seed_sample)],
                )

            except (APIError, RateLimitError):
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                else:
                    raise

        return None

    async def _vary_parameters(
        self, seed_sample: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Create variation with different tool parameters.

        Args:
            seed_sample: Original sample

        Returns:
            New sample with varied parameters
        """
        tool_name = seed_sample.get("tool_name")
        if not tool_name:
            return await self._paraphrase_query(seed_sample)

        # Extract tool call from assistant response
        messages = seed_sample.get("messages", [])
        assistant_content = ""
        user_message = ""

        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_content = msg.get("content", "")
            elif msg.get("role") == "user":
                user_message = msg.get("content", "")

        tool_call = parse_tool_call(assistant_content)
        if not tool_call:
            return await self._paraphrase_query(seed_sample)

        # Get schema for this tool
        schema = self.tool_schemas.get(tool_name, {})

        # Generate varied parameters
        prompt = self.PARAMETER_VARIATION_PROMPT.format(
            tool_name=tool_name,
            original_args=json.dumps(tool_call.get("arguments", {})),
            schema=json.dumps(schema, indent=2),
        )

        temperature = random.uniform(*self.config.temperature_range)

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=self.config.request_timeout,
                )

                content = response.choices[0].message.content
                if not content:
                    continue

                # Parse new arguments
                try:
                    # Clean up potential markdown
                    clean_content = content
                    if "```json" in content:
                        clean_content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        clean_content = content.split("```")[1].split("```")[0]

                    new_args = json.loads(clean_content.strip())
                except json.JSONDecodeError:
                    continue

                # Create new assistant response with varied parameters
                new_tool_call = format_tool_call(tool_name, new_args)

                sample_id = f"aug_{uuid.uuid4().hex[:8]}"

                return create_training_sample(
                    sample_id=sample_id,
                    source="augmented",
                    scenario=seed_sample.get("scenario", "standard"),
                    tool_name=tool_name,
                    user_message=user_message,
                    assistant_response=new_tool_call,
                    tools=[ToolDefinition.from_dict(t) for t in self._get_tools_from_sample(seed_sample)],
                )

            except (APIError, RateLimitError):
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                else:
                    raise

        return None

    def _get_tools_from_sample(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract tool definitions from sample's system message.

        Args:
            sample: Training sample

        Returns:
            List of tool definition dicts
        """
        # For simplicity, return our configured tools as dicts
        return [t.to_dict() for t in self.tools]

    def _compute_hash(self, sample: dict[str, Any]) -> str:
        """Compute content hash for deduplication.

        Args:
            sample: Training sample

        Returns:
            Hash string
        """
        messages = sample.get("messages", [])
        key_parts = []

        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                key_parts.append(msg.get("content", ""))

        content = "|||".join(key_parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


async def augment_dataset_standalone(
    seed_samples: list[dict[str, Any]],
    target_total: int,
    output_path: Path,
    tools: list[ToolDefinition] | None = None,
    config: AugmenterConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> AugmentationResult:
    """Standalone function to augment dataset.

    Args:
        seed_samples: List of seed samples
        target_total: Target total samples
        output_path: Output file path
        tools: Optional tool definitions
        config: Optional augmenter config
        progress_callback: Optional progress callback

    Returns:
        Augmentation result
    """
    augmenter = DataAugmenter(config=config, tools=tools)
    return await augmenter.augment_dataset(
        seed_samples, target_total, output_path, progress_callback
    )
