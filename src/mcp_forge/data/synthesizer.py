"""Main data synthesis orchestrator.

Coordinates seed generation, augmentation, QC validation, and data merging
for the complete synthesis pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mcp_forge.data.augmenter import AugmenterConfig, DataAugmenter
from mcp_forge.data.qc import DataQualityController, QCConfig
from mcp_forge.data.seed_generator import SeedGenerator, SeedGeneratorConfig
from mcp_forge.state import QCReport, SynthesisPlan, ToolDefinition

logger = logging.getLogger(__name__)


class SynthesisError(Exception):
    """Error during data synthesis."""

    pass


@dataclass
class SynthesisResult:
    """Result from data synthesis pipeline."""

    seed_count: int
    augmented_count: int
    total_count: int
    seed_path: Path
    augmented_path: Path
    training_path: Path
    qc_passed: bool
    qc_report: QCReport | None
    scenario_distribution: dict[str, int]


class DataSynthesizer:
    """Orchestrate the full data synthesis pipeline."""

    def __init__(
        self,
        tools: list[ToolDefinition],
        plan: SynthesisPlan,
        output_dir: Path,
        seed_config: SeedGeneratorConfig | None = None,
        augment_config: AugmenterConfig | None = None,
        qc_config: QCConfig | None = None,
    ):
        """Initialize synthesizer.

        Args:
            tools: List of tool definitions from MCP server
            plan: Synthesis plan with target counts and weights
            output_dir: Directory for output files
            seed_config: Optional seed generator config
            augment_config: Optional augmenter config
            qc_config: Optional QC config
        """
        self.tools = tools
        self.plan = plan
        self.output_dir = Path(output_dir)

        # Initialize components
        self.seed_generator = SeedGenerator(
            config=seed_config or SeedGeneratorConfig(),
            tools=tools,
        )
        self.augmenter = DataAugmenter(
            config=augment_config or AugmenterConfig(),
            tools=tools,
        )
        self.qc = DataQualityController(
            tools=tools,
            config=qc_config or QCConfig(),
        )

    async def synthesize(
        self,
        progress_callback: Callable[[str], None] | None = None,
    ) -> SynthesisResult:
        """Run full synthesis pipeline.

        Pipeline stages:
        1. Generate seed samples using GPT
        2. Validate seeds with QC
        3. Augment seeds via paraphrasing
        4. Merge seeds and augmented
        5. Final QC validation
        6. Write output files

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            SynthesisResult with all metrics and paths
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths
        seed_path = self.output_dir / "seed.jsonl"
        augmented_path = self.output_dir / "augmented.jsonl"
        training_path = self.output_dir / "train.jsonl"

        if progress_callback:
            progress_callback("Starting data synthesis pipeline...")

        # Stage 1: Generate seeds
        if progress_callback:
            progress_callback("[1/5] Generating seed samples...")

        seed_result = await self.seed_generator.generate_seeds(
            plan=self.plan,
            output_path=seed_path,
            progress_callback=progress_callback,
        )

        if seed_result.total_generated == 0:
            raise SynthesisError("No seed samples generated. Check API key and configuration.")

        if progress_callback:
            progress_callback(f"Generated {seed_result.total_generated} seed samples")

        # Stage 2: Validate seeds
        if progress_callback:
            progress_callback("[2/5] Validating seed samples...")

        seed_qc_report, valid_seeds = self.qc.validate_dataset(
            data_path=seed_path,
            output_path=None,  # Don't write yet
        )

        if not valid_seeds:
            raise SynthesisError("No valid seed samples after QC. Check generation quality.")

        valid_seed_dicts = [s.to_dict() for s in valid_seeds]

        if progress_callback:
            progress_callback(
                f"Validated {len(valid_seeds)}/{seed_result.total_generated} seeds "
                f"({seed_qc_report.schema_pass_rate:.0%} schema pass rate)"
            )

        # Stage 3: Augment seeds
        if progress_callback:
            progress_callback("[3/5] Augmenting seed samples...")

        augment_result = await self.augmenter.augment_dataset(
            seed_samples=valid_seed_dicts,
            target_total=self.plan.total_samples,
            output_path=augmented_path,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback(
                f"Generated {augment_result.total_augmented} augmented samples "
                f"(synthetic ratio: {augment_result.synthetic_ratio:.0%})"
            )

        # Stage 4: Merge and dedupe
        if progress_callback:
            progress_callback("[4/5] Merging and deduplicating...")

        merged_samples = self._merge_samples(valid_seed_dicts, augment_result.samples)

        if progress_callback:
            progress_callback(f"Merged {len(merged_samples)} total samples")

        # Write merged file
        with open(training_path, "w") as f:
            for sample in merged_samples:
                f.write(json.dumps(sample) + "\n")

        # Stage 5: Final QC validation
        if progress_callback:
            progress_callback("[5/5] Running final QC validation...")

        final_report, final_valid = self.qc.validate_dataset(
            data_path=training_path,
            output_path=training_path,  # Overwrite with cleaned version
        )

        qc_passed = final_report.passes_threshold()

        if progress_callback:
            status = "PASSED" if qc_passed else "FAILED"
            progress_callback(
                f"Final QC {status}: {final_report.valid_samples}/{final_report.total_samples} valid "
                f"({final_report.schema_pass_rate:.0%} schema pass rate)"
            )

        # Calculate scenario distribution
        scenario_distribution = final_report.scenario_coverage

        return SynthesisResult(
            seed_count=len(valid_seeds),
            augmented_count=augment_result.total_augmented,
            total_count=final_report.valid_samples,
            seed_path=seed_path,
            augmented_path=augmented_path,
            training_path=training_path,
            qc_passed=qc_passed,
            qc_report=final_report,
            scenario_distribution=scenario_distribution,
        )

    def _merge_samples(
        self,
        seed_samples: list[dict],
        augmented_samples: list[dict],
    ) -> list[dict]:
        """Merge seed and augmented samples with deduplication.

        Args:
            seed_samples: List of seed sample dicts
            augmented_samples: List of augmented sample dicts

        Returns:
            Merged and deduplicated list
        """
        import hashlib

        seen_hashes: set[str] = set()
        merged: list[dict] = []

        def compute_hash(sample: dict) -> str:
            messages = sample.get("messages", [])
            key_parts = []
            for msg in messages:
                if msg.get("role") in ("user", "assistant"):
                    key_parts.append(msg.get("content", ""))
            content = "|||".join(key_parts)
            return hashlib.sha256(content.encode()).hexdigest()[:16]

        # Add seeds first (higher priority)
        for sample in seed_samples:
            h = compute_hash(sample)
            if h not in seen_hashes:
                seen_hashes.add(h)
                merged.append(sample)

        # Add augmented samples
        for sample in augmented_samples:
            h = compute_hash(sample)
            if h not in seen_hashes:
                seen_hashes.add(h)
                merged.append(sample)

        return merged


async def run_synthesis(
    tools: list[ToolDefinition],
    plan: SynthesisPlan,
    output_dir: Path,
    progress_callback: Callable[[str], None] | None = None,
) -> SynthesisResult:
    """Standalone function to run synthesis.

    Args:
        tools: Tool definitions
        plan: Synthesis plan
        output_dir: Output directory
        progress_callback: Optional progress callback

    Returns:
        Synthesis result
    """
    synthesizer = DataSynthesizer(tools=tools, plan=plan, output_dir=output_dir)
    return await synthesizer.synthesize(progress_callback=progress_callback)
