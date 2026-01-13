"""Data Quality Control engine for training data validation.

v1.1: Mandatory QA gate with schema validation, deduplication, and coverage analysis.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from mcp_forge.state import QCReport, Scenario, ToolDefinition

console = Console()


@dataclass
class QCConfig:
    """Configuration for quality control checks."""

    schema_pass_threshold: float = 0.98
    min_samples_per_tool: int = 10
    dedup_enabled: bool = True
    require_scenario_coverage: bool = True
    auto_repair: bool = True  # Attempt to fix minor issues
    max_response_length: int = 4096  # Max assistant response length

    # Scenario targets (used for coverage checking)
    scenario_targets: dict[str, float] = field(default_factory=lambda: {
        "standard": 0.60,
        "no_tool": 0.15,
        "error": 0.10,
        "ambiguous": 0.10,
        "edge": 0.05
    })


@dataclass
class RepairStats:
    """Statistics about repair operations during QC."""

    repairs_attempted: int = 0
    repairs_successful: int = 0
    repair_details: list[dict[str, Any]] = field(default_factory=list)

    def record(self, sample_id: str, repair_type: str, success: bool, details: str = "") -> None:
        """Record a repair attempt."""
        self.repairs_attempted += 1
        if success:
            self.repairs_successful += 1
        self.repair_details.append({
            "sample_id": sample_id,
            "repair_type": repair_type,
            "success": success,
            "details": details,
        })

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "repairs_attempted": self.repairs_attempted,
            "repairs_successful": self.repairs_successful,
            "repair_details": self.repair_details,
        }


@dataclass
class QCIssue:
    """A single quality control issue."""

    sample_id: str
    issue_type: str  # schema_error, duplicate, missing_field, invalid_tool
    message: str
    severity: str  # error, warning
    repairable: bool = False
    repair_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "issue_type": self.issue_type,
            "message": self.message,
            "severity": self.severity,
            "repairable": self.repairable,
            "repair_action": self.repair_action
        }


@dataclass
class ValidatedSample:
    """A sample that has passed QC validation."""

    id: str
    source: str
    scenario: str
    tool_name: str | None
    messages: list[dict[str, Any]]
    content_hash: str  # For dedup

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "scenario": self.scenario,
            "tool_name": self.tool_name,
            "messages": self.messages,
            "qc_passed": True
        }


class QCFailedError(Exception):
    """Raised when QC validation fails and blocks pipeline progression.

    Provides detailed information about which thresholds failed
    and suggestions for remediation.
    """

    def __init__(
        self,
        report: "QCReport",
        threshold: float,
        min_samples: int,
        message: str | None = None,
    ):
        self.report = report
        self.threshold = threshold
        self.min_samples = min_samples

        # Build detailed error message
        failures = []
        if report.schema_pass_rate < threshold:
            failures.append(
                f"Schema pass rate {report.schema_pass_rate:.1%} < threshold {threshold:.1%}"
            )

        for tool, count in report.tool_coverage.items():
            if count < min_samples:
                failures.append(
                    f"Tool '{tool}' has {count} samples < minimum {min_samples}"
                )

        self.failures = failures
        self.remediation = [
            "Run with --fix to auto-repair fixable issues",
            "Lower threshold with --threshold <value>",
            "Generate more training data for low-coverage tools",
            "Review and fix schema errors manually",
        ]

        if message:
            super().__init__(message)
        else:
            super().__init__(
                f"QC validation failed: {len(failures)} threshold(s) not met. "
                f"Run 'mcp-forge qa --data <file> --tools <file>' for details."
            )

    def get_sample_issues(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get sample of issues for display."""
        return self.report.issues[:limit]

    def format_error(self) -> str:
        """Format error with details for terminal display."""
        lines = [
            "QC Validation Failed",
            "=" * 40,
            "",
            "Failures:",
        ]
        for failure in self.failures:
            lines.append(f"  - {failure}")

        lines.extend([
            "",
            "Sample issues:",
        ])
        for issue in self.get_sample_issues():
            lines.append(f"  [{issue['severity']}] {issue['issue_type']}: {issue['message']}")

        lines.extend([
            "",
            "Remediation suggestions:",
        ])
        for suggestion in self.remediation:
            lines.append(f"  - {suggestion}")

        return "\n".join(lines)


class DataQualityController:
    """Validates and cleans training data."""

    def __init__(
        self,
        tools: list[ToolDefinition],
        config: QCConfig | None = None
    ):
        self.tools = tools
        self.tool_names = {t.name for t in tools}
        self.tool_schemas = {t.name: t.input_schema for t in tools}
        self.config = config or QCConfig()

        # Track state
        self.issues: list[QCIssue] = []
        self.seen_hashes: set[str] = set()
        self.repair_stats = RepairStats()

    def validate_dataset(
        self,
        data_path: Path,
        output_path: Path | None = None,
        dry_run: bool = False,
    ) -> tuple[QCReport, list[ValidatedSample]]:
        """Validate entire dataset and optionally write cleaned version.

        Args:
            data_path: Path to input JSONL file
            output_path: Optional path to write validated samples
            dry_run: If True, don't write files (preview repairs only)

        Returns:
            Tuple of (QCReport, list of validated samples)
        """
        self.issues = []
        self.seen_hashes = set()
        self.repair_stats = RepairStats()

        validated_samples: list[ValidatedSample] = []
        tool_coverage: dict[str, int] = defaultdict(int)
        scenario_coverage: dict[str, int] = defaultdict(int)

        total_samples = 0
        dropped_samples = 0

        with open(data_path) as f:
            lines = f.readlines()

        with Progress() as progress:
            task = progress.add_task("Validating samples...", total=len(lines))

            for line in lines:
                total_samples += 1
                progress.advance(task)

                try:
                    sample = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    self.issues.append(QCIssue(
                        sample_id=f"line_{total_samples}",
                        issue_type="json_error",
                        message=f"Invalid JSON: {e}",
                        severity="error"
                    ))
                    dropped_samples += 1
                    continue

                # Validate sample
                validated = self._validate_sample(sample)

                if validated:
                    validated_samples.append(validated)
                    if validated.tool_name:
                        tool_coverage[validated.tool_name] += 1
                    scenario_coverage[validated.scenario] += 1
                else:
                    dropped_samples += 1

        # Check coverage requirements
        self._check_coverage(tool_coverage, scenario_coverage, total_samples)

        # Calculate metrics
        valid_samples = len(validated_samples)
        schema_errors = sum(1 for i in self.issues if i.issue_type == "schema_error")
        schema_pass_rate = (total_samples - schema_errors) / total_samples if total_samples > 0 else 0

        duplicates = sum(1 for i in self.issues if i.issue_type == "duplicate")
        dedup_rate = duplicates / total_samples if total_samples > 0 else 0

        # Build report
        report = QCReport(
            total_samples=total_samples,
            valid_samples=valid_samples,
            dropped_samples=dropped_samples,
            schema_pass_rate=schema_pass_rate,
            dedup_rate=dedup_rate,
            tool_coverage=dict(tool_coverage),
            scenario_coverage=dict(scenario_coverage),
            issues=[i.to_dict() for i in self.issues]
        )

        # Write cleaned output if requested (unless dry_run)
        if output_path and not dry_run:
            with open(output_path, "w") as f:
                for sample in validated_samples:
                    f.write(json.dumps(sample.to_dict()) + "\n")

        return report, validated_samples

    def _validate_sample(self, sample: dict[str, Any]) -> ValidatedSample | None:
        """Validate a single sample.

        Returns ValidatedSample if valid, None if should be dropped.
        """
        sample_id = sample.get("id", "unknown")

        # Check required fields
        required_fields = ["id", "source", "messages"]
        for fld in required_fields:
            if fld not in sample:
                self.issues.append(QCIssue(
                    sample_id=sample_id,
                    issue_type="missing_field",
                    message=f"Missing required field: {fld}",
                    severity="error"
                ))
                return None

        messages = sample.get("messages", [])
        scenario = sample.get("scenario", "standard")
        tool_name = sample.get("tool_name")

        # Validate scenario
        if scenario not in [s.value for s in Scenario]:
            self.issues.append(QCIssue(
                sample_id=sample_id,
                issue_type="invalid_scenario",
                message=f"Unknown scenario: {scenario}",
                severity="warning",
                repairable=True,
                repair_action="Set to 'standard'"
            ))
            if self.config.auto_repair:
                scenario = "standard"
            else:
                return None

        # For non-no_tool scenarios, validate tool call
        if scenario != "no_tool":
            tool_call = self._extract_tool_call(messages)

            if tool_call is None:
                self.issues.append(QCIssue(
                    sample_id=sample_id,
                    issue_type="missing_tool_call",
                    message="No tool call found in assistant message",
                    severity="error"
                ))
                return None

            # Validate tool name exists
            if tool_call.get("name") not in self.tool_names:
                self.issues.append(QCIssue(
                    sample_id=sample_id,
                    issue_type="invalid_tool",
                    message=f"Unknown tool: {tool_call.get('name')}",
                    severity="error"
                ))
                return None

            # Validate arguments against schema
            tool_name = tool_call["name"]
            schema = self.tool_schemas.get(tool_name, {})

            if schema:
                valid, error = self._validate_schema(tool_call.get("arguments", {}), schema)
                if not valid:
                    self.issues.append(QCIssue(
                        sample_id=sample_id,
                        issue_type="schema_error",
                        message=f"Schema validation failed: {error}",
                        severity="error"
                    ))
                    return None

        # Check for duplicates
        content_hash = self._compute_hash(sample)
        if content_hash in self.seen_hashes:
            self.issues.append(QCIssue(
                sample_id=sample_id,
                issue_type="duplicate",
                message="Duplicate sample detected",
                severity="warning"
            ))
            if self.config.dedup_enabled:
                return None

        self.seen_hashes.add(content_hash)

        return ValidatedSample(
            id=sample_id,
            source=sample.get("source", "unknown"),
            scenario=scenario,
            tool_name=tool_name,
            messages=messages,
            content_hash=content_hash
        )

    def _extract_tool_call(self, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Extract tool call from messages."""
        import re

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")

            # Try to find JSON block - support multiple formats
            patterns = [
                # Hermes format with <tool_call> tags
                r"<tool_call>\s*(.*?)\s*</tool_call>",
                # Standard markdown code blocks
                r"```json\s*(.*?)\s*```",
                r"```\s*(.*?)\s*```",
                # Bare JSON object with "name" field
                r"\{[^{}]*\"name\"[^{}]*\}",
            ]

            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    # For patterns with groups, use group 1; for bare JSON, use group 0
                    json_str = match.group(1) if match.lastindex else match.group(0)
                    try:
                        data = json.loads(json_str)
                        if "name" in data:
                            return {
                                "name": data["name"],
                                "arguments": data.get("arguments", data.get("parameters", {}))
                            }
                    except json.JSONDecodeError:
                        continue

        return None

    def _validate_schema(self, arguments: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate arguments against JSON Schema."""
        try:
            jsonschema.validate(arguments, schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e.message)
        except jsonschema.SchemaError as e:
            return False, f"Invalid schema: {e.message}"

    def _compute_hash(self, sample: dict[str, Any]) -> str:
        """Compute content hash for deduplication."""
        # Hash based on user message + tool call + final response
        messages = sample.get("messages", [])

        key_parts = []
        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                key_parts.append(msg.get("content", ""))

        content = "|||".join(key_parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _check_coverage(
        self,
        tool_coverage: dict[str, int],
        scenario_coverage: dict[str, int],
        total: int
    ) -> None:
        """Check coverage requirements and add warnings."""
        # Check tool coverage
        for tool in self.tool_names:
            count = tool_coverage.get(tool, 0)
            if count < self.config.min_samples_per_tool:
                self.issues.append(QCIssue(
                    sample_id="coverage",
                    issue_type="low_coverage",
                    message=f"Tool '{tool}' has only {count} samples (min: {self.config.min_samples_per_tool})",
                    severity="warning"
                ))

        # Check scenario coverage
        if self.config.require_scenario_coverage:
            for scenario, target in self.config.scenario_targets.items():
                actual = scenario_coverage.get(scenario, 0)
                expected = int(total * target * 0.5)  # Allow 50% variance

                if actual < expected:
                    self.issues.append(QCIssue(
                        sample_id="coverage",
                        issue_type="low_scenario_coverage",
                        message=f"Scenario '{scenario}' has {actual} samples (expected ~{expected})",
                        severity="warning"
                    ))

    def print_report(self, report: QCReport) -> None:
        """Pretty-print QC report to console."""
        console.print("\n[bold]Dataset Quality Report[/bold]")
        console.print("=" * 50)

        # Summary
        console.print(f"\nSamples analyzed: {report.total_samples}")
        console.print(f"Valid samples: {report.valid_samples} ({report.valid_samples/report.total_samples:.1%})")
        console.print(f"Dropped: {report.dropped_samples}")

        # Issue breakdown
        if report.issues:
            console.print("\n[bold]Issues by type:[/bold]")
            issue_counts: dict[str, int] = defaultdict(int)
            for issue in report.issues:
                issue_counts[issue["issue_type"]] += 1
            for issue_type, count in sorted(issue_counts.items()):
                console.print(f"  - {issue_type}: {count}")

        # Schema pass rate
        status = "[green]PASS[/green]" if report.schema_pass_rate >= self.config.schema_pass_threshold else "[red]FAIL[/red]"
        console.print(f"\nSchema Pass Rate: {report.schema_pass_rate:.1%} {status}")
        console.print(f"Dedup Rate: {report.dedup_rate:.1%}")

        # Tool coverage table
        console.print("\n[bold]Tool Coverage:[/bold]")
        tool_table = Table(show_header=True)
        tool_table.add_column("Tool")
        tool_table.add_column("Samples", justify="right")
        tool_table.add_column("Status")

        for tool in sorted(self.tool_names):
            count = report.tool_coverage.get(tool, 0)
            status = "[green]PASS[/green]" if count >= self.config.min_samples_per_tool else "[red]FAIL[/red]"
            tool_table.add_row(tool, str(count), status)

        console.print(tool_table)

        # Scenario coverage table
        console.print("\n[bold]Scenario Coverage:[/bold]")
        scenario_table = Table(show_header=True)
        scenario_table.add_column("Scenario")
        scenario_table.add_column("Samples", justify="right")
        scenario_table.add_column("Percentage", justify="right")

        for scenario in sorted(report.scenario_coverage.keys()):
            count = report.scenario_coverage[scenario]
            pct = count / report.total_samples if report.total_samples > 0 else 0
            scenario_table.add_row(scenario, str(count), f"{pct:.0%}")

        console.print(scenario_table)

        # Final verdict
        if report.passes_threshold(self.config.schema_pass_threshold, self.config.min_samples_per_tool):
            console.print("\n[green]Dataset passes quality gate[/green]")
        else:
            console.print("\n[red]Dataset fails quality gate[/red]")

        # Repair stats if any repairs were attempted
        if self.repair_stats.repairs_attempted > 0:
            console.print("\n[bold]Repair Statistics:[/bold]")
            console.print(f"  Attempted: {self.repair_stats.repairs_attempted}")
            console.print(f"  Successful: {self.repair_stats.repairs_successful}")
            if self.repair_stats.repairs_attempted > 0:
                success_rate = self.repair_stats.repairs_successful / self.repair_stats.repairs_attempted
                console.print(f"  Success rate: {success_rate:.1%}")

    # =========================================================================
    # Repair Handlers
    # =========================================================================

    def _repair_truncated_response(
        self,
        sample: dict[str, Any],
        sample_id: str,
    ) -> bool:
        """Truncate overly long assistant responses.

        Returns True if repair was successful.
        """
        messages = sample.get("messages", [])
        repaired = False

        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if len(content) > self.config.max_response_length:
                    # Find a safe truncation point (end of sentence or paragraph)
                    truncate_at = self.config.max_response_length
                    for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                        last_sep = content[:truncate_at].rfind(sep)
                        if last_sep > truncate_at // 2:
                            truncate_at = last_sep + len(sep)
                            break

                    msg["content"] = content[:truncate_at].rstrip()
                    self.repair_stats.record(
                        sample_id,
                        "truncated_response",
                        True,
                        f"Truncated from {len(content)} to {truncate_at} chars",
                    )
                    repaired = True

        return repaired

    def _repair_whitespace(
        self,
        sample: dict[str, Any],
        sample_id: str,
    ) -> bool:
        """Normalize whitespace in messages.

        - Removes leading/trailing whitespace
        - Normalizes multiple spaces to single space
        - Normalizes line endings

        Returns True if repair was successful.
        """
        messages = sample.get("messages", [])
        repaired = False

        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue

            original = content

            # Normalize line endings
            content = content.replace("\r\n", "\n").replace("\r", "\n")

            # Remove trailing whitespace from each line
            lines = [line.rstrip() for line in content.split("\n")]
            content = "\n".join(lines)

            # Strip leading/trailing whitespace
            content = content.strip()

            if content != original:
                msg["content"] = content
                repaired = True

        if repaired:
            self.repair_stats.record(
                sample_id,
                "whitespace",
                True,
                "Normalized whitespace",
            )

        return repaired

    def _repair_encoding(
        self,
        sample: dict[str, Any],
        sample_id: str,
    ) -> bool:
        """Fix common encoding issues in messages.

        - Replaces common mojibake patterns
        - Removes null bytes
        - Normalizes unicode

        Returns True if repair was successful.
        """
        import unicodedata

        messages = sample.get("messages", [])
        repaired = False

        # Common problematic character replacements
        replacements = {
            "\x00": "",  # Null bytes
            "\ufffd": "",  # Replacement character (unknown encoding)
        }

        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue

            original = content

            # Apply replacements
            for old, new in replacements.items():
                content = content.replace(old, new)

            # Normalize unicode to NFC form
            content = unicodedata.normalize("NFC", content)

            if content != original:
                msg["content"] = content
                repaired = True

        if repaired:
            self.repair_stats.record(
                sample_id,
                "encoding",
                True,
                "Fixed encoding issues",
            )

        return repaired

    def repair_sample(
        self,
        sample: dict[str, Any],
        sample_id: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Apply all applicable repairs to a sample.

        Args:
            sample: The sample to repair
            sample_id: Sample identifier for logging

        Returns:
            Tuple of (was_repaired, repaired_sample)
        """
        # Make a copy to avoid modifying original
        import copy
        repaired_sample = copy.deepcopy(sample)

        any_repaired = False

        # Apply repairs in order of priority
        if self._repair_encoding(repaired_sample, sample_id):
            any_repaired = True

        if self._repair_whitespace(repaired_sample, sample_id):
            any_repaired = True

        if self._repair_truncated_response(repaired_sample, sample_id):
            any_repaired = True

        return any_repaired, repaired_sample
