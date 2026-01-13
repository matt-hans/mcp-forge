"""MCP-Forge CLI entry point.

v1.1: Full command suite including qa, benchmark, pack, and verify-bundle.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console

from mcp_forge import __version__
from mcp_forge.state import (
    PipelineStage,
    StateManager,
    SynthesisPlan,
)

console = Console()


def print_banner():
    """Print MCP-Forge banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ███╗   ███╗ ██████╗██████╗       ███████╗ ██████╗       ║
    ║   ████╗ ████║██╔════╝██╔══██╗      ██╔════╝██╔═══██╗      ║
    ║   ██╔████╔██║██║     ██████╔╝█████╗█████╗  ██║   ██║      ║
    ║   ██║╚██╔╝██║██║     ██╔═══╝ ╚════╝██╔══╝  ██║   ██║      ║
    ║   ██║ ╚═╝ ██║╚██████╗██║           ██║     ╚██████╔╝      ║
    ║   ╚═╝     ╚═╝ ╚═════╝╚═╝           ╚═╝      ╚═════╝       ║
    ║                                                           ║
    ║   Fine-tune LLMs for MCP tool use                         ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


@click.group()
@click.version_option(version=__version__, prog_name="mcp-forge")
@click.pass_context
def cli(ctx):
    """MCP-Forge: Fine-tune local LLMs on MCP server tool schemas."""
    ctx.ensure_object(dict)
    ctx.obj["state_manager"] = StateManager()


# =============================================================================
# Main Commands
# =============================================================================

@cli.command()
@click.option("--server", "-s", help="MCP server command (e.g., 'npx -y @mcp/server-weather')")
@click.option("--tools-file", "-t", type=click.Path(exists=True), help="Import tools from file instead of MCP")
@click.option("--model", "-m", type=click.Choice(["deepseek-r1", "qwen-2.5"]), default="deepseek-r1", help="Model family")
@click.option("--profile", "-p", type=click.Choice(["fast_dev", "balanced", "max_quality"]), default="balanced", help="Training profile")
@click.option("--samples", default=500, help="Total samples to generate")
@click.option("--no-tool-ratio", default=0.15, help="Ratio of no-tool samples")
@click.option("--error-ratio", default=0.10, help="Ratio of error-case samples")
@click.option("--format", "quant_format", type=click.Choice(["q8_0", "q4_k_m"]), default="q8_0", help="GGUF quantization")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory for bundle")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint")
@click.option("--skip-benchmark", is_flag=True, help="Skip benchmark step")
@click.pass_context
def run(
    ctx,
    server: str | None,
    tools_file: str | None,
    model: str,
    profile: str,
    samples: int,
    no_tool_ratio: float,
    error_ratio: float,
    quant_format: str,
    output: str,
    resume: bool,
    skip_benchmark: bool
):
    """Run the full pipeline: inspect -> generate -> train -> validate -> export -> pack."""
    print_banner()

    state_manager: StateManager = ctx.obj["state_manager"]

    if not server and not tools_file:
        console.print("[red]Error: Either --server or --tools-file is required[/red]")
        raise SystemExit(1)

    if resume:
        state = state_manager.load_state()
        if state is None:
            console.print("[yellow]No previous session found. Starting fresh.[/yellow]")
            resume = False
        else:
            console.print(f"[green]Resuming from stage: {state.stage.value}[/green]")

    if not resume:
        # Create synthesis plan
        scenario_weights = {
            "standard": 1.0 - no_tool_ratio - error_ratio - 0.15,  # Remaining after others
            "no_tool": no_tool_ratio,
            "error": error_ratio,
            "ambiguous": 0.10,
            "edge": 0.05
        }

        synthesis_plan = SynthesisPlan(
            total_samples=samples,
            seed_samples=min(100, samples // 5),
            augmented_samples=samples - min(100, samples // 5),
            scenario_weights=scenario_weights
        )

        # Create new session
        state = state_manager.create_session(
            mcp_command=server or "",
            system_prompt="You are a helpful assistant with access to specific tools.",
            model_family=model,
            output_path=output,
            quantization=quant_format,
            profile=profile
        )
        state.synthesis_plan = synthesis_plan
        state_manager.save_state(state)

    # Run pipeline stages
    try:
        _run_pipeline(state, state_manager, tools_file, skip_benchmark)
    except Exception as e:
        state.set_error(str(e))
        state_manager.save_state(state)
        console.print(f"[red]Pipeline failed: {e}[/red]")
        raise SystemExit(1) from e


def _run_pipeline(state, state_manager, tools_file: str | None, skip_benchmark: bool):
    """Execute pipeline stages."""
    from mcp_forge.tools.inspector import inspect_mcp_server

    # Stage 1: Inspect tools
    if state.stage in (PipelineStage.IDLE, PipelineStage.INSPECTING):
        console.print("\n[bold]Stage 1: Inspecting tools...[/bold]")
        state.update_stage(PipelineStage.INSPECTING)
        state_manager.save_state(state)

        if tools_file:
            # TODO: Implement file-based tool loading
            console.print(f"Loading tools from {tools_file}")
            # tools = load_tools_from_file(tools_file)
            raise NotImplementedError("File-based tool loading coming in v1.2")
        else:
            tools = asyncio.run(inspect_mcp_server(state.mcp_command))

        state.tools = tools
        state.toolset_hash = state.compute_toolset_hash()
        console.print(f"   Found {len(tools)} tools")
        for tool in tools:
            console.print(f"   - {tool.name}: {tool.description[:50]}...")

    # Stage 2: Generate data
    if state.stage in (PipelineStage.INSPECTING, PipelineStage.SYNTHESIZING):
        console.print("\n[bold]Stage 2: Generating training data...[/bold]")
        state.update_stage(PipelineStage.SYNTHESIZING)
        state_manager.save_state(state)

        from mcp_forge.data import DataSynthesizer

        synthesizer = DataSynthesizer(
            tools=state.tools,
            plan=state.synthesis_plan,
            output_dir=state_manager.data_dir,
        )

        with console.status("[bold green]Generating training data..."):
            result = asyncio.run(synthesizer.synthesize(
                progress_callback=lambda msg: console.print(f"   {msg}")
            ))

        state.seed_data_path = str(result.seed_path)
        state.training_data_path = str(result.training_path)

        console.print(f"   [green]✓[/green] Generated {result.seed_count} seed samples")
        console.print(f"   [green]✓[/green] Augmented to {result.total_count} total samples")
        console.print(f"   [green]✓[/green] QC: {'PASSED' if result.qc_passed else 'FAILED'}")

    # Stage 3: QC Validation
    if state.stage in (PipelineStage.SYNTHESIZING, PipelineStage.QC_VALIDATING):
        console.print("\n[bold]Stage 3: Validating data quality...[/bold]")
        state.update_stage(PipelineStage.QC_VALIDATING)
        state_manager.save_state(state)

        from mcp_forge.config import load_config
        from mcp_forge.data.qc import DataQualityController, QCConfig, QCFailedError

        # Load QC thresholds from config
        forge_config = load_config()

        qc_config = QCConfig(
            schema_pass_threshold=forge_config.qc_schema_pass_threshold,
            min_samples_per_tool=forge_config.qc_min_samples_per_tool,
            dedup_enabled=forge_config.qc_dedup_enabled,
            auto_repair=False,  # Don't auto-repair in pipeline, explicit --fix needed
            require_scenario_coverage=forge_config.qc_require_scenario_coverage,
        )

        qc = DataQualityController(state.tools, qc_config)

        # Get training data path
        training_data_path = Path(state.training_data_path) if state.training_data_path else state_manager.get_data_path("train.jsonl")

        console.print(f"   Validating: {training_data_path}")
        report, validated = qc.validate_dataset(training_data_path)

        # Store report in state
        state.qc_report = report
        state_manager.save_state(state)

        # Save report to disk
        report_path = state_manager.save_qc_report(report)
        console.print(f"   Report saved: {report_path}")

        # Print summary
        console.print(f"   Samples: {report.total_samples} total, {report.valid_samples} valid")
        console.print(f"   Schema pass rate: {report.schema_pass_rate:.1%}")
        console.print(f"   Dedup rate: {report.dedup_rate:.1%}")

        # Check if passes threshold
        if not report.passes_threshold(qc_config.schema_pass_threshold, qc_config.min_samples_per_tool):
            console.print("\n[red]QC validation failed![/red]")
            qc.print_report(report)

            # Raise blocking error
            raise QCFailedError(
                report=report,
                threshold=qc_config.schema_pass_threshold,
                min_samples=qc_config.min_samples_per_tool,
            )

        console.print("   [green]✓[/green] QC validation passed")

    # Stage 4: Training
    if state.stage in (PipelineStage.QC_VALIDATING, PipelineStage.TRAINING):
        console.print("\n[bold]Stage 4: Training model...[/bold]")
        state.update_stage(PipelineStage.TRAINING)
        state_manager.save_state(state)

        console.print(f"   Model: {state.model_family}")
        console.print(f"   Profile: {state.profile}")
        # TODO: Implement training
        console.print("   [yellow]Training not yet implemented[/yellow]")

    # Stage 5: Validation
    if state.stage in (PipelineStage.TRAINING, PipelineStage.VALIDATING):
        console.print("\n[bold]Stage 5: Validating model...[/bold]")
        state.update_stage(PipelineStage.VALIDATING)
        state_manager.save_state(state)

        # TODO: Implement looped validation
        console.print("   [yellow]Looped validation not yet implemented[/yellow]")

    # Stage 6: Benchmark (optional)
    if not skip_benchmark and state.stage in (PipelineStage.VALIDATING, PipelineStage.BENCHMARKING):
        console.print("\n[bold]Stage 6: Running benchmarks...[/bold]")
        state.update_stage(PipelineStage.BENCHMARKING)
        state_manager.save_state(state)

        # TODO: Implement benchmarking
        console.print("   [yellow]Benchmarking not yet implemented[/yellow]")

    # Stage 7: Export
    if state.stage in (PipelineStage.VALIDATING, PipelineStage.BENCHMARKING, PipelineStage.EXPORTING):
        console.print("\n[bold]Stage 7: Exporting GGUF...[/bold]")
        state.update_stage(PipelineStage.EXPORTING)
        state_manager.save_state(state)

        console.print(f"   Format: {state.quantization}")
        # TODO: Implement export
        console.print("   [yellow]Export not yet implemented[/yellow]")

    # Stage 8: Package
    if state.stage in (PipelineStage.EXPORTING, PipelineStage.PACKAGING):
        console.print("\n[bold]Stage 8: Creating agent bundle...[/bold]")
        state.update_stage(PipelineStage.PACKAGING)
        state_manager.save_state(state)

        # TODO: Implement packaging
        console.print("   [yellow]Packaging not yet implemented[/yellow]")

    # Complete
    state.update_stage(PipelineStage.COMPLETE)
    state_manager.save_state(state)

    console.print("\n" + "=" * 50)
    console.print("[bold green]Pipeline complete![/bold green]")
    console.print(f"\nBundle location: {state.output_path}")


@cli.command()
def doctor():
    """Check environment for CUDA, VRAM, and dependencies."""
    console.print("\n[bold]MCP-Forge Environment Check[/bold]")
    console.print("=" * 50)

    import os
    import platform
    import shutil

    # Python version
    py_version = platform.python_version()
    if tuple(map(int, py_version.split(".")[:2])) >= (3, 10):
        console.print(f"[green]PASS[/green] Python {py_version}")
    else:
        console.print(f"[red]FAIL[/red] Python {py_version} (need 3.10+)")

    # CUDA
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            console.print(f"[green]PASS[/green] CUDA {cuda_version} available")

            # GPU info
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                console.print(f"[green]PASS[/green] GPU {i}: {name} ({memory:.1f}GB)")
        else:
            console.print("[yellow]WARN[/yellow] CUDA not available")
    except ImportError:
        console.print("[red]FAIL[/red] PyTorch not installed")

    # Key dependencies
    deps = [
        ("unsloth", "unsloth"),
        ("transformers", "transformers"),
        ("mcp", "mcp"),
        ("openai", "openai"),
        ("click", "click"),
        ("rich", "rich"),
    ]

    for name, module in deps:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "installed")
            console.print(f"[green]PASS[/green] {name} {version}")
        except ImportError:
            console.print(f"[red]FAIL[/red] {name} not installed")

    # OpenAI API key
    if os.environ.get("OPENAI_API_KEY"):
        console.print("[green]PASS[/green] OPENAI_API_KEY set")
    else:
        console.print("[yellow]WARN[/yellow] OPENAI_API_KEY not set (required for seed generation)")

    # Disk space
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    if free_gb > 50:
        console.print(f"[green]PASS[/green] Disk space: {free_gb:.1f}GB free")
    else:
        console.print(f"[yellow]WARN[/yellow] Disk space: {free_gb:.1f}GB free (recommend 50GB+)")

    console.print("\n" + "=" * 50)


@cli.command()
@click.pass_context
def status(ctx):
    """Show current pipeline state."""
    state_manager: StateManager = ctx.obj["state_manager"]
    state = state_manager.load_state()

    if state is None:
        console.print("[yellow]No active session found.[/yellow]")
        return

    console.print("\n[bold]Pipeline Status[/bold]")
    console.print("=" * 50)
    console.print(f"Session ID: {state.session_id}")
    console.print(f"Stage: {state.stage.value}")
    console.print(f"Model: {state.model_family}")
    console.print(f"Profile: {state.profile}")
    console.print(f"Output: {state.output_path}")
    console.print(f"Created: {state.created_at}")
    console.print(f"Updated: {state.updated_at}")

    if state.error:
        console.print(f"\n[red]Error: {state.error}[/red]")

    if state.tools:
        console.print(f"\nTools: {len(state.tools)}")
        for tool in state.tools:
            console.print(f"  - {tool.name}")

    if state.training_progress > 0:
        console.print(f"\nTraining progress: {state.training_progress:.1%}")
        if state.training_loss:
            console.print(f"Current loss: {state.training_loss:.4f}")


# =============================================================================
# Tools Commands
# =============================================================================

@cli.group()
def tools():
    """Tool management commands."""
    pass


@tools.command("inspect")
@click.option("--server", "-s", required=True, help="MCP server command")
@click.option("--output", "-o", default="tools.json", help="Output file")
@click.option("--timeout", default=30, help="Connection timeout in seconds")
def tools_inspect(server: str, output: str, timeout: int):
    """Extract tools from an MCP server."""
    import json

    from mcp_forge.tools.inspector import format_tool_for_display, inspect_mcp_server

    console.print("\n[bold]Inspecting MCP server...[/bold]")
    console.print(f"Command: {server}")

    try:
        tool_list = asyncio.run(inspect_mcp_server(server, timeout))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1) from e

    console.print(f"\n[green]Found {len(tool_list)} tools:[/green]\n")
    for tool in tool_list:
        console.print(format_tool_for_display(tool))
        console.print()

    # Save to file
    with open(output, "w") as f:
        json.dump([t.to_dict() for t in tool_list], f, indent=2)

    console.print(f"[green]Saved to {output}[/green]")


@tools.command("import")
@click.option("--from", "from_file", required=True, type=click.Path(exists=True), help="Source file")
@click.option("--output", "-o", default="tools.json", help="Output file")
@click.option("--format", "fmt", type=click.Choice(["auto", "openai", "yaml"]), default="auto", help="Input format")
def tools_import(from_file: str, output: str, fmt: str):
    """Import tools from a file (OpenAI format, YAML)."""
    console.print("[yellow]Tool import coming in v1.2[/yellow]")


# =============================================================================
# Data Commands
# =============================================================================


def _save_qc_report(
    report: QCReport,  # noqa: F821
    path: Path,
    format_type: str,
    repair_stats: RepairStats | None = None,  # noqa: F821
    config: QCConfig | None = None,  # noqa: F821
) -> None:
    """Save QC report in the specified format.

    Args:
        report: The QC report to save
        path: Output file path
        format_type: One of 'json', 'markdown', or 'csv'
        repair_stats: Optional repair statistics to include
        config: Optional config for threshold info in report
    """
    import csv
    import json
    from datetime import datetime

    if format_type == "json":
        data = report.to_dict()
        if repair_stats:
            data["repair_stats"] = repair_stats.to_dict()
        if config:
            data["thresholds"] = {
                "schema_pass_threshold": config.schema_pass_threshold,
                "min_samples_per_tool": config.min_samples_per_tool,
            }
        data["generated_at"] = datetime.utcnow().isoformat()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    elif format_type == "markdown":
        lines = [
            "# QC Report",
            "",
            f"**Generated:** {datetime.utcnow().isoformat()}",
            "",
            "## Summary",
            "",
            f"- Total samples: {report.total_samples}",
            f"- Valid samples: {report.valid_samples} ({report.valid_samples/report.total_samples:.1%})",
            f"- Dropped: {report.dropped_samples}",
            f"- Schema pass rate: {report.schema_pass_rate:.1%}",
            f"- Dedup rate: {report.dedup_rate:.1%}",
            "",
            "## Tool Coverage",
            "",
            "| Tool | Samples |",
            "|------|---------|",
        ]
        for tool, count in sorted(report.tool_coverage.items()):
            lines.append(f"| {tool} | {count} |")

        lines.extend([
            "",
            "## Scenario Coverage",
            "",
            "| Scenario | Samples | Percentage |",
            "|----------|---------|------------|",
        ])
        for scenario, count in sorted(report.scenario_coverage.items()):
            pct = count / report.total_samples if report.total_samples > 0 else 0
            lines.append(f"| {scenario} | {count} | {pct:.0%} |")

        if report.issues:
            lines.extend([
                "",
                "## Issues",
                "",
                f"Found {len(report.issues)} issue(s):",
                "",
            ])
            for issue in report.issues[:20]:  # Limit to 20 in markdown
                lines.append(f"- [{issue['severity']}] {issue['issue_type']}: {issue['message']}")
            if len(report.issues) > 20:
                lines.append(f"- ... and {len(report.issues) - 20} more issues")

        if repair_stats and repair_stats.repairs_attempted > 0:
            lines.extend([
                "",
                "## Repair Statistics",
                "",
                f"- Attempted: {repair_stats.repairs_attempted}",
                f"- Successful: {repair_stats.repairs_successful}",
            ])

        with open(path, "w") as f:
            f.write("\n".join(lines))

    elif format_type == "csv":
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "issue_type", "severity", "message", "repairable"])
            for issue in report.issues:
                writer.writerow([
                    issue.get("sample_id", ""),
                    issue.get("issue_type", ""),
                    issue.get("severity", ""),
                    issue.get("message", ""),
                    issue.get("repairable", False),
                ])


@cli.command()
@click.option("--data", "-d", required=True, type=click.Path(exists=True), help="Training data JSONL file")
@click.option("--tools", "-t", required=True, type=click.Path(exists=True), help="Tools JSON file")
@click.option("--fix", is_flag=True, help="Auto-repair and rewrite cleaned data")
@click.option("--output", "-o", help="Output file for cleaned data")
@click.option("--threshold", "-T", type=float, help="Schema pass rate threshold (default: 0.98)")
@click.option("--min-samples", type=int, help="Minimum samples per tool (default: 10)")
@click.option("--no-dedup", is_flag=True, help="Disable duplicate detection")
@click.option("--no-auto-repair", is_flag=True, help="Disable auto-repair of minor issues")
@click.option("--strict", is_flag=True, help="Fail on warnings, not just errors")
@click.option("--report", "-r", help="Output path for QC report")
@click.option("--format", "report_format", type=click.Choice(["json", "markdown", "csv"]), default="json", help="Report format")
@click.option("--dry-run", is_flag=True, help="Preview repairs without writing files")
@click.pass_context
def qa(
    ctx,
    data: str,
    tools: str,
    fix: bool,
    output: str | None,
    threshold: float | None,
    min_samples: int | None,
    no_dedup: bool,
    no_auto_repair: bool,
    strict: bool,
    report: str | None,
    report_format: str,
    dry_run: bool,
):
    """Run dataset quality analysis and optional cleanup.

    Validates training data against tool schemas and reports issues.
    Use --fix to auto-repair and output cleaned data.

    \b
    Threshold Precedence:
      1. CLI flags (--threshold, --min-samples)
      2. Config file (.mcp-forge/config.yaml)
      3. Default values

    \b
    Report Formats:
      json     - Full machine-readable report (default)
      markdown - Human-readable summary
      csv      - Issues as CSV for spreadsheet import
    """
    import json

    from mcp_forge.config import load_config, merge_config_with_cli
    from mcp_forge.data.qc import DataQualityController, QCConfig
    from mcp_forge.state import ToolDefinition

    console.print("\n[bold]Running Dataset QA...[/bold]")

    # Load tools
    with open(tools) as f:
        tool_defs = [ToolDefinition.from_dict(t) for t in json.load(f)]

    # Load config and merge with CLI options
    forge_config = load_config()
    forge_config = merge_config_with_cli(
        forge_config,
        threshold=threshold,
        min_samples=min_samples,
        no_dedup=no_dedup,
        no_auto_repair=no_auto_repair,
        strict=strict,
    )

    # Configure QC from merged config
    config = QCConfig(
        schema_pass_threshold=forge_config.qc_schema_pass_threshold,
        min_samples_per_tool=forge_config.qc_min_samples_per_tool,
        dedup_enabled=forge_config.qc_dedup_enabled,
        auto_repair=fix and forge_config.qc_auto_repair,  # Only repair if --fix is set
        require_scenario_coverage=forge_config.qc_require_scenario_coverage,
        scenario_targets=forge_config.qc_scenario_targets,
    )
    qc = DataQualityController(tool_defs, config)

    # Determine output path
    output_path = Path(output) if output else (Path(data).with_suffix(".clean.jsonl") if fix else None)

    # Run validation
    qc_report, validated = qc.validate_dataset(Path(data), output_path, dry_run=dry_run)

    # Print report
    qc.print_report(qc_report)

    if output_path and not dry_run:
        console.print(f"\n[green]Cleaned data saved to {output_path}[/green]")
    elif output_path and dry_run:
        console.print(f"\n[yellow]Dry run: would save cleaned data to {output_path}[/yellow]")

    # Save report in requested format
    state_manager: StateManager = ctx.obj["state_manager"]

    if report:
        # User specified custom report path
        report_path = Path(report)
        _save_qc_report(qc_report, report_path, report_format, qc.repair_stats, config)
        console.print(f"[green]Report saved to {report_path}[/green]")
    else:
        # Use default state manager path
        default_report_path = state_manager.save_qc_report(qc_report)
        console.print(f"[green]Report saved to {default_report_path}[/green]")

    # In strict mode, fail on any warnings
    has_warnings = any(issue.get("severity") == "warning" for issue in qc_report.issues)
    passes = qc_report.passes_threshold(config.schema_pass_threshold, config.min_samples_per_tool)

    if not passes or (strict and has_warnings):
        console.print("\n[red]QC validation failed![/red]")
        if not passes:
            console.print(f"  Schema pass rate: {qc_report.schema_pass_rate:.1%} (threshold: {config.schema_pass_threshold:.1%})")
        if strict and has_warnings:
            console.print(f"  Strict mode: {sum(1 for i in qc_report.issues if i.get('severity') == 'warning')} warnings found")
        console.print("\n[yellow]Suggestions:[/yellow]")
        console.print("  - Run with --fix to auto-repair fixable issues")
        console.print("  - Lower threshold with --threshold 0.95")
        console.print("  - Review issues in the report above")
        raise SystemExit(1)


@cli.command()
@click.option("--server", "-s", help="MCP server command to extract tools from")
@click.option("--tools", "-t", type=click.Path(exists=True), help="Tools JSON file (alternative to --server)")
@click.option("--samples", default=500, help="Total samples to generate")
@click.option("--seed-samples", default=100, help="Number of seed samples to generate")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output JSONL file")
def generate(server: str | None, tools: str | None, samples: int, seed_samples: int, output: str):
    """Generate training data (seed + augmentation)."""
    import json

    from mcp_forge.data import DataSynthesizer
    from mcp_forge.state import SynthesisPlan, ToolDefinition
    from mcp_forge.tools.inspector import inspect_mcp_server

    if not server and not tools:
        console.print("[red]Error: Either --server or --tools is required[/red]")
        raise SystemExit(1)

    console.print("\n[bold]MCP-Forge Data Generation[/bold]")
    console.print("=" * 50)

    # Load tools
    if tools:
        console.print(f"Loading tools from {tools}...")
        with open(tools) as f:
            tool_list = [ToolDefinition.from_dict(t) for t in json.load(f)]
    else:
        console.print(f"Inspecting MCP server: {server}")
        tool_list = asyncio.run(inspect_mcp_server(server))

    console.print(f"Found {len(tool_list)} tools")

    # Create synthesis plan
    plan = SynthesisPlan(
        total_samples=samples,
        seed_samples=min(seed_samples, samples // 2),
        augmented_samples=samples - min(seed_samples, samples // 2),
    )

    # Create output directory
    output_path = Path(output)
    output_dir = output_path.parent if output_path.suffix else output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run synthesis
    synthesizer = DataSynthesizer(
        tools=tool_list,
        plan=plan,
        output_dir=output_dir,
    )

    console.print(f"\nGenerating {samples} samples ({plan.seed_samples} seeds)...\n")

    result = asyncio.run(synthesizer.synthesize(
        progress_callback=lambda msg: console.print(f"  {msg}")
    ))

    # Rename output if specific file requested
    if output_path.suffix:
        import shutil
        shutil.move(result.training_path, output_path)
        console.print(f"\n[green]Output saved to {output_path}[/green]")
    else:
        console.print(f"\n[green]Output saved to {result.training_path}[/green]")

    console.print("\nResults:")
    console.print(f"  Seeds: {result.seed_count}")
    console.print(f"  Augmented: {result.augmented_count}")
    console.print(f"  Total: {result.total_count}")
    console.print(f"  QC: {'[green]PASSED[/green]' if result.qc_passed else '[red]FAILED[/red]'}")


# =============================================================================
# Training & Validation Commands
# =============================================================================

@cli.command()
@click.option("--data", "-d", required=True, type=click.Path(exists=True), help="Training data JSONL file")
@click.option("--model", "-m", type=click.Choice(["deepseek-r1", "qwen-2.5"]), required=True, help="Model family")
@click.option("--profile", "-p", type=click.Choice(["fast_dev", "balanced", "max_quality"]), default="balanced")
@click.option("--output", "-o", required=True, help="Output directory for LoRA adapter")
def train(data: str, model: str, profile: str, output: str):
    """Fine-tune a model on training data."""
    console.print("[yellow]Training coming soon[/yellow]")


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to LoRA adapter or GGUF")
@click.option("--server", "-s", help="MCP server command for live validation")
@click.option("--stub", type=click.Choice(["weather", "filesystem"]), help="Use deterministic stub")
@click.option("--samples", default=20, help="Number of validation samples")
def validate(model: str, server: str | None, stub: str | None, samples: int):
    """Run looped validation against real or stubbed MCP server."""
    if not server and not stub:
        console.print("[red]Error: Either --server or --stub is required[/red]")
        raise SystemExit(1)

    console.print("[yellow]Looped validation coming soon[/yellow]")


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to model")
@click.option("--tools", "-t", required=True, type=click.Path(exists=True), help="Tools JSON file")
@click.option("--baseline", help="Baseline model for comparison")
@click.option("--output", "-o", help="Output directory for reports")
def benchmark(model: str, tools: str, baseline: str | None, output: str | None):
    """Run full evaluation benchmark suite."""
    console.print("[yellow]Benchmarking coming soon[/yellow]")


# =============================================================================
# Export & Packaging Commands
# =============================================================================

@cli.command("export")
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to LoRA adapter")
@click.option("--format", "quant_format", type=click.Choice(["q8_0", "q4_k_m"]), default="q8_0")
@click.option("--output", "-o", required=True, help="Output GGUF file path")
def export_model(model: str, quant_format: str, output: str):
    """Export model to GGUF format."""
    console.print("[yellow]GGUF export coming soon[/yellow]")


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to GGUF model")
@click.option("--tools", "-t", required=True, type=click.Path(exists=True), help="Tools JSON file")
@click.option("--validation", type=click.Path(exists=True), help="Validation report JSON")
@click.option("--benchmark", "bench", type=click.Path(exists=True), help="Benchmark report JSON")
@click.option("--output", "-o", required=True, help="Output directory for bundle")
def pack(model: str, tools: str, validation: str | None, bench: str | None, output: str):
    """Create distributable agent bundle."""
    console.print("[yellow]Bundle packaging coming soon[/yellow]")


@cli.command("verify-bundle")
@click.argument("bundle_path", type=click.Path(exists=True))
def verify_bundle(bundle_path: str):
    """Verify an agent bundle and run smoke tests."""
    console.print("[yellow]Bundle verification coming soon[/yellow]")


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
