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

    if samples <= 0:
        console.print("[red]Error: --samples must be a positive integer[/red]")
        raise SystemExit(1)

    fixed_ratio = 0.15  # ambiguous (0.10) + edge (0.05)
    if not 0 <= no_tool_ratio <= 1 or not 0 <= error_ratio <= 1:
        console.print("[red]Error: --no-tool-ratio and --error-ratio must be between 0 and 1[/red]")
        raise SystemExit(1)
    if no_tool_ratio + error_ratio + fixed_ratio > 1.0:
        console.print(
            "[red]Error: --no-tool-ratio + --error-ratio is too high "
            f"(max {1.0 - fixed_ratio:.2f} to keep standard samples non-negative)[/red]"
        )
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
            "standard": 1.0 - no_tool_ratio - error_ratio - fixed_ratio,  # Remaining after others
            "no_tool": no_tool_ratio,
            "error": error_ratio,
            "ambiguous": 0.10,
            "edge": 0.05
        }

        seed_samples = max(1, min(100, samples // 5))
        synthesis_plan = SynthesisPlan(
            total_samples=samples,
            seed_samples=seed_samples,
            augmented_samples=samples - seed_samples,
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
            import json

            from mcp_forge.state import ToolDefinition

            console.print(f"Loading tools from {tools_file}")
            try:
                with open(tools_file) as f:
                    tools_data = json.load(f)
                tools = [ToolDefinition.from_dict(t) for t in tools_data]
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid tools JSON: {e}") from e
            except (KeyError, TypeError) as e:
                raise ValueError(f"Invalid tool definition format: {e}") from e
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

        from mcp_forge.training import TrainingConfig, TrainingEngine

        console.print(f"   Model: {state.model_family}")
        console.print(f"   Profile: {state.profile}")

        training_config = TrainingConfig(
            model_family=state.model_family,
            profile=state.profile,
            data_path=Path(state.training_data_path),
            output_dir=state_manager.state_dir / "adapters",
        )

        engine = TrainingEngine(training_config)

        def on_training_progress(step: int, total: int, progress: float, loss: float | None, epoch: float) -> None:
            state.training_progress = progress
            state.training_loss = loss
            state_manager.save_state(state)
            loss_str = f"{loss:.4f}" if loss else "N/A"
            console.print(f"   Step {step}/{total} | Loss: {loss_str}")

        with console.status("[bold green]Training..."):
            adapter_path = engine.train(progress_callback=on_training_progress)

        state.lora_adapter_path = str(adapter_path)
        console.print(f"   [green]✓[/green] Adapter saved to {adapter_path}")

    # Stage 5: Validation
    if state.stage in (PipelineStage.TRAINING, PipelineStage.VALIDATING):
        console.print("\n[bold]Stage 5: Validating model...[/bold]")
        state.update_stage(PipelineStage.VALIDATING)
        state_manager.save_state(state)

        from mcp_forge.validation import ValidationConfig, ValidationRunner
        from mcp_forge.validation.config import StubConfig
        from mcp_forge.validation.runner import generate_validation_samples

        # Default to weather stub for pipeline validation
        stub_config = StubConfig(stub_type="weather", deterministic=True)
        val_config = ValidationConfig(
            model_path=Path(state.lora_adapter_path),
            samples=20,
            stub_config=stub_config,
        )

        runner = ValidationRunner(val_config, state.tools)
        samples = generate_validation_samples(state.tools, count=20)

        def on_progress(current: int, total: int, pct: float) -> None:
            console.print(f"   {current}/{total} ({pct:.0%})")

        result = runner.run(samples, progress_callback=on_progress)

        state.validation_result = result
        state_manager.save_state(state)

        console.print(f"   Parse rate: {result.tool_call_parse_rate:.1%}")
        console.print(f"   Schema conformance: {result.schema_conformance_rate:.1%}")
        console.print(f"   Tool accuracy: {result.tool_selection_accuracy:.1%}")

        if result.passed:
            console.print("   [green]Validation passed[/green]")
        else:
            console.print("   [yellow]Validation below thresholds (continuing)[/yellow]")

    # Stage 6: Benchmark (optional)
    if not skip_benchmark and state.stage in (PipelineStage.VALIDATING, PipelineStage.BENCHMARKING):
        console.print("\n[bold]Stage 6: Running benchmarks...[/bold]")
        state.update_stage(PipelineStage.BENCHMARKING)
        state_manager.save_state(state)

        from mcp_forge.validation import BenchmarkConfig, BenchmarkRunner
        from mcp_forge.validation.config import StubConfig

        # Use weather stub for pipeline benchmarks
        stub_config = StubConfig(stub_type="weather", deterministic=True)
        bench_config = BenchmarkConfig(
            model_path=Path(state.lora_adapter_path),
            model_name=state.model_family,
            samples_per_tool=10,  # Reduced for pipeline speed
            samples_per_scenario=10,
            stub_config=stub_config,
        )

        runner = BenchmarkRunner(bench_config, state.tools)

        def on_benchmark_progress(category: str, current: int, total: int, pct: float) -> None:
            if current % 5 == 0 or current == total:
                console.print(f"   {category}: {pct:.0%}")

        result = runner.run(progress_callback=on_benchmark_progress)

        state.benchmark_result = result
        state_manager.save_state(state)

        # Save report
        report_path = state_manager.save_benchmark_result(result)

        console.print(f"   Overall score: {result.overall_score:.1%}")
        console.print(f"   Report: {report_path}")

        if result.overall_score >= 0.85:
            console.print("   [green]Benchmark passed[/green]")
        else:
            console.print("   [yellow]Benchmark below target (continuing)[/yellow]")

    # Stage 7: Export
    if state.stage in (PipelineStage.VALIDATING, PipelineStage.BENCHMARKING, PipelineStage.EXPORTING):
        console.print("\n[bold]Stage 7: Exporting GGUF...[/bold]")
        state.update_stage(PipelineStage.EXPORTING)
        state_manager.save_state(state)

        from mcp_forge.export import ExportConfig, ExportEngine, QuantizationType
        from mcp_forge.export.metadata import GGUFMetadata

        # Determine output path
        output_dir = Path(state.output_path)
        gguf_path = output_dir / f"{state.model_family}.{state.quantization}.gguf"

        console.print(f"   Format: {state.quantization}")
        console.print(f"   Output: {gguf_path}")

        config = ExportConfig(
            adapter_path=Path(state.lora_adapter_path),
            output_path=gguf_path,
            base_model=state.model_family,
            quantization=QuantizationType(state.quantization),
        )

        # Build metadata from pipeline state
        metadata = GGUFMetadata(
            model_name=f"mcp-forge-{state.model_family}",
            model_family=state.model_family,
            tool_names=[t.name for t in state.tools],
            training_timestamp=state.created_at,
        )
        if state.validation_result:
            metadata.tool_accuracy = state.validation_result.tool_selection_accuracy
            metadata.schema_conformance = state.validation_result.schema_conformance_rate
        if state.benchmark_result:
            metadata.benchmark_score = state.benchmark_result.overall_score

        engine = ExportEngine(config)

        def on_progress(stage: str, pct: float) -> None:
            console.print(f"   [{pct:.0%}] {stage}")

        result = engine.export(metadata=metadata, progress_callback=on_progress)

        if result.success:
            state.gguf_path = str(result.output_path)
            state_manager.save_state(state)

            console.print(f"   Size: {result.gguf_size_mb:.1f} MB")
            console.print(f"   Compression: {result.compression_ratio:.1f}x")
            console.print("   [green]Export complete[/green]")
        else:
            console.print(f"   [red]Export failed: {result.error}[/red]")
            state.error = result.error
            state.update_stage(PipelineStage.FAILED)
            state_manager.save_state(state)
            raise SystemExit(1)

    # Stage 8: Package
    if state.stage in (PipelineStage.EXPORTING, PipelineStage.PACKAGING):
        console.print("\n[bold]Stage 8: Creating agent bundle...[/bold]")
        state.update_stage(PipelineStage.PACKAGING)
        state_manager.save_state(state)

        from mcp_forge.export.bundle import BundleConfig, BundleEngine

        # Determine bundle output path
        output_dir = Path(state.output_path)

        console.print(f"   Output: {output_dir}")

        # Build config from pipeline state
        config = BundleConfig(
            gguf_path=Path(state.gguf_path),
            tools=state.tools,
            output_dir=output_dir,
            model_name=f"mcp-forge-{state.model_family}",
            model_family=state.model_family,
            training_samples=state.synthesis_plan.total_samples if state.synthesis_plan else 0,
        )

        # Add quality metrics if available
        if state.validation_result:
            config.tool_accuracy = state.validation_result.tool_selection_accuracy
            config.schema_conformance = state.validation_result.schema_conformance_rate
        if state.benchmark_result:
            config.benchmark_score = state.benchmark_result.overall_score

        engine = BundleEngine(config)

        def on_bundle_progress(msg: str) -> None:
            console.print(f"   {msg}")

        result = engine.package(progress_callback=on_bundle_progress)

        if result.success:
            state.bundle_path = str(result.bundle_path)
            state_manager.save_state(state)

            console.print(f"   Size: {result.bundle_size_mb:.1f} MB")
            console.print(f"   Files: {', '.join(result.files_created)}")
            console.print("   [green]Bundle complete[/green]")
        else:
            console.print(f"   [red]Packaging failed: {result.error}[/red]")
            state.error = result.error
            state.update_stage(PipelineStage.FAILED)
            state_manager.save_state(state)
            raise SystemExit(1)

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
    from importlib import metadata

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
            try:
                version = metadata.version(name)
            except metadata.PackageNotFoundError:
                version = getattr(mod, "__version__", "installed")
            console.print(f"[green]PASS[/green] {name} {version}")
        except ImportError:
            console.print(f"[red]FAIL[/red] {name} not installed")
        except Exception as e:
            console.print(f"[red]FAIL[/red] {name} failed to import ({e})")

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
    from datetime import datetime, timezone

    if format_type == "json":
        data = report.to_dict()
        if repair_stats:
            data["repair_stats"] = repair_stats.to_dict()
        if config:
            data["thresholds"] = {
                "schema_pass_threshold": config.schema_pass_threshold,
                "min_samples_per_tool": config.min_samples_per_tool,
            }
        data["generated_at"] = datetime.now(timezone.utc).isoformat()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    elif format_type == "markdown":
        valid_rate = report.valid_samples / report.total_samples if report.total_samples > 0 else 0
        lines = [
            "# QC Report",
            "",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Summary",
            "",
            f"- Total samples: {report.total_samples}",
            f"- Valid samples: {report.valid_samples} ({valid_rate:.1%})",
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
    if samples <= 0:
        console.print("[red]Error: --samples must be a positive integer[/red]")
        raise SystemExit(1)

    seed_target = max(1, min(seed_samples, max(1, samples // 2)))
    plan = SynthesisPlan(
        total_samples=samples,
        seed_samples=seed_target,
        augmented_samples=samples - seed_target,
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
    from mcp_forge.training import TrainingConfig, TrainingEngine

    console.print("\n[bold]MCP-Forge Training[/bold]")
    console.print("=" * 50)
    console.print(f"Data: {data}")
    console.print(f"Model: {model}")
    console.print(f"Profile: {profile}")
    console.print(f"Output: {output}")
    console.print()

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    training_config = TrainingConfig(
        model_family=model,
        profile=profile,
        data_path=Path(data),
        output_dir=output_path,
    )

    engine = TrainingEngine(training_config)

    def on_progress(step: int, total: int, progress: float, loss: float | None, epoch: float) -> None:
        loss_str = f"{loss:.4f}" if loss else "N/A"
        console.print(f"Step {step}/{total} | Loss: {loss_str} | Epoch: {epoch:.2f}")

    console.print("Loading model...")
    engine.load_model()
    console.print("[green]Model loaded[/green]\n")

    console.print("Starting training...")
    try:
        adapter_path = engine.train(progress_callback=on_progress)
        console.print("\n[green]Training complete![/green]")
        console.print(f"Adapter saved to: {adapter_path}")
    except Exception as e:
        console.print(f"\n[red]Training failed: {e}[/red]")
        raise SystemExit(1) from e


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to LoRA adapter or GGUF")
@click.option("--server", "-s", help="MCP server command for live validation")
@click.option("--stub", type=click.Choice(["weather", "filesystem"]), help="Use deterministic stub")
@click.option("--samples", default=20, help="Number of validation samples")
@click.option("--tools", "-t", type=click.Path(exists=True), help="Tools JSON file (required for stub)")
@click.option("--threshold", type=float, help="Override default pass threshold (0.90)")
@click.pass_context
def validate(
    ctx,
    model: str,
    server: str | None,
    stub: str | None,
    samples: int,
    tools: str | None,
    threshold: float | None,
):
    """Run looped validation against real or stubbed MCP server."""
    import json

    from mcp_forge.state import ToolDefinition
    from mcp_forge.validation import ValidationConfig, ValidationRunner
    from mcp_forge.validation.config import StubConfig
    from mcp_forge.validation.runner import generate_validation_samples

    if not server and not stub:
        console.print("[red]Error: Either --server or --stub is required[/red]")
        raise SystemExit(1)

    console.print("\n[bold]MCP-Forge Looped Validation[/bold]")
    console.print("=" * 50)

    # Load tools
    if stub:
        # Get tools from stub
        from mcp_forge.validation.stubs import StubRegistry

        stub_instance = StubRegistry.get(stub)
        tool_defs = [
            ToolDefinition(
                name=t["name"],
                description=t["description"],
                input_schema=t["inputSchema"],
            )
            for t in stub_instance.get_tools()
        ]
    elif tools:
        with open(tools) as f:
            tool_defs = [ToolDefinition.from_dict(t) for t in json.load(f)]
    else:
        # Extract from MCP server
        from mcp_forge.tools.inspector import inspect_mcp_server

        tool_defs = asyncio.run(inspect_mcp_server(server))

    console.print(f"Model: {model}")
    console.print(f"Mode: {'Stub (' + stub + ')' if stub else 'Live MCP'}")
    console.print(f"Tools: {len(tool_defs)}")
    console.print(f"Samples: {samples}")

    # Configure validation
    stub_config = StubConfig(stub_type=stub, deterministic=True) if stub else None
    config = ValidationConfig(
        model_path=Path(model),
        samples=samples,
        stub_config=stub_config,
        mcp_command=server,
        accuracy_threshold=threshold or 0.90,
    )

    runner = ValidationRunner(config, tool_defs)

    # Generate validation samples
    console.print("\nGenerating validation samples...")
    validation_samples = generate_validation_samples(tool_defs, count=samples)
    console.print(f"Generated {len(validation_samples)} samples")

    # Run validation
    console.print("\nRunning validation...")

    def on_progress(current: int, total: int, pct: float) -> None:
        console.print(f"  Sample {current}/{total} ({pct:.0%})")

    try:
        result = runner.run(validation_samples, progress_callback=on_progress)
    except Exception as e:
        console.print(f"\n[red]Validation failed: {e}[/red]")
        raise SystemExit(1) from e

    # Print results
    console.print("\n" + "=" * 50)
    console.print("[bold]Validation Results[/bold]")
    console.print(f"Passed: {'[green]YES[/green]' if result.passed else '[red]NO[/red]'}")
    console.print(f"Samples: {result.samples_passed}/{result.samples_tested}")
    console.print(f"Parse rate: {result.tool_call_parse_rate:.1%}")
    console.print(f"Schema conformance: {result.schema_conformance_rate:.1%}")
    console.print(f"Tool selection accuracy: {result.tool_selection_accuracy:.1%}")
    console.print(f"Loop completion: {result.loop_completion_rate:.1%}")

    if result.failures:
        console.print(f"\n[yellow]Failures ({len(result.failures)}):[/yellow]")
        for f in result.failures[:5]:
            console.print(f"  - {f['error'][:60]}...")

    # Save report
    state_manager: StateManager = ctx.obj["state_manager"]
    state_manager.ensure_dirs()

    report_path = state_manager.get_report_path("validation_latest.json")
    with open(report_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    console.print(f"\nReport saved: {report_path}")

    if not result.passed:
        raise SystemExit(1)


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to model")
@click.option("--tools", "-t", required=True, type=click.Path(exists=True), help="Tools JSON file")
@click.option("--baseline", type=click.Path(exists=True), help="Baseline benchmark JSON for comparison")
@click.option("--samples-per-tool", default=20, help="Samples per tool (default: 20)")
@click.option("--samples-per-scenario", default=20, help="Samples per scenario (default: 20)")
@click.option("--stub", type=click.Choice(["weather", "filesystem"]), help="Use deterministic stub")
@click.option("--output", "-o", help="Output directory for reports")
@click.pass_context
def benchmark(
    ctx,
    model: str,
    tools: str,
    baseline: str | None,
    samples_per_tool: int,
    samples_per_scenario: int,
    stub: str | None,
    output: str | None,
):
    """Run full evaluation benchmark suite.

    Measures tool accuracy, no-tool correctness, and response latency
    across all tools and scenarios. Generates detailed reports in JSON
    and Markdown format.

    \b
    Metrics Measured:
      - Tool selection accuracy (target: >=90%)
      - No-tool correctness (target: >=85%)
      - Loop completion rate (target: >=95%)
      - Response latency (mean, p50, p95, p99)

    \b
    Example:
      mcp-forge benchmark -m ./adapter -t tools.json --stub weather
    """
    import json

    from mcp_forge.state import BenchmarkResult, ToolDefinition
    from mcp_forge.validation import BenchmarkConfig, BenchmarkRunner
    from mcp_forge.validation.config import StubConfig

    console.print("\n[bold]MCP-Forge Benchmark Suite[/bold]")
    console.print("=" * 50)

    # Load tools
    with open(tools) as f:
        tool_defs = [ToolDefinition.from_dict(t) for t in json.load(f)]

    console.print(f"Model: {model}")
    console.print(f"Tools: {len(tool_defs)}")
    console.print(f"Samples: {samples_per_tool}/tool, {samples_per_scenario}/scenario")

    # Configure benchmark
    if stub:
        stub_config = StubConfig(stub_type=stub, deterministic=True)
        mcp_command = None
    else:
        stub_config = None
        mcp_command = None  # Would need --server option for real MCP

    if not stub_config and not mcp_command:
        console.print("[yellow]Warning: No --stub specified, using weather stub by default[/yellow]")
        stub_config = StubConfig(stub_type="weather", deterministic=True)

    config = BenchmarkConfig(
        model_path=Path(model),
        model_name=Path(model).name,
        samples_per_tool=samples_per_tool,
        samples_per_scenario=samples_per_scenario,
        stub_config=stub_config,
        mcp_command=mcp_command,
        baseline_path=Path(baseline) if baseline else None,
    )

    runner = BenchmarkRunner(config, tool_defs)

    # Run benchmark
    console.print("\nRunning benchmark...")

    def on_progress(category: str, current: int, total: int, pct: float) -> None:
        if current % 10 == 0 or current == total:
            console.print(f"  {category}: {current}/{total} ({pct:.0%})")

    try:
        result = runner.run(progress_callback=on_progress)
    except Exception as e:
        console.print(f"\n[red]Benchmark failed: {e}[/red]")
        raise SystemExit(1) from e

    # Load and compare to baseline if provided
    if baseline:
        with open(baseline) as f:
            baseline_data = json.load(f)
        baseline_result = BenchmarkResult.from_dict(baseline_data)
        result.baseline_comparison = runner.compare_to_baseline(result, baseline_result)

    # Print results
    console.print("\n" + "=" * 50)
    console.print("[bold]Benchmark Results[/bold]")
    console.print(f"Overall Score: {result.overall_score:.1%}")

    console.print("\n[bold]Per-Tool Results:[/bold]")
    for tool, metrics in result.per_tool_results.items():
        console.print(f"  {tool}:")
        console.print(f"    Accuracy: {metrics.get('accuracy', 0):.1%}")
        console.print(f"    Schema: {metrics.get('schema', 0):.1%}")
        console.print(f"    Latency: {metrics.get('latency_mean_ms', 0):.1f}ms (p95: {metrics.get('latency_p95_ms', 0):.1f}ms)")

    console.print("\n[bold]Per-Scenario Results:[/bold]")
    for scenario, metrics in result.per_scenario_results.items():
        console.print(f"  {scenario}: {metrics.get('pass_rate', 0):.1%}")

    if result.baseline_comparison:
        console.print("\n[bold]Baseline Comparison:[/bold]")
        console.print(f"  vs {result.baseline_comparison['baseline_model']}")
        delta = result.baseline_comparison['overall_delta']
        color = "green" if delta >= 0 else "red"
        console.print(f"  Overall delta: [{color}]{delta:+.1%}[/{color}]")

    # Save reports
    state_manager: StateManager = ctx.obj["state_manager"]
    report_path = state_manager.save_benchmark_result(result)
    console.print(f"\nReports saved: {report_path}")

    # Check thresholds
    no_tool_rate = result.per_scenario_results.get("no_tool", {}).get("pass_rate", 0)
    tool_accuracies = [m.get("accuracy", 0) for m in result.per_tool_results.values()]
    avg_accuracy = sum(tool_accuracies) / len(tool_accuracies) if tool_accuracies else 0

    passes = (
        avg_accuracy >= config.accuracy_threshold and
        no_tool_rate >= config.no_tool_threshold
    )

    if passes:
        console.print("\n[green]Benchmark passed all thresholds![/green]")
    else:
        console.print("\n[yellow]Warning: Some metrics below target thresholds[/yellow]")
        if avg_accuracy < config.accuracy_threshold:
            console.print(f"  Tool accuracy: {avg_accuracy:.1%} (target: {config.accuracy_threshold:.0%})")
        if no_tool_rate < config.no_tool_threshold:
            console.print(f"  No-tool correctness: {no_tool_rate:.1%} (target: {config.no_tool_threshold:.0%})")


# =============================================================================
# Export & Packaging Commands
# =============================================================================

@cli.command("export")
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to LoRA adapter")
@click.option("--format", "quant_format", type=click.Choice(["q8_0", "q4_k_m", "q4_k_s", "q5_k_m", "f16"]), default="q8_0", help="Quantization format")
@click.option("--output", "-o", required=True, help="Output GGUF file path")
@click.option("--base-model", help="Base model (auto-detected from adapter if not specified)")
@click.option("--no-verify", is_flag=True, help="Skip GGUF verification after export")
@click.pass_context
def export_model(
    ctx: click.Context,
    model: str,
    quant_format: str,
    output: str,
    base_model: str | None,
    no_verify: bool,
) -> None:
    """Export LoRA adapter to GGUF format.

    Merges the LoRA adapter with the base model and converts to
    quantized GGUF format for deployment with llama.cpp or Ollama.

    \b
    Quantization Formats:
      q8_0   - 8-bit, best quality (default)
      q4_k_m - 4-bit k-quant, good balance
      q4_k_s - 4-bit k-quant small
      q5_k_m - 5-bit k-quant medium
      f16    - Half precision (no quantization)

    \b
    Example:
      mcp-forge export -m ./adapter -o model.gguf --format q4_k_m
    """
    from pathlib import Path

    from mcp_forge.export import ExportConfig, ExportEngine, QuantizationType
    from mcp_forge.export.metadata import GGUFMetadata

    console.print("\n[bold]MCP-Forge GGUF Export[/bold]")
    console.print("=" * 50)

    adapter_path = Path(model)
    output_path = Path(output)

    # Ensure output has .gguf extension
    if not output_path.suffix == ".gguf":
        output_path = output_path.with_suffix(".gguf")

    console.print(f"Adapter: {adapter_path}")
    console.print(f"Output: {output_path}")
    console.print(f"Quantization: {quant_format}")

    # Detect base model from adapter config if not specified
    detected_base = base_model
    if not detected_base:
        adapter_config = adapter_path / "adapter_config.json"
        if adapter_config.exists():
            import json
            with open(adapter_config) as f:
                config_data = json.load(f)
            detected_base = config_data.get("base_model_name_or_path", "")
            console.print(f"Detected base model: {detected_base}")

    if not detected_base:
        detected_base = "unsloth/DeepSeek-R1-Distill-Qwen-7B"
        console.print(f"[yellow]Using default base model: {detected_base}[/yellow]")

    # Create config
    config = ExportConfig(
        adapter_path=adapter_path,
        output_path=output_path,
        base_model=detected_base,
        quantization=QuantizationType(quant_format),
        verify_after_export=not no_verify,
    )

    # Build metadata
    state_manager: StateManager | None = ctx.obj.get("state_manager") if ctx.obj else None
    metadata = GGUFMetadata(
        model_name=adapter_path.name,
        model_family=detected_base.split("/")[-1] if "/" in detected_base else detected_base,
        quantization_type=quant_format,
    )

    # If we have pipeline state, enrich metadata
    if state_manager:
        try:
            state = state_manager.load_state()
            if state:
                metadata.tool_names = [t.name for t in state.tools]
                metadata.tool_count = len(state.tools)
                if state.validation_result:
                    metadata.tool_accuracy = state.validation_result.tool_selection_accuracy
                    metadata.schema_conformance = state.validation_result.schema_conformance_rate
                if state.benchmark_result:
                    metadata.benchmark_score = state.benchmark_result.overall_score
        except Exception:
            pass  # No state available, continue without

    engine = ExportEngine(config)

    # Run export with progress
    console.print("\nStarting export...")

    def on_progress(stage: str, pct: float) -> None:
        console.print(f"  [{pct:.0%}] {stage}")

    try:
        result = engine.export(metadata=metadata, progress_callback=on_progress)
    except Exception as e:
        console.print(f"\n[red]Export failed: {e}[/red]")
        raise SystemExit(1) from e

    # Display results
    console.print("\n" + "=" * 50)

    if result.success:
        console.print("[bold green]Export successful![/bold green]")
        console.print(f"\nOutput: {result.output_path}")
        console.print(f"Size: {result.gguf_size_mb:.1f} MB")
        console.print(f"Compression: {result.compression_ratio:.1f}x")
        console.print(f"Time: {result.total_time_seconds:.1f}s")

        if result.verified:
            console.print("[green]Verification: PASSED[/green]")
        elif no_verify:
            console.print("[yellow]Verification: SKIPPED[/yellow]")

        # Save export report
        if state_manager:
            state_manager.ensure_dirs()
            import json
            report_path = state_manager.get_report_path("export_latest.json")
            with open(report_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            console.print(f"\nReport: {report_path}")
    else:
        console.print(f"[red]Export failed: {result.error}[/red]")
        raise SystemExit(1)


@cli.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to GGUF model")
@click.option("--tools", "-t", type=click.Path(exists=True), help="Tools JSON file")
@click.option("--validation", type=click.Path(exists=True), help="Validation report JSON")
@click.option("--benchmark", "bench", type=click.Path(exists=True), help="Benchmark report JSON")
@click.option("--output", "-o", required=True, help="Output directory for bundle")
@click.option("--name", help="Model name for bundle (default: derived from model filename)")
@click.option("--description", help="Short description for README")
@click.option("--no-modelfile", is_flag=True, help="Skip Ollama Modelfile generation")
@click.option("--no-readme", is_flag=True, help="Skip README generation")
@click.pass_context
def pack(
    ctx,
    model: str,
    tools: str | None,
    validation: str | None,
    bench: str | None,
    output: str,
    name: str | None,
    description: str | None,
    no_modelfile: bool,
    no_readme: bool,
):
    """Create distributable agent bundle.

    Packages a GGUF model with tool definitions, deployment configuration,
    and documentation into a distributable bundle.

    \b
    Bundle Contents:
      model.gguf   - The quantized model file
      tools.json   - Tool schema definitions (OpenAI function format)
      config.yaml  - Deployment configuration
      README.md    - Usage instructions (optional)
      Modelfile    - Ollama import file (optional)

    \b
    Examples:
      # Basic bundle from GGUF and tools
      mcp-forge pack -m model.gguf -t tools.json -o ./dist/agent

      # With validation/benchmark reports
      mcp-forge pack -m model.gguf -t tools.json --validation val.json -o ./dist/agent

      # From pipeline state (uses saved tools and metrics)
      mcp-forge pack -m model.gguf -o ./dist/agent
    """
    import json

    from mcp_forge.export.bundle import BundleConfig, BundleEngine
    from mcp_forge.state import ToolDefinition

    console.print("\n[bold]MCP-Forge Bundle Packaging[/bold]")
    console.print("=" * 50)

    state_manager: StateManager = ctx.obj["state_manager"]

    # Load tools from file or pipeline state
    tool_defs: list[ToolDefinition] = []
    if tools:
        console.print(f"Loading tools from {tools}...")
        with open(tools) as f:
            tool_defs = [ToolDefinition.from_dict(t) for t in json.load(f)]
    else:
        # Try to load from pipeline state
        state = state_manager.load_state()
        if state and state.tools:
            console.print("Using tools from pipeline state...")
            tool_defs = state.tools
        else:
            console.print("[red]Error: No tools specified. Use --tools or run pipeline first.[/red]")
            raise SystemExit(1)

    console.print(f"Model: {model}")
    console.print(f"Tools: {len(tool_defs)}")
    console.print(f"Output: {output}")

    # Load metrics from reports or pipeline state
    tool_accuracy = None
    schema_conformance = None
    benchmark_score = None
    model_family = "unknown"
    training_samples = 0

    # Check validation report
    if validation:
        try:
            with open(validation) as f:
                val_data = json.load(f)
            tool_accuracy = val_data.get("tool_selection_accuracy")
            schema_conformance = val_data.get("schema_conformance_rate")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load validation report: {e}[/yellow]")

    # Check benchmark report
    if bench:
        try:
            with open(bench) as f:
                bench_data = json.load(f)
            benchmark_score = bench_data.get("overall_score")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load benchmark report: {e}[/yellow]")

    # Enrich from pipeline state if available
    state = state_manager.load_state()
    if state:
        if state.validation_result and tool_accuracy is None:
            tool_accuracy = state.validation_result.tool_selection_accuracy
            schema_conformance = state.validation_result.schema_conformance_rate
        if state.benchmark_result and benchmark_score is None:
            benchmark_score = state.benchmark_result.overall_score
        model_family = state.model_family
        if state.synthesis_plan:
            training_samples = state.synthesis_plan.total_samples

    # Create config
    config = BundleConfig(
        gguf_path=Path(model),
        tools=tool_defs,
        output_dir=Path(output),
        include_modelfile=not no_modelfile,
        include_readme=not no_readme,
        model_name=name or "",
        model_description=description or "",
        tool_accuracy=tool_accuracy,
        schema_conformance=schema_conformance,
        benchmark_score=benchmark_score,
        model_family=model_family,
        training_samples=training_samples,
    )

    # Create bundle
    engine = BundleEngine(config)

    def on_progress(msg: str) -> None:
        console.print(f"  {msg}")

    console.print("\nPackaging bundle...")
    result = engine.package(progress_callback=on_progress)

    # Display results
    console.print("\n" + "=" * 50)

    if result.success:
        console.print("[bold green]Bundle created successfully![/bold green]")
        console.print(f"\nLocation: {result.bundle_path}")
        console.print(f"Size: {result.bundle_size_mb:.1f} MB")
        console.print("\nFiles:")
        for f in result.files_created:
            console.print(f"  - {f}")

        if result.validation_passed:
            console.print("\n[green]Validation: PASSED[/green]")
        else:
            console.print("\n[yellow]Validation warnings:[/yellow]")
            for err in result.validation_errors:
                console.print(f"  - {err}")

        # Print next steps
        console.print("\n[bold]Next Steps:[/bold]")
        console.print(f"  1. cd {output}")
        console.print(f"  2. ollama create {config.model_name} -f Modelfile")
        console.print(f"  3. ollama run {config.model_name}")

        # Save report
        state_manager.ensure_dirs()
        report_path = state_manager.get_report_path("bundle_latest.json")
        with open(report_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"\nReport: {report_path}")

    else:
        console.print(f"[red]Bundle creation failed: {result.error}[/red]")
        if result.validation_errors:
            console.print("\nErrors:")
            for err in result.validation_errors:
                console.print(f"  - {err}")
        raise SystemExit(1)


@cli.command("verify-bundle")
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--smoke-test", is_flag=True, help="Run inference smoke test (requires llama-cpp-python)")
def verify_bundle_cmd(bundle_path: str, smoke_test: bool):
    """Verify an agent bundle and optionally run smoke tests.

    Checks bundle integrity including:
    - Required files present (model.gguf, tools.json, config.yaml)
    - GGUF file has valid magic bytes
    - JSON/YAML files are valid
    - Tool counts match across files

    With --smoke-test, also loads the model and runs a quick inference
    to verify it works (requires llama-cpp-python).

    \b
    Example:
      mcp-forge verify-bundle ./dist/agent
      mcp-forge verify-bundle ./dist/agent --smoke-test
    """
    from mcp_forge.export.bundle import verify_bundle

    console.print("\n[bold]MCP-Forge Bundle Verification[/bold]")
    console.print("=" * 50)
    console.print(f"Bundle: {bundle_path}")

    if smoke_test:
        console.print("Mode: Full verification with smoke test")
    else:
        console.print("Mode: File integrity check")

    console.print("\nVerifying...")

    valid, errors = verify_bundle(Path(bundle_path), smoke_test=smoke_test)

    # List bundle contents
    bundle_dir = Path(bundle_path)
    files = list(bundle_dir.iterdir())
    console.print(f"\nBundle contents ({len(files)} files):")
    total_size = 0
    for f in sorted(files):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} bytes"
            console.print(f"  {f.name}: {size_str}")

    console.print(f"\nTotal size: {total_size / (1024 * 1024):.1f} MB")

    # Display results
    console.print("\n" + "=" * 50)

    if valid:
        console.print("[bold green]Bundle verification PASSED[/bold green]")

        # Show bundle info from config
        config_path = bundle_dir / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            console.print("\nBundle info:")
            if config and "model" in config:
                console.print(f"  Model: {config['model'].get('name', 'unknown')}")
                console.print(f"  Family: {config['model'].get('family', 'unknown')}")
            if config and "tools" in config:
                console.print(f"  Tools: {config['tools'].get('count', 0)}")

    else:
        console.print("[bold red]Bundle verification FAILED[/bold red]")
        console.print("\nErrors:")
        for err in errors:
            console.print(f"  [red]✗[/red] {err}")
        raise SystemExit(1)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
