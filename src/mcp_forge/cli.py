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

        # TODO: Implement QC validation
        console.print("   [yellow]QC validation not yet implemented[/yellow]")
        state.training_data_path = str(state_manager.get_data_path("train.jsonl"))

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

@cli.command()
@click.option("--data", "-d", required=True, type=click.Path(exists=True), help="Training data JSONL file")
@click.option("--tools", "-t", required=True, type=click.Path(exists=True), help="Tools JSON file")
@click.option("--fix", is_flag=True, help="Auto-repair and rewrite cleaned data")
@click.option("--output", "-o", help="Output file for cleaned data")
@click.pass_context
def qa(ctx, data: str, tools: str, fix: bool, output: str | None):
    """Run dataset quality analysis and optional cleanup."""
    import json

    from mcp_forge.data.qc import DataQualityController, QCConfig
    from mcp_forge.state import ToolDefinition

    console.print("\n[bold]Running Dataset QA...[/bold]")

    # Load tools
    with open(tools) as f:
        tool_defs = [ToolDefinition.from_dict(t) for t in json.load(f)]

    # Configure QC
    config = QCConfig(auto_repair=fix)
    qc = DataQualityController(tool_defs, config)

    # Determine output path
    output_path = Path(output) if output else (Path(data).with_suffix(".clean.jsonl") if fix else None)

    # Run validation
    report, validated = qc.validate_dataset(Path(data), output_path)

    # Print report
    qc.print_report(report)

    if output_path:
        console.print(f"\n[green]Cleaned data saved to {output_path}[/green]")

    # Save report
    state_manager: StateManager = ctx.obj["state_manager"]
    report_path = state_manager.save_qc_report(report)
    console.print(f"[green]Report saved to {report_path}[/green]")


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

    console.print(f"\nResults:")
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
