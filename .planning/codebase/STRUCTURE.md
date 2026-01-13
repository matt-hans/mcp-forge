# Structure

## Current Layout (Flat, Pre-Modularization)

```
mcp-forge/
├── .claude/                    # Claude Code configuration
│   ├── commands/gsd/           # GSD workflow commands
│   ├── get-shit-done/          # GSD templates and references
│   ├── hooks/                  # Session hooks
│   ├── rules/                  # Project rules
│   └── settings.json           # Claude settings
├── .planning/                  # Project planning (created by GSD)
│   └── codebase/               # This codebase map
├── cli.py                      # CLI entry point (Click commands)
├── state.py                    # Pipeline state management
├── inspector.py                # MCP server inspection
├── qc.py                       # Data quality controller
├── pyproject.toml              # Project configuration
├── CLAUDE.md                   # Project documentation
└── TECHNICAL_ARCHITECTURE_v1.1.md  # Architecture document
```

## Planned Layout (from TECHNICAL_ARCHITECTURE)

```
mcp-forge/
├── src/
│   └── mcp_forge/
│       ├── __init__.py
│       ├── cli.py              # Click command definitions
│       ├── config.py           # Configuration management
│       ├── state.py            # Pipeline state management
│       ├── tools/              # Tool layer
│       │   ├── provider.py     # ToolProvider abstraction
│       │   ├── mcp_provider.py # MCP server inspector
│       │   ├── file_provider.py # JSON/YAML file loader
│       │   └── openai_adapter.py
│       ├── data/               # Data layer
│       │   ├── synthesizer.py  # Hybrid data generation
│       │   ├── synthesis_plan.py
│       │   ├── qc.py           # Quality control engine
│       │   └── formatter.py    # Chat template handling
│       ├── training/           # Training layer
│       │   ├── trainer.py      # Unsloth training wrapper
│       │   └── profiles.py     # Training profiles
│       ├── eval/               # Evaluation layer
│       │   ├── validator.py    # Looped validation
│       │   ├── benchmark.py    # Evaluation suite
│       │   └── stubs.py        # Deterministic MCP stubs
│       ├── export/             # Export layer
│       │   ├── exporter.py     # GGUF conversion
│       │   └── packager.py     # Agent bundle creation
│       └── models/             # Model-specific configs
│           ├── deepseek.py
│           └── qwen.py
├── tests/
├── pyproject.toml
└── README.md
```

## Module Responsibilities

| Module | Current Location | Responsibility |
|--------|------------------|----------------|
| CLI | `cli.py` | Command routing, orchestration, user interaction |
| State | `state.py` | Pipeline state, checkpointing, dataclasses |
| Inspector | `inspector.py` | MCP connection, tool schema extraction |
| QC | `qc.py` | Data validation, dedup, coverage analysis |

## Runtime Directory Structure

```
.mcp-forge/                     # Created at runtime
├── state.json                  # Pipeline checkpoint
├── data/                       # Training data files
│   ├── seed.jsonl              # GPT-4o generated seeds
│   └── train.jsonl             # Validated training data
├── logs/                       # Pipeline logs
└── reports/                    # QC and benchmark reports
    ├── qc_*.json
    └── benchmark_*.json
```

## Output Structure (Agent Bundle)

```
dist/<agent_name>/
├── model.gguf                  # Quantized model
├── tools.json                  # Tool definitions
├── prompt_template.json        # System prompt + format
├── agent.yaml                  # Metadata manifest
├── tests/
│   └── validation.json         # Benchmark prompts + expected results
├── README.md                   # Usage instructions
└── Modelfile                   # Ollama-ready config
```

## Package Entry Points

Defined in `pyproject.toml:54-55`:
- `mcp-forge` → `mcp_forge.cli:main`
- `forge` → `mcp_forge.cli:main`
