# Stack

## Languages

| Language | Version | Usage |
|----------|---------|-------|
| Python | >=3.10 | Primary language for all components |

## Frameworks

| Framework | Version | Purpose |
|-----------|---------|---------|
| Click | >=8.1.0 | CLI framework for command routing |
| Rich | >=13.0.0 | Terminal output formatting, tables, progress bars |
| Unsloth | git+latest | 2x faster LoRA fine-tuning with 70% less VRAM |
| TRL | >=0.8.0 | SFTTrainer for supervised fine-tuning |
| Transformers | >=4.40.0 | Model loading and tokenization |

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mcp | >=1.0.0 | Model Context Protocol SDK for server inspection |
| openai | >=1.0.0 | GPT-4o API for seed data generation |
| pydantic | >=2.0.0 | Data validation |
| aiofiles | >=23.0.0 | Async file I/O |
| torch | >=2.1.0 | Deep learning framework |
| datasets | >=2.18.0 | Dataset handling |
| bitsandbytes | >=0.43.0 | 4-bit quantization support |
| jsonschema | (implied) | JSON Schema validation in QC |

## Development Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=7.0.0 | Test framework |
| pytest-asyncio | >=0.21.0 | Async test support |
| pytest-cov | >=4.0.0 | Coverage reporting |
| ruff | >=0.1.0 | Linting and formatting |
| mypy | >=1.0.0 | Static type checking |

## Build Tools

| Tool | Purpose |
|------|---------|
| hatchling | Build backend for pyproject.toml |
| pip | Package installation |

## Runtime Requirements

| Requirement | Specification |
|-------------|---------------|
| CUDA | 12.x required for training |
| VRAM | 16GB+ recommended (6GB min for 8B models) |
| GPU | NVIDIA RTX 3090 or equivalent |
| Disk Space | 50GB+ free recommended |
| OS | Linux (Ubuntu 22.04/24.04) |
