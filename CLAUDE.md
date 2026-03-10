# CLAUDE.md — autoresearch-lab

## Project Overview

General-purpose autonomous research engine extending karpathy/autoresearch.
Transforms autoresearch into a system that works across domains (agriculture, social science, healthcare, finance) by generating `program.md` from structured domain questions or LLM-powered conversations.

## Architecture

```
Domain expert describes problem
    ↓ (natural language or questionnaire)
Program Generator (template or LLM-powered)
    ↓
program.md
    ↓
Autoresearch agent runs experiments autonomously
```

Two training backends:
- **CUDA** (original): `train.py` + `prepare.py` — requires NVIDIA GPU
- **MLX** (Apple Silicon): `train_mlx.py` + `prepare_mlx.py` — runs natively on Mac

Four generation modes:
- **Scaffold**: Data file → complete experiment directory (prepare.py + train.py + program.md)
- **Template**: Questionnaire → YAML → Jinja2 → program.md
- **Config**: JSON file → program.md (for automation)
- **LLM**: Natural language conversation → program.md (any provider via LiteLLM)

## Repository Structure

```
train.py / prepare.py           — CUDA backend (karpathy/autoresearch)
train_mlx.py / prepare_mlx.py   — MLX backend (Apple Silicon)
program.md                       — experiment protocol
generator/                       — program generator system
  scaffold.py                    — experiment scaffolder (data → full experiment dir)
  program_generator.py           — unified CLI entry point (template + LLM modes)
  llm_generator.py               — LLM-powered conversational generator
  llm_client.py                  — provider-agnostic LLM client (LiteLLM)
  domain_questions.py            — domain Q&A engine
  templates.py                   — program.md template renderer
experiments/                     — scaffolded experiment directories (gitignored)
domains/                         — domain YAML configs
  agriculture.yaml
  social_science.yaml
  healthcare.yaml
  finance.yaml
examples/                        — example generated programs
```

## Key Rules

1. **NEVER modify** `prepare.py` or `prepare_mlx.py` — they contain fixed evaluation harnesses.
2. **Only modify** `train.py` (CUDA) or `train_mlx.py` (MLX) during experiments.
3. Generated `program.md` must remain human-readable.
4. Preserve reproducibility — use fixed seeds.
5. Keep domain input, program generation, and experiment execution separate.

## Running

```bash
# Scaffold a domain experiment from data
uv run python -m generator.scaffold \
    --data path/to/data.csv --target column_name \
    --metric mae --task regression --output-dir experiments/my-exp

# Run the scaffolded experiment
cd experiments/my-exp && uv run python train.py

# Language model pretraining (original autoresearch)
uv run prepare_mlx.py              # macOS
uv run prepare.py                  # CUDA
uv run train_mlx.py                # macOS
uv run train.py                    # CUDA

# Generate program.md — template mode
uv run python -m generator.program_generator --domain agriculture

# Generate program.md — LLM mode
uv run python -m generator.program_generator --llm --model local
uv run python -m generator.program_generator --llm --model gpt4o --domain agriculture

# List available LLM presets
uv run python -m generator.program_generator --list-models
```

## Dependencies

```bash
uv pip install -e "."             # Base (generator with templates)
uv pip install -e ".[llm]"       # + LLM support (litellm)
uv pip install -e ".[mlx]"       # + MLX training backend
uv pip install -e ".[cuda]"      # + CUDA training backend
```
