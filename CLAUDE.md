# CLAUDE.md — autoresearch-lab

## Project Overview

General-purpose autonomous research engine extending karpathy/autoresearch.
Point it at any tabular dataset — it scaffolds a complete experiment (prepare.py + train.py + program.md), runs a baseline, then an AI agent loops forever improving the model.

## Architecture

```
Data (CSV/Parquet/Excel)
    ↓ LLM chat: "what do you want to predict?"
Scaffold (generator/scaffold.py)
    ↓
Experiment directory:
  prepare.py   — locked evaluation harness (DO NOT MODIFY)
  train.py     — universal baseline (agent edits this)
  program.md   — experiment rules + git safety
  flow.excalidraw — visual flow diagram
    ↓
Autoresearch agent loops: edit → commit → run → evaluate → keep/revert
```

Two training backends for language model pretraining demo:
- **CUDA** (original): `train.py` + `prepare.py` — requires NVIDIA GPU
- **MLX** (Apple Silicon): `train_mlx.py` + `prepare_mlx.py` — runs natively on Mac

## Repository Structure

```
train.py / prepare.py           — CUDA backend (language modeling demo)
train_mlx.py / prepare_mlx.py   — MLX backend (Apple Silicon)
program.md                       — experiment protocol
generator/                       — scaffold system
  scaffold.py                    — experiment scaffolder (data → chat → full experiment dir)
  auto_detect.py                 — LLM chat to determine target/task/metric
  llm_client.py                  — provider-agnostic LLM client (LiteLLM)
experiments/                     — scaffolded experiment directories (gitignored)
```

## Key Rules

1. **NEVER modify** `prepare.py` or `prepare_mlx.py` — they contain fixed evaluation harnesses.
2. **Only modify** `train.py` (CUDA) or `train_mlx.py` (MLX) during experiments.
3. Generated `program.md` must remain human-readable.
4. Preserve reproducibility — use fixed seeds.

## Running

```bash
# Interactive chat — LLM asks what you want to predict
uv run python -m generator.scaffold \
    --data path/to/data.csv --output-dir experiments/my-exp

# One-shot — describe your goal
uv run python -m generator.scaffold \
    --data path/to/data.csv --output-dir experiments/my-exp \
    --description "predict yield from soil data"

# Manual — skip LLM
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
```

## Dependencies

```bash
uv pip install -e ".[domain,llm]" # Scaffold (pandas, sklearn, litellm)
uv pip install -e ".[mlx]"        # + MLX training backend
uv pip install -e ".[cuda]"       # + CUDA training backend
```
