# autoresearch-lab

General-purpose autonomous research engine. Extends [karpathy/autoresearch](https://github.com/karpathy/autoresearch) with an LLM-powered program generator and Apple Silicon support.

An AI agent runs experiments autonomously — modifying code, training, evaluating, keeping improvements, discarding failures — in a loop, while you sleep.

## The idea

The original [autoresearch](https://github.com/karpathy/autoresearch) proved that an AI agent can do meaningful ML research autonomously on a single GPU. But it's locked to one task (language model pretraining on text).

**autoresearch-lab** extracts the pattern and makes it reproducible across domains:

```
The pattern (works for anything):

    1. Define the problem         → program.md
    2. Lock the evaluation        → prepare.py (read-only)
    3. Let the agent experiment   → train.py (agent edits this)
    4. Fixed time budget          → 5 minutes per experiment
    5. Keep or discard            → results.tsv
    6. Loop forever
```

The included training scripts demonstrate this with **language model pretraining** (the original autoresearch task). The program generator + domain system shows how the same pattern applies to any field.

## Why this pattern works for any domain

The core loop is domain-agnostic:

```
modify code → run → evaluate → better? keep : discard → repeat
```

What changes per domain is:

| Component | Language Modeling (included) | Agriculture (example) | Healthcare (example) |
|-----------|----------------------------|----------------------|---------------------|
| **Data** | climbmix-400b text corpus | Crop yield CSV | Patient outcomes |
| **Model** | GPT transformer | XGBoost / Random Forest | Logistic Regression |
| **Metric** | val_bpb (bits per byte) | MAE | AUC-ROC |
| **Constraints** | Don't modify eval | Interpretability required | Clinical explainability |
| **What agent edits** | train.py | train.py | train.py |

The experiment loop, the program.md format, the keep/discard logic, the results logging — all identical. Only the contents of `prepare.py` + `train.py` + `program.md` change.

### To adapt this for your domain:

1. **Scaffold an experiment** from your data — generates `prepare.py`, `train.py`, and `program.md` automatically
2. **Run the loop** — the agent takes it from here

```bash
# One command generates everything
uv run python -m generator.scaffold \
    --data your_data.csv \
    --target target_column \
    --metric mae \
    --task regression \
    --output-dir experiments/my-experiment

# Run it
cd experiments/my-experiment
uv run python train.py
```

Or do it manually: generate `program.md` with the LLM generator, write `prepare.py` and `train.py` yourself.

## What's included

### Training demo (language modeling)

The included `train.py` / `prepare.py` are the actual autoresearch training scripts — a real, working autonomous research setup:

- **Dataset**: [climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) — 400B token text corpus
- **Model**: GPT with RoPE, sliding window attention, value embeddings, ReluSquared MLP
- **Optimizer**: MuonAdamW (CUDA) / AdamW (MLX)
- **Metric**: val_bpb (bits per byte) — lower is better
- **Budget**: 5 minutes per experiment

Two backends:
- `train.py` + `prepare.py` — CUDA (NVIDIA GPU, PyTorch, Flash Attention 3)
- `train_mlx.py` + `prepare_mlx.py` — MLX (Apple Silicon, no PyTorch)

### Experiment scaffold

Point it at your data, tell it what to predict — it generates a complete experiment directory with `prepare.py` (locked evaluation harness), `train.py` (agent-editable baseline), and `program.md` (experiment rules).

Supports regression (mae, rmse, r2) and classification (auc, f1, accuracy). Auto-detects numeric vs categorical columns, builds sklearn pipelines, handles encoding and missing values.

### Program generator

An LLM-powered tool that creates `program.md` from natural language. Domain experts describe their problem; the system outputs a structured experiment plan.

Pre-configured domains: **agriculture**, **healthcare**, **social science**, **finance** — each with tailored questions, constraints, and model recommendations.

## Quick start

### 1. Scaffold an experiment

```bash
git clone <repo-url>
cd autoresearch-lab
uv pip install -e ".[domain]"

# Auto-detect — just point at your data
uv run python -m generator.scaffold \
    --data path/to/data.csv \
    --output-dir experiments/my-experiment

# Or specify everything manually
uv run python -m generator.scaffold \
    --data path/to/data.csv \
    --target target_column \
    --metric mae \
    --task regression \
    --output-dir experiments/my-experiment
```

Auto-detect analyzes your data and picks the target column, task type, and metric. You can override any of them.

Supported metrics: `mae`, `rmse`, `r2` (regression) | `auc`, `f1`, `accuracy` (classification)

### 2. Run the baseline

```bash
cd experiments/my-experiment
uv run python train.py
```

The baseline prints a **time budget suggestion** based on how fast it trained — tells you whether to increase or decrease `TIME_BUDGET` in `prepare.py` before running overnight.

### 3. Run the autonomous loop

```bash
# Initialize git (the agent uses commits to track experiments)
git init && git add -A && git commit -m "baseline"

# Launch Claude Code in autonomous mode
claude --dangerously-skip-permissions
```

Claude reads `program.md`, understands the rules, and loops forever:

```
Edit train.py → git commit → uv run train.py → better? → keep or revert → repeat
```

The agent only commits `train.py` — never pushes, never commits logs or results.

### Language model pretraining (original autoresearch)

```bash
cp .env.example .env
uv sync
uv run prepare_mlx.py              # macOS (Apple Silicon)
uv run prepare.py                  # Linux/CUDA
uv run train_mlx.py                # macOS
uv run train.py                    # CUDA
```

## Scaffolding experiments

Generate a complete experiment directory from your data:

```bash
# Auto-detect everything (just point at data)
uv run python -m generator.scaffold \
    --data data.csv \
    --output-dir experiments/my-experiment

# Auto-detect with LLM (smarter analysis)
uv run python -m generator.scaffold \
    --data data.csv \
    --output-dir experiments/my-experiment \
    --llm --model gpt4o

# Manual (specify everything)
uv run python -m generator.scaffold \
    --data data.csv \
    --target yield \
    --metric mae \
    --task regression \
    --output-dir experiments/my-experiment \
    --time-budget 300
```

This creates:

| File | Purpose |
|------|---------|
| `prepare.py` | Data loading + evaluation harness (DO NOT MODIFY) |
| `train.py` | Universal baseline model (agent edits this — works for any dataset) |
| `program.md` | Experiment rules, git rules, and constraints |
| `flow.excalidraw` | Visual flow diagram (open in [excalidraw.com](https://excalidraw.com)) |
| `pyproject.toml` | Dependencies for the experiment |
| `.gitignore` | Ignores logs, results, and build artifacts |

### How it works

- **`prepare.py`** is generated per-experiment with the data path, target column, metric, and eval function baked in. Read-only.
- **`train.py`** is universal — one file works for any dataset. It reads `TASK_TYPE` from `prepare.py` and auto-selects the right model (GradientBoostingRegressor or GradientBoostingClassifier). The agent modifies this file.
- **Auto-detect** analyzes column types, names, and cardinality to guess target/task/metric. You can override any field.
- **Time budget suggestion** — after the baseline runs, it suggests an appropriate `TIME_BUDGET` based on training time.
- **Flow diagram** — each experiment gets a customized Excalidraw diagram showing the data flow and experiment loop with the actual metric name.

## Generating program.md

Three ways to create an experiment plan (without scaffolding):

### 1. LLM-powered (recommended)

Describe your problem in natural language. The LLM asks follow-up questions and generates a tailored program.md.

```bash
# Install LLM support
uv pip install litellm

# Local model (Ollama — no API key needed)
uv run python -m generator.program_generator --llm --model local

# OpenAI
uv run python -m generator.program_generator --llm --model gpt4o

# Claude
uv run python -m generator.program_generator --llm --model claude-sonnet

# With domain context (loads domain YAML for smarter questions)
uv run python -m generator.program_generator --llm --model local --domain agriculture

# Single-shot from file (no conversation, good for CI)
uv run python -m generator.program_generator --llm --model gpt4o --description problem.txt

# Custom OpenAI-compatible server
uv run python -m generator.program_generator --llm --model openai/my-model --base-url http://localhost:8000/v1
```

### 2. Interactive questionnaire

Answer domain-specific questions. Answers are mapped to a structured template.

```bash
uv run python -m generator.program_generator --domain agriculture
uv run python -m generator.program_generator --domain healthcare
```

### 3. JSON config (for automation / CI)

```bash
uv run python -m generator.program_generator --domain agriculture --config my_config.json
```

## Environment setup

Copy `.env.example` to `.env` and fill in the keys for your LLM provider:

```bash
cp .env.example .env
```

| Variable | Provider | Where to get it |
|----------|----------|-----------------|
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Claude | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| `GROQ_API_KEY` | Groq | [console.groq.com/keys](https://console.groq.com/keys) |
| `TOGETHER_API_KEY` | Together AI | [api.together.xyz/settings/api-keys](https://api.together.xyz/settings/api-keys) |
| `OPENROUTER_API_KEY` | OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) |

**Local models (Ollama) need no API keys** — just install Ollama and pull a model:

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2
ollama serve
```

You can also set `AUTORESEARCH_MODEL` in `.env` to change the default model:

```bash
AUTORESEARCH_MODEL=ollama/llama3.2       # default
AUTORESEARCH_MODEL=gpt-4o               # use OpenAI
AUTORESEARCH_MODEL=claude-sonnet-4-20250514   # use Claude
```

## Supported LLM providers

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers). Built-in presets:

| Preset | Model | Needs |
|--------|-------|-------|
| `local` | `ollama/llama3.2` | Ollama running locally |
| `local-large` | `ollama/llama3.1:70b` | Ollama + enough RAM |
| `local-small` | `ollama/llama3.2:1b` | Ollama (fast, lightweight) |
| `local-code` | `ollama/deepseek-coder-v2` | Ollama |
| `local-mistral` | `ollama/mistral` | Ollama |
| `gpt4o` | `gpt-4o` | `OPENAI_API_KEY` |
| `gpt4o-mini` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `claude-sonnet` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| `claude-haiku` | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| `claude-opus` | `claude-opus-4-20250514` | `ANTHROPIC_API_KEY` |
| `groq-llama` | `groq/llama-3.1-70b-versatile` | `GROQ_API_KEY` |
| `together-llama` | `together_ai/meta-llama/Llama-3-70b` | `TOGETHER_API_KEY` |

Or use any LiteLLM model string directly: `--model ollama/phi3`, `--model groq/mixtral-8x7b`

```bash
# List all presets
uv run python -m generator.program_generator --list-models
```

## Training backends

### CUDA (NVIDIA GPU)

Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Uses PyTorch, Flash Attention 3, MuonAdamW optimizer.

```bash
uv pip install -e ".[cuda]"
uv run prepare.py
uv run train.py
```

### MLX (Apple Silicon)

Based on [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx). Runs natively on Mac with unified memory, no PyTorch required.

```bash
uv pip install -e ".[mlx]"
uv run prepare_mlx.py
uv run train_mlx.py
```

## Domains

Pre-configured domains with tailored questions, constraints, and model recommendations:

| Domain | File | Key focus |
|--------|------|-----------|
| Agriculture | `domains/agriculture.yaml` | Crop yield, soil analysis, interpretability |
| Healthcare | `domains/healthcare.yaml` | Clinical prediction, explainability, temporal validation |
| Social Science | `domains/social_science.yaml` | Behavioral research, fairness constraints |
| Finance | `domains/finance.yaml` | Risk modeling, look-ahead bias prevention |

Add your own domain by creating a YAML file in `domains/`. See existing files for the schema.

## Project structure

```
autoresearch-lab/
├── .env.example                     ← API key template
├── pyproject.toml                   ← dependencies & project config
├── train.py / prepare.py            ← CUDA backend (language modeling demo)
├── train_mlx.py / prepare_mlx.py    ← MLX backend (Apple Silicon)
├── program.md                       ← experiment protocol
├── generator/
│   ├── scaffold.py                  ← experiment scaffolder (data → full experiment dir)
│   ├── auto_detect.py               ← auto-detect target/task/metric from data
│   ├── program_generator.py         ← unified CLI (template + LLM modes)
│   ├── llm_generator.py             ← LLM conversation engine
│   ├── llm_client.py                ← provider-agnostic client (LiteLLM)
│   ├── domain_questions.py          ← domain Q&A loader
│   └── templates.py                 ← Jinja2 template renderer
├── domains/                         ← domain YAML configs
│   ├── agriculture.yaml
│   ├── social_science.yaml
│   ├── healthcare.yaml
│   └── finance.yaml
└── examples/                        ← example generated programs
    ├── agriculture_program.md
    └── social_behavior_program.md
```

## Install

```bash
# Base (template generator only — no ML deps)
uv sync

# With LLM-powered generation
uv pip install -e ".[llm]"

# With Apple Silicon training
uv pip install -e ".[mlx]"

# With NVIDIA GPU training
uv pip install -e ".[cuda]"

# Everything
uv pip install -e ".[llm,mlx]"      # macOS
uv pip install -e ".[llm,cuda]"     # Linux/NVIDIA
```

## Roadmap

- [x] Dual backend (CUDA + MLX)
- [x] Program generator (template + LLM-powered)
- [x] Multi-provider LLM support (OpenAI, Claude, Ollama, Groq, Together, etc.)
- [x] Domain configs (agriculture, healthcare, social science, finance)
- [x] Experiment scaffold (auto-generate prepare.py + train.py from data)
- [x] Auto-detect target/task/metric from data (heuristic + LLM)
- [x] Universal train.py (one file works for regression and classification)
- [x] Flow diagram generation (Excalidraw per experiment)
- [x] Time budget suggestion (based on baseline training time)
- [ ] Experiment memory (persist learnings across runs via mem0)
- [ ] Multi-agent research teams (design + execution + analysis agents)
- [ ] Automated research report generation

## References

Source repositories this project builds on:

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — original CUDA implementation by Andrej Karpathy. Single-GPU autonomous research with GPT model, MuonAdamW optimizer, Flash Attention 3.
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — Apple Silicon (MLX) port. Runs natively on Mac with unified memory, no PyTorch or CUDA required.

Tools and libraries:

- [LiteLLM](https://github.com/BerriAI/litellm) — unified LLM provider interface (OpenAI, Anthropic, Ollama, Groq, Together, 100+ providers)
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [Ollama](https://ollama.com) — run LLMs locally on your machine
