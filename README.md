# autoresearch-lab

General-purpose autonomous research engine. Extends [karpathy/autoresearch](https://github.com/karpathy/autoresearch) with an experiment scaffold and Apple Silicon support.

An AI agent runs experiments autonomously — modifying code, training, evaluating, keeping improvements, discarding failures — in a loop, while you sleep.

## The idea

The original [autoresearch](https://github.com/karpathy/autoresearch) proved that an AI agent can do meaningful ML research autonomously on a single GPU. But it's locked to one task (language model pretraining on text).

**autoresearch-lab** extracts the pattern and makes it reproducible for any tabular dataset:

```
The pattern:

    1. Point at your data            → auto-detects target/task/metric
    2. Scaffold the experiment       → prepare.py + train.py + program.md
    3. Run the baseline              → prints score, overfitting, top features
    4. Let the agent experiment      → train.py (agent edits this)
    5. Fixed time budget             → 5 minutes per experiment
    6. Keep or discard               → results.tsv
    7. Loop forever
```

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

# With LLM-powered auto-detect (smarter analysis)
uv run python -m generator.scaffold \
    --data path/to/data.csv \
    --output-dir experiments/my-experiment \
    --llm --model gpt4o
```

Auto-detect analyzes your data and picks the target column, task type, and metric. You can override any of them.

Supported metrics: `mae`, `rmse`, `r2` (regression) | `auc`, `f1`, `accuracy` (classification)

### 2. Run the baseline

```bash
cd experiments/my-experiment
uv run python train.py
```

The baseline prints:
- **Validation score** (and train score for overfitting detection)
- **Overfitting warning** if train-val gap > 20%
- **Top 15 features** by importance
- **Time budget suggestion** based on training time

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

## What the scaffold generates

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
- **Flow diagram** — each experiment gets a customized Excalidraw diagram showing the data flow and experiment loop.

## Language model pretraining (original autoresearch)

The included root-level `train.py` / `prepare.py` are the original autoresearch scripts — a working autonomous LM pretraining setup:

```bash
cp .env.example .env
uv sync

# Apple Silicon (MLX)
uv run prepare_mlx.py && uv run train_mlx.py

# NVIDIA GPU (CUDA)
uv run prepare.py && uv run train.py
```

## Environment setup

Copy `.env.example` to `.env` and fill in API keys (only needed for `--llm` auto-detect):

```bash
cp .env.example .env
```

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Claude |
| `GROQ_API_KEY` | Groq |
| `TOGETHER_API_KEY` | Together AI |
| `OPENROUTER_API_KEY` | OpenRouter |

**Local models (Ollama) need no API keys** — just install [Ollama](https://ollama.com) and pull a model.

## Project structure

```
autoresearch-lab/
├── pyproject.toml                   ← dependencies & project config
├── train.py / prepare.py            ← CUDA backend (language modeling demo)
├── train_mlx.py / prepare_mlx.py    ← MLX backend (Apple Silicon)
├── program.md                       ← experiment protocol
├── generator/
│   ├── scaffold.py                  ← experiment scaffolder (data → full experiment dir)
│   ├── auto_detect.py               ← auto-detect target/task/metric from data
│   └── llm_client.py                ← provider-agnostic LLM client (for --llm mode)
└── experiments/                     ← scaffolded experiment directories (gitignored)
```

## Install

```bash
# Base
uv sync

# With experiment scaffolding (pandas, sklearn)
uv pip install -e ".[domain]"

# With LLM auto-detect
uv pip install -e ".[llm]"

# With Apple Silicon training
uv pip install -e ".[mlx]"

# With NVIDIA GPU training
uv pip install -e ".[cuda]"
```

## Roadmap

- [x] Dual backend (CUDA + MLX)
- [x] Experiment scaffold (auto-generate prepare.py + train.py from data)
- [x] Auto-detect target/task/metric from data (heuristic + LLM)
- [x] Universal train.py (one file works for regression and classification)
- [x] Flow diagram generation (Excalidraw per experiment)
- [x] Time budget suggestion (based on baseline training time)
- [ ] Experiment memory (persist learnings across runs via mem0)
- [ ] Multi-agent research teams (design + execution + analysis agents)
- [ ] Automated research report generation

## References

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — original CUDA implementation by Andrej Karpathy
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) — Apple Silicon (MLX) port
- [LiteLLM](https://github.com/BerriAI/litellm) — unified LLM provider interface
- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
