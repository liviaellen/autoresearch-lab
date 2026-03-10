# autoresearch-lab

General-purpose autonomous research engine extending karpathy/autoresearch.

## Setup

1. **Agree on a run tag** (e.g. `mar10`). Branch: `autoresearch/<tag>`.
2. **Choose backend**:
   - CUDA: `uv run prepare.py && uv run train.py`
   - MLX (macOS): `uv run prepare_mlx.py && uv run train_mlx.py`
3. **Read the in-scope files**:
   - `prepare.py` / `prepare_mlx.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. **Do not modify.**
   - `train.py` / `train_mlx.py` — the file you modify.
4. **Verify data**: `~/.cache/autoresearch/` must contain data shards and tokenizer.
5. **Initialize results.tsv** with header row. Run baseline first.

## Experimentation

Fixed 5-minute time budget. Launch: `uv run train.py` (CUDA) or `uv run train_mlx.py` (MLX).

**CAN do:** Modify `train.py` / `train_mlx.py` — architecture, optimizer, hyperparameters, batch size, model size.

**CANNOT do:**
- Modify `prepare.py` / `prepare_mlx.py`
- Install new packages
- Modify the evaluation harness

**Goal:** Lowest `val_bpb`.

## Output

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

## Logging

Tab-separated `results.tsv`:

```
commit	val_bpb	memory_gb	status	description
```

## Experiment Loop

LOOP FOREVER:

1. Check git state
2. Edit `train.py` or `train_mlx.py` with experimental idea
3. Commit
4. Run: `uv run train.py > run.log 2>&1` (or `train_mlx.py`)
5. Check results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If crash: `tail -n 50 run.log`, attempt fix
7. Log to `results.tsv`
8. If improved: keep commit. If worse: revert.

**NEVER STOP.** Run autonomously until manually interrupted.

## Program Generator

For domain-specific research, use the program generator:

```bash
python generator/program_generator.py --domain agriculture --config my_config.json
```

This generates a customized `program.md` from domain questions and templates.
See `generator/` for details and `domains/` for domain definitions.
