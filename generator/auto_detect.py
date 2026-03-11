"""
Interactive experiment setup — chat with an LLM to configure your experiment.

Profiles your data, then has a conversation to determine target/task/metric.
Two modes:
    1. Interactive chat — LLM asks you questions, you answer
    2. Description — pass a text description, LLM figures it out

Usage:
    # Interactive (default)
    python -m generator.auto_detect --data crops.csv --model gpt4o

    # One-shot from description
    python -m generator.auto_detect --data crops.csv --model gpt4o \
        --description "I want to predict crop yield based on soil and weather data"
"""

import json
import os
import re
import sys

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Data profiling
# ---------------------------------------------------------------------------

def profile_data(path: str) -> dict:
    """Read dataset and return a profile summary."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".json", ".jsonl"):
        df = pd.read_json(path, lines=ext == ".jsonl")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    profile = {
        "path": path,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": {},
    }

    for col in df.columns:
        info = {
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "n_missing": int(df[col].isnull().sum()),
            "sample_values": df[col].dropna().head(5).tolist(),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            info["min"] = float(df[col].min())
            info["max"] = float(df[col].max())
            info["mean"] = float(df[col].mean())
        profile["columns"][col] = info

    return profile


def profile_to_text(profile: dict) -> str:
    """Convert profile dict to human/LLM-readable text."""
    lines = [
        f"Dataset: {profile['path']}",
        f"Rows: {profile['n_rows']}, Columns: {profile['n_cols']}",
        "",
        "Columns:",
    ]
    for name, info in profile["columns"].items():
        parts = [f"  {name}: {info['dtype']}, {info['n_unique']} unique"]
        if info["n_missing"] > 0:
            parts.append(f", {info['n_missing']} missing")
        if "mean" in info:
            parts.append(f", range [{info['min']:.3g}, {info['max']:.3g}], mean={info['mean']:.3g}")
        parts.append(f"\n    samples: {info['sample_values'][:3]}")
        lines.append("".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat-based detection
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a data science assistant helping a researcher set up an ML experiment.

You have access to a dataset profile. Your job is to have a SHORT conversation to determine:
1. Which column to predict (target)
2. Whether it's regression or classification
3. The best evaluation metric

Available metrics:
- Regression: mae (Mean Absolute Error), rmse (Root Mean Squared Error), r2 (R-squared)
- Classification: auc (AUC-ROC, best for binary), f1 (F1 weighted, best for multiclass), accuracy

Rules:
- Be concise. No lectures. Max 2-3 sentences per response.
- Ask at most 2 clarifying questions before making a recommendation.
- When you have enough info, output your recommendation as a JSON block like this:

```json
{"target": "column_name", "task": "regression", "metric": "mae", "reasoning": "one sentence"}
```

- The JSON block signals you're done. Only output it when you're confident.
- If the user confirms your suggestion, output the JSON block.
- If the user overrides something, adjust and output the updated JSON block.

Here is the dataset profile:

{data_summary}"""

_ONESHOT_PROMPT = """You are a data science expert. A researcher wants to run an ML experiment on this dataset.

Here is the dataset profile:

{data_summary}

The researcher says: "{description}"

Based on the data and their description, determine the experiment setup.

Respond with ONLY a JSON block (no other text):
```json
{{"target": "column_name", "task": "regression or classification", "metric": "mae/rmse/r2/auc/f1/accuracy", "reasoning": "one sentence"}}
```"""


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON config from LLM response."""
    # Try to find JSON in code block
    match = re.search(r'```(?:json)?\s*(\{[^`]*\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find bare JSON
    match = re.search(r'\{[^{}]*"target"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def _validate_result(result: dict, profile: dict) -> str | None:
    """Validate LLM result. Returns error message or None if valid."""
    if result.get("target") not in profile["columns"]:
        return f"Target '{result.get('target')}' not found in columns: {list(profile['columns'].keys())}"
    if result.get("task") not in ("regression", "classification"):
        return f"Invalid task: {result.get('task')}. Must be 'regression' or 'classification'"
    valid_metrics = {"mae", "rmse", "r2", "auc", "f1", "accuracy"}
    if result.get("metric") not in valid_metrics:
        return f"Invalid metric: {result.get('metric')}. Must be one of {valid_metrics}"
    # Check metric-task match
    reg_metrics = {"mae", "rmse", "r2"}
    cls_metrics = {"auc", "f1", "accuracy"}
    if result["task"] == "regression" and result["metric"] in cls_metrics:
        return f"Metric '{result['metric']}' is for classification, but task is 'regression'"
    if result["task"] == "classification" and result["metric"] in reg_metrics:
        return f"Metric '{result['metric']}' is for regression, but task is 'classification'"
    return None


def chat_detect(
    profile: dict,
    model: str = "gpt4o",
    base_url: str | None = None,
    description: str | None = None,
) -> dict:
    """Interactive or one-shot LLM-based experiment setup.

    Args:
        profile: Data profile from profile_data()
        model: LLM model preset or full model string
        base_url: Custom API base URL
        description: If provided, skip chat and use one-shot mode

    Returns:
        dict with: target, task, metric, reasoning, confidence, source
    """
    from generator.llm_client import LLMConfig, chat, resolve_model

    config = LLMConfig(
        model=resolve_model(model),
        temperature=0.3,
        max_tokens=1024,
        base_url=base_url,
    )

    summary = profile_to_text(profile)

    # --- One-shot mode (description provided) ---
    if description:
        prompt = _ONESHOT_PROMPT.format(data_summary=summary, description=description)
        response = chat(
            messages=[{"role": "user", "content": prompt}],
            config=config,
        )
        result = _extract_json(response)
        if not result:
            raise ValueError(f"Could not parse LLM response:\n{response}")

        error = _validate_result(result, profile)
        if error:
            raise ValueError(f"LLM suggestion invalid: {error}")

        result["confidence"] = "high"
        result["source"] = "llm"
        return result

    # --- Interactive chat mode ---
    system_msg = _SYSTEM_PROMPT.format(data_summary=summary)
    messages = [{"role": "system", "content": system_msg}]

    # Start with a greeting that shows the data
    print(f"\n  Dataset: {profile['path']}")
    print(f"  Rows: {profile['n_rows']}, Columns: {profile['n_cols']}")
    print(f"  Columns: {', '.join(profile['columns'].keys())}")
    print()

    # First LLM message — ask what they want to predict
    messages.append({
        "role": "user",
        "content": "I just loaded my dataset. Help me set up the experiment.",
    })

    for turn in range(10):  # max 10 turns
        response = chat(messages=messages, config=config)
        messages.append({"role": "assistant", "content": response})

        # Check if LLM produced a recommendation
        result = _extract_json(response)
        if result:
            error = _validate_result(result, profile)
            if error:
                # Tell LLM about the error and let it retry
                messages.append({"role": "user", "content": f"Error: {error}. Please fix."})
                continue

            # Show recommendation and ask for confirmation
            print(f"\n  AI: {response}\n")
            user_input = input("  You (enter to accept, or type to adjust): ").strip()

            if not user_input:
                # Accepted
                result["confidence"] = "high"
                result["source"] = "llm-chat"
                return result
            else:
                # User wants to adjust
                messages.append({"role": "user", "content": user_input})
                continue

        # No JSON yet — show response and get user input
        # Strip any markdown formatting for terminal display
        display = response.strip()
        print(f"\n  AI: {display}\n")
        user_input = input("  You: ").strip()

        if not user_input:
            messages.append({"role": "user", "content": "Just pick the best option."})
        else:
            messages.append({"role": "user", "content": user_input})

    raise RuntimeError("Could not determine experiment settings after 10 turns")


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

def detect(
    data_path: str,
    model: str = "gpt4o",
    base_url: str | None = None,
    description: str | None = None,
) -> dict:
    """Detect experiment settings from data via LLM chat.

    Args:
        data_path: Path to dataset file
        model: LLM model to use
        base_url: Custom API base URL
        description: One-shot description (skips interactive chat)

    Returns:
        dict with: target, task, metric, confidence, reasoning, profile
    """
    profile = profile_data(data_path)

    result = chat_detect(
        profile,
        model=model,
        base_url=base_url,
        description=description,
    )
    result["profile"] = profile
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Chat with an LLM to set up your experiment",
    )
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--model", default="gpt4o", help="LLM model (default: gpt4o)")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--description", help="One-shot: describe what you want to predict")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    result = detect(
        args.data,
        model=args.model,
        base_url=args.base_url,
        description=args.description,
    )

    if args.json:
        output = {k: v for k, v in result.items() if k != "profile"}
        print(json.dumps(output, indent=2))
    else:
        print(f"\nExperiment setup:")
        print(f"  Target:  {result['target']}")
        print(f"  Task:    {result['task']}")
        print(f"  Metric:  {result['metric']}")
        if result.get("reasoning"):
            print(f"  Reason:  {result['reasoning']}")
        print()
        print(f"To scaffold:")
        print(f"  python -m generator.scaffold \\")
        print(f"      --data {args.data} \\")
        print(f"      --target {result['target']} \\")
        print(f"      --metric {result['metric']} \\")
        print(f"      --task {result['task']} \\")
        print(f"      --output-dir experiments/my-experiment")


if __name__ == "__main__":
    main()
