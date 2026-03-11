"""
Auto-detect experiment settings from data.

Two modes:
    1. Heuristic (no LLM) — analyzes column types, names, cardinality
    2. LLM-powered — sends a data summary to an LLM for smarter analysis

Usage:
    # Auto-detect (no LLM)
    python -m generator.auto_detect --data crops.csv

    # LLM-powered analysis
    python -m generator.auto_detect --data crops.csv --llm --model local
"""

import json
import os
import re

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
# Heuristic auto-detection
# ---------------------------------------------------------------------------

# Common target column names
_TARGET_NAMES = {
    "target", "label", "class", "y", "output", "outcome",
    "diagnosis", "survived", "churned", "churn", "default",
    "price", "yield", "salary", "revenue", "score", "rating",
    "is_fraud", "fraud", "spam", "sentiment",
}

_CLASSIFICATION_HINTS = {
    "class", "label", "diagnosis", "survived", "churned", "churn",
    "default", "is_fraud", "fraud", "spam", "sentiment", "category",
    "type", "status", "outcome", "target",
}


def heuristic_detect(profile: dict) -> dict:
    """Guess target, task type, and metric from data profile.

    Returns:
        dict with keys: target, task, metric, confidence, reasoning
    """
    columns = profile["columns"]
    col_names = list(columns.keys())

    # Step 1: Find likely target column
    target = None
    target_reason = ""

    # Check for known target names
    for name in col_names:
        if name.lower() in _TARGET_NAMES:
            target = name
            target_reason = f"'{name}' matches known target column name"
            break

    # Fallback: last column (common convention)
    if target is None:
        target = col_names[-1]
        target_reason = f"'{target}' is the last column (common convention)"

    # Step 2: Determine task type
    target_info = columns[target]
    n_unique = target_info["n_unique"]

    if target_info["dtype"] == "object":
        task = "classification"
        task_reason = f"target '{target}' is categorical (dtype=object)"
    elif n_unique == 2:
        task = "classification"
        task_reason = f"target '{target}' has exactly 2 unique values (binary)"
    elif n_unique <= 10 and n_unique < profile["n_rows"] * 0.05:
        task = "classification"
        task_reason = f"target '{target}' has {n_unique} unique values (likely categorical)"
    elif target.lower() in _CLASSIFICATION_HINTS:
        task = "classification"
        task_reason = f"target name '{target}' suggests classification"
    else:
        task = "regression"
        task_reason = f"target '{target}' is numeric with {n_unique} unique values"

    # Step 3: Pick metric
    if task == "classification":
        if n_unique == 2:
            metric = "auc"
            metric_reason = "binary classification → AUC-ROC"
        else:
            metric = "f1"
            metric_reason = f"multiclass ({n_unique} classes) → F1 weighted"
    else:
        metric = "mae"
        metric_reason = "regression → MAE (robust to outliers)"

    # Confidence
    name_match = target.lower() in _TARGET_NAMES
    confidence = "high" if name_match else "medium"

    return {
        "target": target,
        "task": task,
        "metric": metric,
        "confidence": confidence,
        "reasoning": {
            "target": target_reason,
            "task": task_reason,
            "metric": metric_reason,
        },
    }


# ---------------------------------------------------------------------------
# LLM-powered detection
# ---------------------------------------------------------------------------

_LLM_PROMPT = """You are a data science expert. Analyze this dataset and determine:
1. Which column is the prediction target
2. Whether this is regression or classification
3. The best evaluation metric

{data_summary}

Respond in JSON only (no markdown, no explanation):
{{
    "target": "column_name",
    "task": "regression" or "classification",
    "metric": "mae" or "rmse" or "r2" or "auc" or "f1" or "accuracy",
    "reasoning": "one sentence explaining your choices"
}}"""


def llm_detect(profile: dict, model: str = "local", base_url: str | None = None) -> dict:
    """Use an LLM to analyze the data and detect experiment settings."""
    from generator.llm_client import LLMConfig, chat, resolve_model

    summary = profile_to_text(profile)
    prompt = _LLM_PROMPT.format(data_summary=summary)

    config = LLMConfig(
        model=resolve_model(model),
        temperature=0.1,
        max_tokens=512,
        base_url=base_url,
    )

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        config=config,
    )

    # Extract JSON from response
    try:
        # Try direct parse
        result = json.loads(response)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code block
        match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if match:
            result = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse LLM response as JSON:\n{response}")

    # Validate
    assert result["target"] in profile["columns"], \
        f"LLM suggested target '{result['target']}' not found in columns"
    assert result["task"] in ("regression", "classification"), \
        f"Invalid task: {result['task']}"
    valid_metrics = {"mae", "rmse", "r2", "auc", "f1", "accuracy"}
    assert result["metric"] in valid_metrics, \
        f"Invalid metric: {result['metric']}"

    result["confidence"] = "high"
    result["source"] = "llm"
    return result


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------

def detect(
    data_path: str,
    model: str | None = None,
    base_url: str | None = None,
) -> dict:
    """Auto-detect experiment settings from data.

    Args:
        data_path: Path to dataset file
        model: LLM model to use (None = heuristic only)
        base_url: Custom API base URL

    Returns:
        dict with: target, task, metric, confidence, reasoning
    """
    profile = profile_data(data_path)

    if model:
        try:
            result = llm_detect(profile, model=model, base_url=base_url)
            result["profile"] = profile
            return result
        except Exception as e:
            print(f"LLM detection failed ({e}), falling back to heuristics")

    result = heuristic_detect(profile)
    result["profile"] = profile
    result["source"] = "heuristic"
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-detect experiment settings from a dataset",
    )
    parser.add_argument("--data", required=True, help="Path to dataset")
    parser.add_argument("--llm", action="store_true", help="Use LLM for analysis")
    parser.add_argument("--model", default="local", help="LLM model (default: local)")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    model = args.model if args.llm else None
    result = detect(args.data, model=model, base_url=args.base_url)

    if args.json:
        output = {k: v for k, v in result.items() if k != "profile"}
        print(json.dumps(output, indent=2))
    else:
        print(f"\nDataset: {result['profile']['path']}")
        print(f"  Rows: {result['profile']['n_rows']}, Columns: {result['profile']['n_cols']}")
        print()
        print(f"Detected settings ({result['source']}, {result.get('confidence', 'unknown')} confidence):")
        print(f"  Target:  {result['target']}")
        print(f"  Task:    {result['task']}")
        print(f"  Metric:  {result['metric']}")
        print()
        if isinstance(result.get("reasoning"), dict):
            for key, val in result["reasoning"].items():
                print(f"  {key}: {val}")
        elif isinstance(result.get("reasoning"), str):
            print(f"  Reasoning: {result['reasoning']}")
        print()
        print("To scaffold:")
        print(f"  python -m generator.scaffold \\")
        print(f"      --data {args.data} \\")
        print(f"      --target {result['target']} \\")
        print(f"      --metric {result['metric']} \\")
        print(f"      --task {result['task']} \\")
        print(f"      --output-dir experiments/my-experiment")


if __name__ == "__main__":
    main()
