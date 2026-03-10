"""
Scaffold generator — creates a runnable experiment directory from user inputs.

Given a data path, target column, metric, and task type, generates:
    1. prepare.py  — data loading, splitting, eval function (READ-ONLY)
    2. train.py    — baseline model (agent edits this)
    3. program.md  — experiment rules

Usage:
    python -m generator.scaffold \\
        --data crop_data.csv \\
        --target yield \\
        --metric mae \\
        --task regression \\
        --output-dir experiments/agriculture

    python -m generator.scaffold \\
        --data patients.parquet \\
        --target diagnosis \\
        --metric auc \\
        --task classification \\
        --time-budget 300 \\
        --output-dir experiments/healthcare
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

METRICS = {
    "mae": {
        "name": "Mean Absolute Error",
        "direction": "lower",
        "import": "from sklearn.metrics import mean_absolute_error",
        "eval_body": "    y_pred = model.predict(X_val)\n    return mean_absolute_error(y_true, y_pred)",
        "task": "regression",
    },
    "rmse": {
        "name": "Root Mean Squared Error",
        "direction": "lower",
        "import": "from sklearn.metrics import root_mean_squared_error",
        "eval_body": "    y_pred = model.predict(X_val)\n    return root_mean_squared_error(y_true, y_pred)",
        "task": "regression",
    },
    "r2": {
        "name": "R-squared",
        "direction": "higher",
        "import": "from sklearn.metrics import r2_score",
        "eval_body": "    y_pred = model.predict(X_val)\n    return r2_score(y_true, y_pred)",
        "task": "regression",
    },
    "auc": {
        "name": "AUC-ROC",
        "direction": "higher",
        "import": "from sklearn.metrics import roc_auc_score, f1_score",
        "eval_body": (
            "    if hasattr(model, 'predict_proba'):\n"
            "        y_prob = model.predict_proba(X_val)\n"
            "        if y_prob.ndim == 2 and y_prob.shape[1] == 2:\n"
            "            y_prob = y_prob[:, 1]\n"
            "        return roc_auc_score(y_true, y_prob)\n"
            "    else:\n"
            "        y_pred = model.predict(X_val)\n"
            "        return f1_score(y_true, y_pred, average='weighted')"
        ),
        "task": "classification",
    },
    "f1": {
        "name": "F1 Score",
        "direction": "higher",
        "import": "from sklearn.metrics import f1_score",
        "eval_body": "    y_pred = model.predict(X_val)\n    return f1_score(y_true, y_pred, average='weighted')",
        "task": "classification",
    },
    "accuracy": {
        "name": "Accuracy",
        "direction": "higher",
        "import": "from sklearn.metrics import accuracy_score",
        "eval_body": "    y_pred = model.predict(X_val)\n    return accuracy_score(y_true, y_pred)",
        "task": "classification",
    },
}


# ---------------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------------

def generate_prepare(data_path, target_column, task_type, metric_key, time_budget):
    metric = METRICS[metric_key]
    comparison = "new_score < old_score" if metric["direction"] == "lower" else "new_score > old_score"

    if task_type == "classification":
        split_code = (
            '    stratify = y if y.nunique() <= 50 else None\n'
            '    X_train, X_val, y_train, y_val = train_test_split(\n'
            '        X, y, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=stratify\n'
            '    )'
        )
    else:
        split_code = (
            '    X_train, X_val, y_train, y_val = train_test_split(\n'
            '        X, y, test_size=VAL_SIZE, random_state=RANDOM_SEED\n'
            '    )'
        )

    return f'''"""
Data loading and evaluation for autoresearch experiments.
DO NOT MODIFY THIS FILE — it is the fixed evaluation harness.

Dataset: {data_path}
Target:  {target_column}
Metric:  {metric["name"]} ({metric["direction"]} is better)
Task:    {task_type}
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
{metric["import"]}

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DATA_PATH = "{data_path}"
TARGET_COLUMN = "{target_column}"
TASK_TYPE = "{task_type}"
METRIC_NAME = "{metric_key}"
METRIC_DIRECTION = "{metric["direction"]}"  # "lower" or "higher"
TIME_BUDGET = {time_budget}  # seconds
RANDOM_SEED = 42
VAL_SIZE = 0.2

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load dataset and return as pandas DataFrame."""
    path = DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {{path}}. "
            f"Place your data file there or update DATA_PATH in prepare.py."
        )

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\\t")
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".json", ".jsonl"):
        df = pd.read_json(path, lines=ext == ".jsonl")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {{ext}}")

    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"Target column '{{TARGET_COLUMN}}' not found. "
            f"Available columns: {{list(df.columns)}}"
        )

    print(f"Loaded {{len(df)}} rows, {{len(df.columns)}} columns from {{path}}")
    print(f"Target: {{TARGET_COLUMN}} | Task: {{TASK_TYPE}} | Metric: {{METRIC_NAME}}")
    return df


def split_data(df):
    """Split into train/val sets. Returns (X_train, X_val, y_train, y_val)."""
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
{split_code}
    print(f"Train: {{len(X_train)}} rows | Val: {{len(X_val)}} rows")
    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(model, X_val, y_val):
    """Evaluate model on validation set. Returns the metric value."""
    y_true = y_val
{metric["eval_body"]}


def is_better(new_score, old_score):
    """Returns True if new_score is better than old_score."""
    if old_score is None:
        return True
    return {comparison}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_feature_info(df):
    """Print summary of features for the agent's reference."""
    X = df.drop(columns=[TARGET_COLUMN])
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = X.select_dtypes(include=["datetime"]).columns.tolist()
    missing = X.isnull().sum()
    missing = missing[missing > 0]

    print(f"\\nFeature summary:")
    print(f"  Numeric ({{len(numeric)}}): {{numeric[:10]}}{{\'...\' if len(numeric) > 10 else \'\'}}")
    print(f"  Categorical ({{len(categorical)}}): {{categorical[:10]}}{{\'...\' if len(categorical) > 10 else \'\'}}")
    if datetime_cols:
        print(f"  Datetime ({{len(datetime_cols)}}): {{datetime_cols}}")
    if len(missing) > 0:
        print(f"  Missing values: {{dict(missing)}}")
    else:
        print(f"  No missing values")
    print()


if __name__ == "__main__":
    df = load_data()
    get_feature_info(df)
    X_train, X_val, y_train, y_val = split_data(df)
    print("\\nData ready. Run train.py to start experiments.")
'''


def generate_train(task_type, metric_key):
    if task_type == "regression":
        model_import = "from sklearn.ensemble import GradientBoostingRegressor"
        model_class = "GradientBoostingRegressor"
    else:
        model_import = "from sklearn.ensemble import GradientBoostingClassifier"
        model_class = "GradientBoostingClassifier"

    return f'''"""
Autoresearch training script — MODIFY THIS FILE.

This is the file the agent edits. Everything is fair game:
model choice, feature engineering, preprocessing, hyperparameters.

Usage: uv run train.py
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
{model_import}

from prepare import (
    TIME_BUDGET,
    METRIC_NAME,
    METRIC_DIRECTION,
    TARGET_COLUMN,
    load_data,
    split_data,
    evaluate,
    get_feature_info,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Hyperparameters (edit these)
# ---------------------------------------------------------------------------

N_ESTIMATORS = 100
LEARNING_RATE = 0.1
MAX_DEPTH = 5

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()

df = load_data()
get_feature_info(df)
X_train, X_val, y_train, y_val = split_data(df)

# Auto-detect column types
numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"Using {{len(numeric_features)}} numeric + {{len(categorical_features)}} categorical features")

# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", {model_class}(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=42,
    )),
])

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

print(f"\\nTraining...")
t_train_start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - t_train_start
print(f"Training completed in {{training_time:.1f}}s")

if training_time > TIME_BUDGET:
    print(f"WARNING: Training exceeded time budget ({{training_time:.0f}}s > {{TIME_BUDGET}}s)")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

score = evaluate(model, X_val, y_val)
total_time = time.time() - t_start

print("---")
print(f"val_{{METRIC_NAME}}:     {{score:.6f}}")
print(f"training_seconds: {{training_time:.1f}}")
print(f"total_seconds:    {{total_time:.1f}}")
print(f"n_features:       {{len(numeric_features) + len(categorical_features)}}")
print(f"n_train:          {{len(X_train)}}")
print(f"n_val:            {{len(X_val)}}")
print(f"model:            {model_class}")
print(f"n_estimators:     {{N_ESTIMATORS}}")
print(f"max_depth:        {{MAX_DEPTH}}")
'''


def generate_program(data_path, target_column, task_type, metric_key, time_budget, experiment_name):
    metric = METRICS[metric_key]
    direction_text = "Minimize" if metric["direction"] == "lower" else "Maximize"
    return f'''# Autonomous Research — {experiment_name}

## Setup

1. Verify data exists: `{data_path}`
2. Run `uv run prepare.py` to validate data loading.
3. Initialize `results.tsv` with header row.
4. Run baseline: `uv run train.py`

## Rules

**Goal:** {direction_text} `val_{metric_key}`.

**Time budget:** {time_budget // 60} minutes per experiment.

**CAN do:**
- Modify `train.py` — everything is fair game: model, features, preprocessing, hyperparameters.
- Use any approach: sklearn, numpy, pandas. Whatever gets the best score.

**CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation and data loading.
- Install new packages beyond what's in pyproject.toml.
- Modify the `evaluate()` function.

## Data

- **File:** `{data_path}`
- **Target column:** `{target_column}`
- **Task:** {task_type}
- **Metric:** {metric["name"]} ({metric["direction"]} is better)

## Experiment Loop

LOOP FOREVER:

1. Edit `train.py` with an experimental idea
2. Commit: `git add train.py && git commit -m "experiment: <description>"`
3. Run: `uv run train.py > run.log 2>&1`
4. Check: `grep "^val_{metric_key}:" run.log`
5. If crash: `tail -n 50 run.log`, attempt fix
6. Log to `results.tsv`
7. If improved: keep. If worse: revert.

## Logging

Tab-separated `results.tsv`:

```
commit\\tval_{metric_key}\\tstatus\\tdescription
```

**NEVER STOP.** Run autonomously until manually interrupted.
'''


# ---------------------------------------------------------------------------
# Scaffold generator
# ---------------------------------------------------------------------------

def scaffold(
    data_path: str,
    target_column: str,
    metric_key: str,
    task_type: str,
    output_dir: str,
    time_budget: int = 300,
    experiment_name: str | None = None,
):
    """Generate prepare.py, train.py, and program.md for a domain experiment."""

    if metric_key not in METRICS:
        available = ", ".join(METRICS.keys())
        raise ValueError(f"Unknown metric '{metric_key}'. Available: {available}")

    if not experiment_name:
        experiment_name = os.path.basename(output_dir) or "experiment"

    # Resolve data path relative to output dir
    abs_data = os.path.abspath(data_path)
    abs_out = os.path.abspath(output_dir)
    try:
        rel_data = os.path.relpath(abs_data, abs_out)
    except ValueError:
        rel_data = abs_data

    os.makedirs(output_dir, exist_ok=True)

    # Generate prepare.py
    prepare_path = os.path.join(output_dir, "prepare.py")
    with open(prepare_path, "w") as f:
        f.write(generate_prepare(rel_data, target_column, task_type, metric_key, time_budget))

    # Generate train.py
    train_path = os.path.join(output_dir, "train.py")
    with open(train_path, "w") as f:
        f.write(generate_train(task_type, metric_key))

    # Generate program.md
    program_path = os.path.join(output_dir, "program.md")
    with open(program_path, "w") as f:
        f.write(generate_program(rel_data, target_column, task_type, metric_key, time_budget, experiment_name))

    # Generate pyproject.toml
    pyproject_path = os.path.join(output_dir, "pyproject.toml")
    with open(pyproject_path, "w") as f:
        f.write(f'''[project]
name = "{experiment_name}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pandas",
    "numpy",
    "scikit-learn",
    "pyarrow",
    "openpyxl",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
''')

    # Generate .gitignore
    with open(os.path.join(output_dir, ".gitignore"), "w") as f:
        f.write("__pycache__/\n*.pyc\n.venv/\n.env\nrun.log\n*.log\n")

    return {
        "prepare": prepare_path,
        "train": train_path,
        "program": program_path,
        "pyproject": pyproject_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scaffold a new autoresearch experiment from your data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Regression
  python -m generator.scaffold \\
      --data data/crops.csv --target yield --metric mae --task regression \\
      --output-dir experiments/crop-yield

  # Classification
  python -m generator.scaffold \\
      --data data/patients.parquet --target diagnosis --metric auc --task classification \\
      --output-dir experiments/diagnosis

  # With custom time budget
  python -m generator.scaffold \\
      --data data/prices.csv --target price --metric rmse --task regression \\
      --time-budget 600 --output-dir experiments/pricing

Available metrics:
  Regression:     mae, rmse, r2
  Classification: auc, f1, accuracy
""",
    )
    parser.add_argument("--data", required=True, help="Path to dataset (CSV, Parquet, JSON, Excel)")
    parser.add_argument("--target", required=True, help="Target column name to predict")
    parser.add_argument("--metric", required=True, help="Evaluation metric (mae, rmse, r2, auc, f1, accuracy)")
    parser.add_argument("--task", required=True, choices=["regression", "classification"], help="Task type")
    parser.add_argument("--output-dir", required=True, help="Directory to generate experiment files into")
    parser.add_argument("--time-budget", type=int, default=300, help="Time budget in seconds (default: 300)")
    parser.add_argument("--name", help="Experiment name (default: output dir name)")

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Warning: Data file '{args.data}' not found. Files will be generated anyway.")
        print(f"Make sure the file exists before running the experiment.\n")

    scaffold(
        data_path=args.data,
        target_column=args.target,
        metric_key=args.metric,
        task_type=args.task,
        output_dir=args.output_dir,
        time_budget=args.time_budget,
        experiment_name=args.name,
    )

    metric = METRICS[args.metric]
    print(f"\nExperiment scaffolded in {args.output_dir}/\n")
    print(f"  prepare.py     <- data loading + eval (DO NOT MODIFY)")
    print(f"  train.py       <- baseline model (agent modifies this)")
    print(f"  program.md     <- experiment rules")
    print(f"  pyproject.toml <- dependencies")
    print()
    print(f"Next steps:")
    print(f"  cd {args.output_dir}")
    print(f"  uv sync")
    print(f"  uv run prepare.py          # validate data loads correctly")
    print(f"  uv run train.py            # run baseline experiment")
    print()
    print(f"  Metric: {metric['name']} ({metric['direction']} is better)")
    print(f"  Target: {args.target}")
    print(f"  Budget: {args.time_budget}s per experiment")


if __name__ == "__main__":
    main()
