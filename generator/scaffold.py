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

    # Validate target
    target_missing = df[TARGET_COLUMN].isnull().sum()
    if target_missing > 0:
        print(f"WARNING: Target '{{TARGET_COLUMN}}' has {{target_missing}} missing values ({{target_missing/len(df)*100:.1f}}%). Dropping them.")
        df = df.dropna(subset=[TARGET_COLUMN])
    if df[TARGET_COLUMN].nunique() <= 1:
        raise ValueError(f"Target '{{TARGET_COLUMN}}' has only {{df[TARGET_COLUMN].nunique()}} unique value(s). Nothing to predict.")

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
    """Print summary of features and target for the agent's reference."""
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = X.select_dtypes(include=["datetime"]).columns.tolist()
    missing = X.isnull().sum()
    missing = missing[missing > 0]

    # Target distribution
    print(f"\\nTarget: {{TARGET_COLUMN}}")
    if TASK_TYPE == "classification":
        counts = y.value_counts()
        total = len(y)
        print(f"  Classes ({{len(counts)}}):")
        for cls, cnt in counts.items():
            print(f"    {{cls}}: {{cnt}} ({{cnt/total*100:.1f}}%)")
        majority_pct = counts.iloc[0] / total * 100
        if majority_pct > 80:
            print(f"  WARNING: Imbalanced — majority class is {{majority_pct:.0f}}% of data")
    else:
        print(f"  min={{y.min():.3g}}, max={{y.max():.3g}}, mean={{y.mean():.3g}}, median={{y.median():.3g}}, std={{y.std():.3g}}")
        skew = y.skew()
        if abs(skew) > 1:
            print(f"  WARNING: Skewed distribution (skew={{skew:.2f}}). Consider log-transform.")

    # Feature summary
    print(f"\\nFeatures ({{len(X.columns)}}):")
    print(f"  Numeric ({{len(numeric)}}): {{numeric[:10]}}{{\'...\' if len(numeric) > 10 else \'\'}}")
    print(f"  Categorical ({{len(categorical)}}): {{categorical[:10]}}{{\'...\' if len(categorical) > 10 else \'\'}}")
    if datetime_cols:
        print(f"  Datetime ({{len(datetime_cols)}}): {{datetime_cols}}")

    # High-cardinality warnings
    for col in categorical:
        n_unique = X[col].nunique()
        if n_unique > 50:
            print(f"  WARNING: '{{col}}' has {{n_unique}} unique values — consider target encoding instead of one-hot")

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


def generate_train():
    return '''"""
Autoresearch training script — MODIFY THIS FILE.

This is the file the agent edits. Everything is fair game:
model choice, feature engineering, preprocessing, hyperparameters.

Reads TASK_TYPE from prepare.py to auto-select the right model.
Works for both regression and classification — no changes needed.

Usage: uv run train.py
"""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from prepare import (
    TASK_TYPE,
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
MAX_CATEGORICAL_CARDINALITY = 50  # OHE below this, ordinal above

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

# Split categoricals: low-cardinality (one-hot) vs high-cardinality (ordinal)
low_card_cats = [c for c in categorical_features if X_train[c].nunique() <= MAX_CATEGORICAL_CARDINALITY]
high_card_cats = [c for c in categorical_features if X_train[c].nunique() > MAX_CATEGORICAL_CARDINALITY]

print(f"Task: {TASK_TYPE} | Metric: {METRIC_NAME} ({METRIC_DIRECTION} is better)")
print(f"Using {len(numeric_features)} numeric + {len(low_card_cats)} low-card cat + {len(high_card_cats)} high-card cat features")

# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

transformers = []

if numeric_features:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features))

if low_card_cats:
    transformers.append(("cat_low", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), low_card_cats))

if high_card_cats:
    transformers.append(("cat_high", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ]), high_card_cats))

preprocessor = ColumnTransformer(transformers) if transformers else "passthrough"

# ---------------------------------------------------------------------------
# Model — auto-selects based on TASK_TYPE from prepare.py
# ---------------------------------------------------------------------------

if TASK_TYPE == "regression":
    estimator = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=42,
    )
else:
    estimator = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=42,
    )

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", estimator),
])

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

print(f"\\nTraining {estimator.__class__.__name__}...")
t_train_start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - t_train_start
print(f"Training completed in {training_time:.1f}s")

if training_time > TIME_BUDGET:
    print(f"WARNING: Training exceeded time budget ({training_time:.0f}s > {TIME_BUDGET}s)")

# ---------------------------------------------------------------------------
# Evaluation — train + val (to detect overfitting)
# ---------------------------------------------------------------------------

train_score = evaluate(model, X_train, y_train)
val_score = evaluate(model, X_val, y_val)
total_time = time.time() - t_start

print("---")
print(f"train_{METRIC_NAME}:   {train_score:.6f}")
print(f"val_{METRIC_NAME}:     {val_score:.6f}")
if METRIC_DIRECTION == "lower":
    gap = val_score - train_score
    overfit = gap > 0 and gap > val_score * 0.2
else:
    gap = train_score - val_score
    overfit = gap > 0 and gap > val_score * 0.2
if overfit:
    print(f"WARNING: Possible overfitting (train-val gap: {abs(gap):.4f}). Try regularization or fewer estimators.")
print(f"training_seconds: {training_time:.1f}")
print(f"total_seconds:    {total_time:.1f}")
n_total = len(numeric_features) + len(low_card_cats) + len(high_card_cats)
print(f"n_features:       {n_total}")
print(f"n_train:          {len(X_train)}")
print(f"n_val:            {len(X_val)}")
print(f"model:            {estimator.__class__.__name__}")
print(f"n_estimators:     {N_ESTIMATORS}")
print(f"max_depth:        {MAX_DEPTH}")

# ---------------------------------------------------------------------------
# Feature importance (top 15)
# ---------------------------------------------------------------------------

try:
    importances = estimator.feature_importances_
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f"f{i}" for i in range(len(importances))]
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\\nTop features:")
    for i in sorted_idx[:15]:
        name = feature_names[i] if i < len(feature_names) else f"f{i}"
        print(f"  {name:40s} {importances[i]:.4f}")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Time budget suggestion
# ---------------------------------------------------------------------------

if training_time < 1:
    suggested = 60
elif training_time < 10:
    suggested = int(training_time * 20)
elif training_time < 60:
    suggested = int(training_time * 10)
else:
    suggested = int(training_time * 5)

# Round to nice number
if suggested < 60:
    suggested = 60
elif suggested < 300:
    suggested = (suggested // 60 + 1) * 60
else:
    suggested = (suggested // 300 + 1) * 300

print()
print(f"--- Time Budget Suggestion ---")
print(f"Baseline training took {training_time:.1f}s.")
print(f"Current TIME_BUDGET in prepare.py: {TIME_BUDGET}s ({TIME_BUDGET // 60} min)")
print(f"Suggested TIME_BUDGET: {suggested}s ({suggested // 60} min)")
if training_time < 5:
    print(f"Baseline is fast. Agent can try much heavier models (deeper trees, ensembles, neural nets).")
    print(f"Consider {suggested}s-{suggested * 3}s to give room for complex experiments.")
elif training_time > TIME_BUDGET * 0.8:
    print(f"WARNING: Baseline already uses {training_time/TIME_BUDGET*100:.0f}% of the budget.")
    print(f"Increase TIME_BUDGET so the agent has room to try heavier models.")
else:
    print(f"Good headroom. Agent can try models up to ~{TIME_BUDGET // int(max(training_time, 1))}x heavier.")
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
- Push to remote. Only commit locally.

## Git Rules

- Only commit `train.py`: `git add train.py && git commit -m "experiment: <description>"`
- NEVER `git add -A` or `git add .` — only stage train.py.
- NEVER `git push` — all work stays local.
- Do not commit logs, results, or data files.

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


def generate_excalidraw(data_filename, target_column, task_type, metric_key, time_budget, experiment_name):
    """Generate an Excalidraw diagram showing the experiment flow."""
    import json

    metric = METRICS[metric_key]
    direction = metric["direction"]
    metric_name = metric["name"]
    direction_text = "lower is better" if direction == "lower" else "higher is better"
    budget_min = time_budget // 60

    def text_el(id, x, y, w, h, text, size=14, color="#374151", align="center", valign="middle", container=None):
        el = {
            "type": "text", "id": id, "x": x, "y": y, "width": w, "height": h,
            "text": text, "originalText": text,
            "fontSize": size, "fontFamily": 3, "textAlign": align, "verticalAlign": valign,
            "strokeColor": color, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid",
            "roughness": 0, "opacity": 100, "angle": 0,
            "seed": hash(id) % 100000, "version": 1, "versionNonce": hash(id + "v") % 100000,
            "isDeleted": False, "groupIds": [], "boundElements": None,
            "link": None, "locked": False, "containerId": container, "lineHeight": 1.25,
        }
        return el

    def rect_el(id, x, y, w, h, fill, stroke, bound_ids=None):
        el = {
            "type": "rectangle", "id": id, "x": x, "y": y, "width": w, "height": h,
            "strokeColor": stroke, "backgroundColor": fill,
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 0, "opacity": 100, "angle": 0,
            "seed": hash(id) % 100000, "version": 1, "versionNonce": hash(id + "v") % 100000,
            "isDeleted": False, "groupIds": [],
            "boundElements": [{"id": b, "type": t} for b, t in (bound_ids or [])],
            "link": None, "locked": False, "roundness": {"type": 3},
        }
        return el

    def ellipse_el(id, x, y, w, h, fill, stroke, bound_ids=None):
        el = {
            "type": "ellipse", "id": id, "x": x, "y": y, "width": w, "height": h,
            "strokeColor": stroke, "backgroundColor": fill,
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 0, "opacity": 100, "angle": 0,
            "seed": hash(id) % 100000, "version": 1, "versionNonce": hash(id + "v") % 100000,
            "isDeleted": False, "groupIds": [],
            "boundElements": [{"id": b, "type": t} for b, t in (bound_ids or [])],
            "link": None, "locked": False,
        }
        return el

    def diamond_el(id, x, y, w, h, fill, stroke, bound_ids=None):
        el = {
            "type": "diamond", "id": id, "x": x, "y": y, "width": w, "height": h,
            "strokeColor": stroke, "backgroundColor": fill,
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 0, "opacity": 100, "angle": 0,
            "seed": hash(id) % 100000, "version": 1, "versionNonce": hash(id + "v") % 100000,
            "isDeleted": False, "groupIds": [],
            "boundElements": [{"id": b, "type": t} for b, t in (bound_ids or [])],
            "link": None, "locked": False,
        }
        return el

    def arrow_el(id, x, y, points, stroke, start_id=None, end_id=None, style="solid"):
        el = {
            "type": "arrow", "id": id, "x": x, "y": y,
            "width": abs(points[-1][0] - points[0][0]),
            "height": abs(points[-1][1] - points[0][1]),
            "strokeColor": stroke, "backgroundColor": "transparent",
            "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": style,
            "roughness": 0, "opacity": 100, "angle": 0,
            "seed": hash(id) % 100000, "version": 1, "versionNonce": hash(id + "v") % 100000,
            "isDeleted": False, "groupIds": [], "boundElements": None,
            "link": None, "locked": False,
            "points": points,
            "startBinding": {"elementId": start_id, "focus": 0, "gap": 2} if start_id else None,
            "endBinding": {"elementId": end_id, "focus": 0, "gap": 2} if end_id else None,
            "startArrowhead": None, "endArrowhead": "arrow",
        }
        return el

    elements = []

    # --- Title ---
    elements.append(text_el("title", 180, 20, 500, 40, experiment_name, size=28, color="#1e40af", align="center", valign="top"))
    elements.append(text_el("subtitle", 180, 60, 500, 25,
        f"Task: {task_type} | Metric: {metric_name} | Target: {target_column}",
        size=14, color="#64748b", align="center", valign="top"))

    # --- Row 1: Data → prepare.py (locked) ---
    elements.append(ellipse_el("data_el", 50, 120, 160, 70, "#fed7aa", "#c2410c",
        [("data_txt", "text"), ("a1", "arrow")]))
    elements.append(text_el("data_txt", 70, 140, 120, 35,
        f"{data_filename}", size=13, color="#c2410c", container="data_el"))

    elements.append(arrow_el("a1", 212, 155, [[0, 0], [68, 0]], "#c2410c", "data_el", "prepare_el"))

    elements.append(rect_el("prepare_el", 280, 120, 180, 70, "#fee2e2", "#dc2626",
        [("prepare_txt", "text"), ("a1", "arrow"), ("a2", "arrow")]))
    elements.append(text_el("prepare_txt", 295, 130, 150, 50,
        f"prepare.py\nDO NOT MODIFY\n{metric_name} ({direction_text})",
        size=11, color="#dc2626", container="prepare_el"))

    # --- Row 1: prepare.py → train.py ---
    elements.append(arrow_el("a2", 462, 155, [[0, 0], [58, 0]], "#dc2626", "prepare_el", "train_el", style="dashed"))
    elements.append(text_el("imports_lbl", 470, 138, 40, 15, "imports", size=10, color="#64748b", valign="top"))

    elements.append(rect_el("train_el", 520, 120, 180, 70, "#a7f3d0", "#047857",
        [("train_txt", "text"), ("a2", "arrow"), ("a3", "arrow")]))
    elements.append(text_el("train_txt", 535, 130, 150, 50,
        f"train.py\nAGENT EDITS THIS\nUniversal for all data",
        size=11, color="#047857", container="train_el"))

    # --- Row 1: train.py → program.md ---
    elements.append(arrow_el("a3", 702, 155, [[0, 0], [48, 0]], "#047857", "train_el", "program_el"))

    elements.append(rect_el("program_el", 750, 120, 160, 70, "#ddd6fe", "#6d28d9",
        [("program_txt", "text"), ("a3", "arrow")]))
    elements.append(text_el("program_txt", 760, 130, 140, 50,
        f"program.md\nExperiment rules\n{budget_min} min budget",
        size=11, color="#6d28d9", container="program_el"))

    # --- Divider ---
    elements.append(text_el("loop_title", 280, 230, 400, 30,
        "Autonomous Experiment Loop", size=22, color="#1e40af", align="center", valign="top"))

    # --- Loop: Edit → Commit → Run → Evaluate → Keep/Revert → repeat ---
    # Edit
    elements.append(rect_el("loop_edit", 80, 290, 140, 60, "#a7f3d0", "#047857",
        [("loop_edit_txt", "text"), ("la1", "arrow"), ("la_keep_back", "arrow"), ("la_revert_back", "arrow")]))
    elements.append(text_el("loop_edit_txt", 90, 300, 120, 40,
        "Edit train.py\nnew idea", size=12, color="#047857", container="loop_edit"))

    # Edit → Commit
    elements.append(arrow_el("la1", 222, 320, [[0, 0], [58, 0]], "#1e3a5f", "loop_edit", "loop_commit"))

    # Commit
    elements.append(rect_el("loop_commit", 280, 290, 130, 60, "#93c5fd", "#1e3a5f",
        [("loop_commit_txt", "text"), ("la1", "arrow"), ("la2", "arrow")]))
    elements.append(text_el("loop_commit_txt", 290, 300, 110, 40,
        "git commit", size=12, color="#1e3a5f", container="loop_commit"))

    # Commit → Run
    elements.append(arrow_el("la2", 412, 320, [[0, 0], [58, 0]], "#1e3a5f", "loop_commit", "loop_run"))

    # Run
    elements.append(rect_el("loop_run", 470, 290, 140, 60, "#3b82f6", "#1e3a5f",
        [("loop_run_txt", "text"), ("la2", "arrow"), ("la3", "arrow")]))
    elements.append(text_el("loop_run_txt", 480, 300, 120, 40,
        "Run\nuv run train.py", size=12, color="#ffffff", container="loop_run"))

    # Run → Evaluate
    elements.append(arrow_el("la3", 612, 320, [[0, 0], [48, 0]], "#1e3a5f", "loop_run", "loop_eval"))

    # Evaluate (diamond)
    elements.append(diamond_el("loop_eval", 660, 280, 150, 80, "#fef3c7", "#b45309",
        [("loop_eval_txt", "text"), ("la3", "arrow"), ("la_yes", "arrow"), ("la_no", "arrow")]))
    elements.append(text_el("loop_eval_txt", 695, 307, 80, 25,
        f"val_{metric_key}\nbetter?", size=12, color="#b45309", container="loop_eval"))

    # Yes → Keep
    elements.append(arrow_el("la_yes", 735, 362, [[0, 0], [0, 48]], "#047857", "loop_eval", "loop_keep"))
    elements.append(text_el("yes_lbl", 745, 375, 25, 15, "yes", size=11, color="#047857", valign="top"))

    elements.append(rect_el("loop_keep", 665, 410, 140, 50, "#a7f3d0", "#047857",
        [("loop_keep_txt", "text"), ("la_yes", "arrow"), ("la_keep_back", "arrow")]))
    elements.append(text_el("loop_keep_txt", 680, 420, 110, 30,
        "Keep + Log", size=13, color="#047857", container="loop_keep"))

    # No → Revert
    elements.append(arrow_el("la_no", 812, 320, [[0, 0], [48, 0]], "#dc2626", "loop_eval", "loop_revert"))
    elements.append(text_el("no_lbl", 825, 305, 20, 15, "no", size=11, color="#dc2626", valign="top"))

    elements.append(rect_el("loop_revert", 860, 295, 120, 50, "#fee2e2", "#dc2626",
        [("loop_revert_txt", "text"), ("la_no", "arrow"), ("la_revert_back", "arrow")]))
    elements.append(text_el("loop_revert_txt", 870, 305, 100, 30,
        "git revert", size=13, color="#dc2626", container="loop_revert"))

    # Keep → back to Edit (loop)
    elements.append(arrow_el("la_keep_back", 665, 435,
        [[0, 0], [-515, 0], [-515, -115]],
        "#047857", "loop_keep", "loop_edit"))

    # Revert → back to Edit (loop)
    elements.append(arrow_el("la_revert_back", 920, 347,
        [[0, 0], [0, 140], [-770, 140], [-770, -27]],
        "#dc2626", "loop_revert", "loop_edit", style="dashed"))

    # Footer
    elements.append(text_el("footer", 250, 510, 450, 20,
        f"Runs autonomously — {budget_min} min per experiment — overnight",
        size=13, color="#64748b", align="center", valign="top"))

    diagram = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff", "gridSize": 20},
        "files": {},
    }
    return json.dumps(diagram, indent=2)


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

    # Validate metric matches task type
    expected_task = METRICS[metric_key]["task"]
    if expected_task != task_type:
        reg_metrics = [k for k, v in METRICS.items() if v["task"] == "regression"]
        cls_metrics = [k for k, v in METRICS.items() if v["task"] == "classification"]
        raise ValueError(
            f"Metric '{metric_key}' is for {expected_task}, but task is '{task_type}'. "
            f"Use: {reg_metrics if task_type == 'regression' else cls_metrics}"
        )

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
        f.write(generate_train())

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
        f.write(
            "# Python\n"
            "__pycache__/\n"
            "*.pyc\n"
            "*.pyo\n"
            ".venv/\n"
            "\n"
            "# Secrets\n"
            ".env\n"
            "\n"
            "# Logs & results (don't push these)\n"
            "run.log\n"
            "*.log\n"
            "results.tsv\n"
            "results/\n"
            "\n"
            "# Rendered diagrams (regenerate with excalidraw)\n"
            "*.png\n"
            "\n"
            "# OS\n"
            ".DS_Store\n"
            "\n"
            "# Build artifacts\n"
            "*.egg-info/\n"
            "dist/\n"
            "build/\n"
            "\n"
            "# uv lock (regenerate with uv sync)\n"
            "uv.lock\n"
        )

    # Generate experiment flow diagram
    data_filename = os.path.basename(data_path)
    diagram_path = os.path.join(output_dir, "flow.excalidraw")
    with open(diagram_path, "w") as f:
        f.write(generate_excalidraw(data_filename, target_column, task_type, metric_key, time_budget, experiment_name))

    return {
        "prepare": prepare_path,
        "train": train_path,
        "program": program_path,
        "pyproject": pyproject_path,
        "diagram": diagram_path,
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
  # Auto-detect everything (just provide data)
  python -m generator.scaffold --data data/crops.csv --output-dir experiments/crop-yield

  # Auto-detect with LLM (smarter analysis)
  python -m generator.scaffold --data data/crops.csv --output-dir experiments/crop-yield --llm --model gpt4o

  # Manual (specify everything)
  python -m generator.scaffold \\
      --data data/crops.csv --target yield --metric mae --task regression \\
      --output-dir experiments/crop-yield

Available metrics:
  Regression:     mae, rmse, r2
  Classification: auc, f1, accuracy
""",
    )
    parser.add_argument("--data", required=True, help="Path to dataset (CSV, Parquet, JSON, Excel)")
    parser.add_argument("--target", help="Target column name (auto-detected if omitted)")
    parser.add_argument("--metric", help="Evaluation metric (auto-detected if omitted)")
    parser.add_argument("--task", choices=["regression", "classification"], help="Task type (auto-detected if omitted)")
    parser.add_argument("--output-dir", required=True, help="Directory to generate experiment files into")
    parser.add_argument("--time-budget", type=int, default=300, help="Time budget in seconds (default: 300)")
    parser.add_argument("--name", help="Experiment name (default: output dir name)")
    parser.add_argument("--llm", action="store_true", help="Use LLM agent to analyze data")
    parser.add_argument("--model", default="local", help="LLM model for --llm mode (default: local)")
    parser.add_argument("--base-url", help="Custom API base URL for --llm mode")

    args = parser.parse_args()

    # Auto-detect missing settings
    needs_detect = not all([args.target, args.metric, args.task])
    if needs_detect:
        if not os.path.exists(args.data):
            parser.error(f"Data file '{args.data}' not found. Auto-detection requires the file to exist.")

        from generator.auto_detect import detect

        llm_model = args.model if args.llm else None
        detected = detect(args.data, model=llm_model, base_url=args.base_url)

        print(f"Auto-detected from {args.data} ({detected['source']}):")
        if not args.target:
            args.target = detected["target"]
            reason = detected["reasoning"] if isinstance(detected["reasoning"], str) else detected["reasoning"].get("target", "")
            print(f"  Target: {args.target}  ({reason})")
        if not args.task:
            args.task = detected["task"]
            reason = detected["reasoning"] if isinstance(detected["reasoning"], str) else detected["reasoning"].get("task", "")
            print(f"  Task:   {args.task}  ({reason})")
        if not args.metric:
            args.metric = detected["metric"]
            reason = detected["reasoning"] if isinstance(detected["reasoning"], str) else detected["reasoning"].get("metric", "")
            print(f"  Metric: {args.metric}  ({reason})")
        print()
    else:
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
    print(f"Experiment scaffolded in {args.output_dir}/\n")
    print(f"  prepare.py       <- data loading + eval (DO NOT MODIFY)")
    print(f"  train.py         <- baseline model (agent modifies this)")
    print(f"  program.md       <- experiment rules")
    print(f"  flow.excalidraw  <- experiment flow diagram (open in excalidraw.com)")
    print(f"  pyproject.toml   <- dependencies")
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
