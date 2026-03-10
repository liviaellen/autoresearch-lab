# Research Objective

Predict crop yield using environmental and soil features.

# Dataset

Agricultural dataset with crop, environmental, and soil features.

**Location:** `crop_data.csv`

# Evaluation Metric

**Primary metric:** MAE
**Direction:** lower is better

# Experiment Strategy

Start with tree-based models (Random Forest, XGBoost).
Then test gradient boosting variants.
Finally explore neural network approaches if time permits.
Focus on feature engineering with domain knowledge.

**Model families to explore:**
- Random Forest
- XGBoost / LightGBM
- Gradient Boosting
- Neural Networks (if time permits)

# Allowed Modifications

- `train.py`
- `train_mlx.py`

# Forbidden Actions

- Installing new packages
- Modifying evaluation pipeline
- Modifying data loading
- Removing validation split

# Constraints

- Models must provide feature importance when interpretability is required
- Preserve seasonal patterns in time-series data
- Handle missing environmental sensor readings gracefully

# Runtime

**Time budget per experiment:** 5 minutes

# Reporting

Log results to results.tsv (tab-separated).
Include feature importance rankings for kept experiments.
