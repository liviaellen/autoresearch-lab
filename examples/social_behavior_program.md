# Research Objective

Predict student dropout risk from behavioral and demographic data.

# Dataset

Social science dataset with behavioral outcomes and demographic features.

**Location:** `student_data.csv`

# Evaluation Metric

**Primary metric:** AUC-ROC
**Direction:** higher is better

# Experiment Strategy

Start with logistic regression as interpretable baseline.
Then test tree-based ensemble methods.
Evaluate fairness metrics alongside performance.
Apply bias mitigation if fairness constraints are violated.

**Model families to explore:**
- Logistic Regression
- Random Forest
- XGBoost
- Fairness-aware classifiers

# Allowed Modifications

- `train.py`
- `train_mlx.py`

# Forbidden Actions

- Installing new packages
- Modifying evaluation pipeline
- Removing fairness checks
- Using protected attributes without justification

# Constraints

- Enforce fairness constraints across protected demographic groups
- Handle class imbalance appropriately
- Report disaggregated metrics by demographic group
- Avoid using protected attributes as direct features unless justified
- Demographic parity across gender and ethnicity groups

# Runtime

**Time budget per experiment:** 5 minutes

# Reporting

Log results to results.tsv (tab-separated).
Include disaggregated metrics by demographic group.
Flag any fairness constraint violations.
