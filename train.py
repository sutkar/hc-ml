import os
import json
import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

# 1. Connect to the shared DuckDB warehouse
db_path = "../warehouse/hc.duckdb"
print(f"Connecting to DuckDB warehouse at {db_path}...")
conn = duckdb.connect(db_path)

# Load data from the governed semantic mart
df = conn.execute("SELECT * FROM sv_encounters").fetchdf()
conn.close()

print(f"Loaded {len(df)} records from sv_encounters.")

# 2. Define Features and Target Label
target_col = "readmitted"
# Drop identifiers and target from feature list
drop_cols = ["encounter_id", "source_dataset", target_col]
feature_cols = [col for col in df.columns if col not in drop_cols]

# Simple imputation/cleaning for baseline modeling if necessary
X = df[feature_cols].copy()
# Convert categorical string columns to categorical codes for LightGBM
for col in X.select_dtypes(include=['object', 'string']).columns:
    X[col] = X[col].astype('category')

y = df[target_col]

# 3. Stratified Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} rows | Test set: {len(X_test)} rows")

# 4. Train a Baseline LightGBM Classifier
print("Training LightGBM baseline model...")
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

print(f"\n--- MODEL EVALUATION METRICS ---")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC:  {pr_auc:.4f}")

# 6. Save Metrics and Model Artifacts
os.makedirs("models", exist_ok=True)
metrics = {
    "model_type": "LightGBMClassifier",
    "roc_auc": round(float(roc_auc), 4),
    "pr_auc": round(float(pr_auc), 4),
    "train_rows": len(X_train),
    "test_rows": len(X_test)
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Saved evaluation metrics to models/metrics.json")