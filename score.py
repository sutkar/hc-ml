import os
import duckdb
import pandas as pd
import shap
import lightgbm as lgb

# 1. Connect to the shared DuckDB warehouse
db_path = "../warehouse/hc.duckdb"
print(f"Connecting to DuckDB warehouse at {db_path}...")
conn = duckdb.connect(db_path)

# Load data from the governed semantic mart
df = conn.execute("SELECT * FROM sv_encounters").fetchdf()
conn.close()

print(f"Loaded {len(df)} records for batch scoring.")

# 2. Prepare Features
target_col = "readmitted"
drop_cols = ["encounter_id", "source_dataset", target_col]
feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols].copy()
for col in X.select_dtypes(include=['object', 'string']).columns:
    X[col] = X[col].astype('category')

y = df[target_col]

# 3. Train a model on the full dataset (or load artifact) for scoring inference
print("Training model for inference scoring...")
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model.fit(X, y)

# 4. Generate Probabilities
df["prob_readmit"] = model.predict_proba(X)[:, 1]
df["model_version"] = "lgbm_baseline_v1"
df["scored_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

# 5. Compute TreeSHAP Explanations
print("Computing TreeSHAP values for all encounters...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Handle LightGBM SHAP value dimensions (binary classification returns list or 3D array)
if isinstance(shap_values.values, list):
    s_vals = shap_values.values[1]
elif len(shap_values.values.shape) == 3:
    s_vals = shap_values.values[:, :, 1]
else:
    s_vals = shap_values.values

# 6. Extract Top 5 SHAP Drivers per Encounter
print("Extracting top 5 SHAP feature drivers per encounter...")
shap_records = []
for i, enc_id in enumerate(df["encounter_id"]):
    row_shap = s_vals[i]
    row_features = X.columns
    row_vals = X.iloc[i]
    
    # Pair feature names, shap values, and actual feature values
    feature_impacts = []
    for f_name, s_val, f_val in zip(row_features, row_shap, row_vals):
        feature_impacts.append({
            "encounter_id": enc_id,
            "feature": f_name,
            "feature_value": str(f_val),
            "shap_value": float(s_val),
            "abs_shap": abs(float(s_val))
        })
    
    # Sort by absolute impact and take top 5
    feature_impacts.sort(key=lambda x: x["abs_shap"], reverse=True)
    top_5 = feature_impacts[:5]
    
    for rank, item in enumerate(top_5, start=1):
        shap_records.append({
            "encounter_id": item["encounter_id"],
            "feature": item["feature"],
            "feature_value": item["feature_value"],
            "shap_value": item["shap_value"],
            "rank": rank
        })

shap_df = pd.DataFrame(shap_records)

# 7. Write Results Back to Shared DuckDB Warehouse
print("Writing inference results and SHAP explanations to DuckDB...")
conn = duckdb.connect(db_path)

# Create/Overwrite inference_results table
conn.execute("DROP TABLE IF EXISTS inference_results;")
conn.execute("""
    CREATE TABLE inference_results AS 
    SELECT encounter_id, model_version, scored_at, prob_readmit 
    FROM df
""")

# Create/Overwrite inference_shap table
conn.execute("DROP TABLE IF EXISTS inference_shap;")
conn.execute("CREATE TABLE inference_shap AS SELECT * FROM shap_df")

# Verify row counts
res_count = conn.execute("SELECT COUNT(*) FROM inference_results").fetchone()[0]
shap_count = conn.execute("SELECT COUNT(*) FROM inference_shap").fetchone()[0]
conn.close()

print(f"\n--- INFERENCE WRITE-BACK SUCCESS ---")
print(f"inference_results rows: {res_count}")
print(f"inference_shap rows (Top 5 per encounter): {shap_count}")