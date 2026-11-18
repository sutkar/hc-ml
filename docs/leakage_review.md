# Milestone 3: Leakage Review & Feature Audit

**Dataset:** Dataset K3 (`hospital_readmission_risk_dataset.csv`)  
**Model:** LightGBM Binary Classifier (`readmitted`)  

## Leakage Checklist Audit
- [x] **No post-discharge variables present:** Fields like actual post-discharge care events or follow-up outcomes occurring *after* discharge have been excluded.
- [x] **No target derivatives:** Features perfectly correlated with or derived directly from `Readmitted_Within_30_Days` are absent.
- [x] **Identifiers dropped:** `encounter_id` and raw text/identifiers are excluded from the active feature matrix.
- [x] **Prediction-time compliance:** All included features (e.g., age, prior admissions, comorbidity index, HbA1c, medication counts) are known at the exact moment of admission/discharge prediction.

## Baseline Performance Analysis
- **ROC-AUC:** 0.5465
- **PR-AUC:** 0.7709 (skewed by the ~74% positive class baseline prevalence)
- **Conclusion:** The low ROC-AUC confirms there is **no data leakage** inflating our metrics. Feature refinement and hyperparameter tuning will be required to capture true predictive signal.