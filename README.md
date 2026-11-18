### B. `hc-ml/README.md`
```markdown
# hc-ml: Machine Learning & TreeSHAP Inference Engine

## Purpose
Trains the LightGBM baseline readmission model, enforces the mandatory leakage review, and computes batch inference scores alongside TreeSHAP explainability drivers[cite: 1].

## Setup & Installation (Windows PowerShell)
1. Open PowerShell and navigate to `hc-ml`.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1