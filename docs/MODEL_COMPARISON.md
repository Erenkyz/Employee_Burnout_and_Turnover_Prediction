# Controlled model comparison

Run from the repository root after installing `requirements-dev.txt`:

```powershell
git pull origin feature/v2-professional
.\.venv\Scripts\python.exe scripts/compare_models.py
```

Output: `data/model_comparison_metrics.json`.

Both candidates use the same fixed 60/20/20 stratified split, the same 16-feature whitelist, and train-only preprocessing. Thresholds are selected using validation F1 and measured on held-out test data. A dummy prior model is included.

**Important limitations:** This is not a reproduction of the original notebook; the features and split differ. SMOTEENN after one-hot encoding can interpolate categorical dummy columns, generating invalid synthetic categories. This experiment is exploratory; if a winner is chosen based on test results, an additional untouched evaluation is needed to estimate the selected model's generalization. Models are not suitable for individual employment decisions.

Do not commit generated row-level data or employee risk tables. Share only aggregate metrics.
