# Stability and calibration diagnostic

Run from repository root:

```powershell
git pull origin feature/v2-professional
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe scripts/evaluate_stability.py
```

Output: `data/model_stability_metrics.json`. The report includes five seeded stratified 60/20/20 splits, validation-only F1 threshold selection, held-out test metrics, 10-bin ECE, and per-run calibration-curve coordinates.

The summary reports mean, sample standard deviation, minimum and maximum. These are descriptive statistics across **overlapping** test sets, not confidence intervals. ECE depends on the binning scheme and is unstable with small samples. This diagnostic does not calibrate probabilities. Selecting a model based on these test metrics would require fresh untouched evaluation.

Use only for portfolio research, not individual employment decisions.
