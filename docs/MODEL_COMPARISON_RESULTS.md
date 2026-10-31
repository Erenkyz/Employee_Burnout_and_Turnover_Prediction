# Controlled model comparison — observed results

Source: locally executed `scripts/compare_models.py`, supplied as `model_comparison_metrics.json`. Fixed seed 42; 882 train / 294 validation / 294 test; shared 16-feature whitelist. Validation-selected thresholds: 0.63 for both models.

| Test metric | Class-weighted logistic | SMOTEENN + logistic | Dummy prior |
|---|---:|---:|---:|
| Average precision | 0.517591 | 0.469595 | 0.159864 |
| ROC-AUC | 0.794125 | 0.759669 | 0.500000 |
| Brier (lower better) | 0.183644 | 0.206019 | 0.134309 |
| Precision | 0.428571 | 0.352273 | 0 |
| Recall | 0.638298 | 0.659574 | 0 |
| F1 | 0.512821 | 0.459259 | 0 |

## Interpretation

On this held-out split, class-weighted logistic performs better on precision, F1, average precision, ROC-AUC and Brier; SMOTEENN achieves slightly higher recall. Neither model beats the prior-only dummy on Brier, indicating that probability calibration requires further study. This is a single-split exploratory comparison, not statistical evidence of a stable winner.

The SMOTEENN implementation interpolates one-hot-encoded categorical variables, potentially producing implausible synthetic records. Future work should consider a categorical-aware alternative, repeated stratified validation, calibration on a separate validation set, and a final untouched test evaluation. The original notebook's metrics cannot be directly compared because its feature set and split differ.

Research/portfolio use only; not validated for individual employment decisions.
