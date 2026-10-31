# ML validation gate — no performance claims until reproduced

## Audit requirements
1. Extract notebook cells and reconstruct the exact feature engineering and target definition.
2. Split before fitting encoders, scalers, imputers, selectors or any resampler. SMOTEENN must operate only on training folds inside an imbalanced-learn pipeline.
3. Use stratified train/validation/test partitions with fixed seeds. Select hyperparameters and threshold on validation folds only.
4. Compare a dummy baseline, calibrated logistic regression, and optional tree model. Record PR-AUC, ROC-AUC, recall, precision, F1, confusion matrix and Brier score on untouched test data.
5. Inspect whether engineered burnout or workload columns encode information unavailable at prediction time. Exclude unverified fields.
6. Check subgroup error rates and calibration; do not use protected characteristics for automated employment decisions.
7. Document model artifact provenance, Python dependency versions and reproducible training command.

## Current status
Existing README metrics are historical claims and have **not** been independently reproduced. Existing model artifacts remain untouched. V2 dashboard deliberately excludes individual-level predictions until these checks pass.

## V2 baseline implementation
`python scripts/evaluate_baseline.py` creates a local JSON metrics report using a 60/20/20 stratified split, train-only preprocessing, validation-only threshold selection, and an untouched test set. This is a new comparison baseline, **not** reproduction of the original notebook. The original notebook's leakage status remains unverified.
