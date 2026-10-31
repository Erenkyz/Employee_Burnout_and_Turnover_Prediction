# ML Evaluation — first measured V2 baseline

## Provenance
Results supplied from a local Windows / Python 3.13 run of `python scripts/evaluate_baseline.py` on the repository's public IBM HR Attrition CSV. All ten pytest checks passed in the same environment. These results are **observed**, not inferred from CI alone.

## Evaluation protocol
- Stratified 60/20/20 split: train 882, validation 294, test 294.
- Train-only imputation, scaling, and categorical encoding.
- Logistic regression with class_weight='balanced'; no SMOTEENN.
- Validation-selected F1 threshold: 0.63.
- Fixed 16-feature whitelist; excludes EmployeeNumber, Gender, MaritalStatus and unverified burnout_risk.

## Untouched test results

| Metric | Logistic baseline | Dummy prior baseline |
| --- | ---: | ---: |
| Average precision (PR-AUC summary) | 0.517591 | 0.159864 |
| ROC-AUC | 0.794125 | 0.500000 |
| Brier score (lower is better) | 0.183644 | 0.134309 |
| Precision at selected threshold | 0.428571 | 0.000000 |
| Recall at selected threshold | 0.638298 | 0.000000 |
| F1 at selected threshold | 0.512821 | 0.000000 |

## Interpretation and open risks
The model has ranking signal above the dummy baseline but worse Brier score, suggesting probability calibration needs investigation. A validation-selected F1 threshold is not a calibrated probability and does not establish business suitability.

The original README claimed precision 0.45, recall 0.64, F1 0.53, accuracy 0.82 at threshold 0.66 using Logistic Regression + SMOTEENN. These numbers are **not independently reproduced**, and the test samples, feature definitions, preprocessing and model pipeline may differ. No head-to-head conclusion is justified.

The legacy notebook must be inspected to determine whether SMOTEENN, preprocessing or threshold selection touched test data. Run `python scripts/audit_legacy_notebook.py` and manually review the reported cell source and upstream variable assignments. Static pattern matching cannot prove absence of leakage.

**Scope:** research demonstration only. Do not use these predictions for individual employment decisions.
