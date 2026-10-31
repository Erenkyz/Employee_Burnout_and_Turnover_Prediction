# Legacy model audit — code-level review

Source: complete notebook code cell 17, provided from the local notebook by the project author. This is a static review; the legacy notebook was not re-executed.

## Confirmed
- A stratified 80/20 train/validation split is performed before pipeline fitting.
- StandardScaler and OneHotEncoder are steps inside an imbalanced-learn pipeline, followed by SMOTEENN and class-weighted LogisticRegression. `pipeline.fit(X_train, y_train)` therefore fits preprocessing and resampling on training data, not validation data.
- Validation probabilities are used to choose a threshold maximizing precision subject to recall >= 0.60; the same validation data is used to print the classification report. This **does not provide an independent test estimate** and may be optimistic.
- The legacy model includes randomly synthesized `avg_weekly_hours`, `annual_leave_days`, `performance_score`, and derived `burnout_risk`. The last is not a validated burnout measurement. The model also includes Gender and MaritalStatus.
- The validation-derived risk table contains individual-level rows and should not be distributed or deployed for personnel decisions.

## Not established
- No direct train/validation contamination is apparent in the shown pipeline cell. Earlier `df_clean` transformations and notebook execution order were not fully audited.
- Whether legacy reported scores can be reproduced or compared with V2: feature sets, splits and thresholds differ.
- Whether SMOTEENN improves generalization: no held-out controlled comparison yet.
- Whether predicted probabilities are calibrated: no legacy calibration evaluation.

## Recommended next experiment
1. Use a fixed stratified 60/20/20 split shared by candidate models.
2. Restrict candidates to the same documented, non-synthetic feature whitelist; exclude sensitive demographics.
3. Compare logistic regression with class weighting against a carefully configured SMOTEENN pipeline using only training data; tune thresholds on validation only.
4. Evaluate once on untouched test data, including PR-AUC, ROC-AUC, F1, precision, recall, Brier score, and calibration curve. Record versions and seeds.
5. Treat the IBM HR dataset as a portfolio research demonstration, never an employment decision tool.
