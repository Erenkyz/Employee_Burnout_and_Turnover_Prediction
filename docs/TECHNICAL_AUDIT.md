# V2 Professional — Initial Technical Audit

## Confirmed observations
- Root contains a notebook, IBM HR attrition CSV, Streamlit script, serialized model artifacts and a README.
- README instructs users to run `streamlit run app.py` and install `requirements.txt`, but the original files are named `Project_app.py` and `Requirements.txt.txt`.
- App fills many model inputs with fixed defaults and exposes a manually entered `burnout_risk` slider.
- App renders heuristic explanations as if they were feature-level model explanations; these must be replaced with validated explanations.
- README reports Logistic Regression + SMOTEENN, threshold 0.66, test recall 0.64, precision 0.45 and accuracy 0.82. These are reported metrics, not independently reproduced in this audit.
- There was no separate source package, automated tests, or CI configuration in the original root listing.

## Open verification items
1. Inspect notebook training sequence for train/test leakage, particularly resampling and preprocessing.
2. Reproduce the held-out evaluation and confirm threshold selection does not use test labels.
3. Evaluate calibration and PR-AUC, with uncertainty and subgroup checks.
4. Establish provenance for engineered workload/burnout fields; avoid interpreting synthetic scores as clinical burnout.
5. Validate artifact compatibility before replacing any deployed model.

## Design and governance
Use this dataset for portfolio demonstration and aggregate workforce analytics. Do not present employee-level scores as validated employment decisions. Avoid sensitive-attribute-based individual recommendations. Protect any future real HR data with access control, minimization and appropriate legal review.

## First implementation
Introduced a small testable analytics package, data quality checks, pinned-range dependencies and CI. Original notebook, Streamlit app and model artifacts remain unchanged.
