# Data architecture — V2 foundation

## Flow
Public demonstration CSV -> schema and quality checks -> privacy-minimized SQL warehouse -> aggregated department KPIs.

## Reproduce
```bash
pip install -r requirements-dev.txt
python scripts/build_warehouse.py
python -m pytest -q
```

The first warehouse backend is SQLite for a zero-cost reproducible local pipeline. PostgreSQL is a later migration target, not a deployed dependency.

## Data contract
- Required identity for source validation: EmployeeNumber (checked for duplicates, never stored).
- Target: Attrition must be exactly Yes or No.
- SQL table: workforce, limited to Department, JobRole, OverTime, JobSatisfaction, WorkLifeBalance, YearsAtCompany and left_company.
- Any missing cell currently blocks warehouse ingestion. Future work should document justified imputation and source-specific policies.
- Database file is generated locally and must not be committed.

## Limitations
- Original dataset is a static HR attrition demonstration dataset.
- Department-level aggregates are descriptive, not causal or clinical burnout measurements.
- SQL warehouse does not yet include time dimensions or historical snapshots.
