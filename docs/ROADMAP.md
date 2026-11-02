# Project status and roadmap

This document tracks the current Workforce Intelligence Platform and possible next steps. It distinguishes shipped features from work that is still exploratory.

## Implemented

- [x] Data quality checks for the public IBM HR demonstration dataset
- [x] CSV-to-SQLite ETL with a limited set of analytical columns
- [x] SQL-based workforce KPIs and group-level attrition comparisons
- [x] Streamlit dashboard: Overview, SQL Explorer, Data Quality, ML Evaluation
- [x] Minimum group-size filtering and aggregate CSV export
- [x] Reproducible logistic regression baseline and SMOTEENN comparison
- [x] Saved model comparison report displayed in the dashboard
- [x] Automated tests and GitHub Actions checks
- [x] Docker image and Docker Compose setup
- [x] Public Streamlit deployment and setup documentation

## Potential next steps

These are not part of the current deployed application.

- [ ] Add scheduled ingestion and historical snapshots
- [ ] Evaluate PostgreSQL if the data volume or deployment requirements justify it
- [ ] Add monitoring for pipeline failures and data drift
- [ ] Extend aggregate reporting with documented business definitions
- [ ] Study probability calibration and categorical-aware resampling
- [ ] Validate model behavior across datasets before considering any operational use

## Scope and limitations

The project is based on a static public demonstration dataset. It is designed to show data engineering, SQL analytics, dashboard development, and experimental model evaluation. Aggregate relationships are descriptive rather than causal. The models are not validated for individual employment decisions or burnout diagnosis.

See [data architecture](DATA_ARCHITECTURE.md), [model comparison](MODEL_COMPARISON_RESULTS.md), and [technical audit](TECHNICAL_AUDIT.md) for details.
