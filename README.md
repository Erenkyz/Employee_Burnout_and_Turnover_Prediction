# Workforce Intelligence Platform

A reproducible portfolio project for **Data Engineering, Data Analytics, Business Analysis and Machine Learning** using the public IBM HR attrition demonstration dataset.

> V2 is deployed as a live demonstration. The original research notebook remains available. This is a **demo**, not a validated system for real-world employment decisions. Attrition is not a clinical burnout diagnosis.

## Live demonstration

[Open Workforce Intelligence Platform](https://workforce-intelligence-erenkyz.streamlit.app) · [GitHub Actions](https://github.com/Erenkyz/Employee_Burnout_and_Turnover_Prediction/actions)

The live application provides aggregate SQL analytics, data-quality checks, and a versioned ML evaluation report. It does not make individual employee predictions.

## Quick start (Python 3.11)

```bash
git clone https://github.com/Erenkyz/Employee_Burnout_and_Turnover_Prediction.git
cd Employee_Burnout_and_Turnover_Prediction
python -m venv .venv
# Activate the environment for your operating system.
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

Build the local SQL warehouse and print department KPIs:

```bash
python scripts/build_warehouse.py
```

The generated SQLite database is stored at `data/workforce.db` and is git-ignored. No cloud account or paid service is required.

Docker deployment (Docker Desktop required):

```bash
docker compose up --build -d
```

Open http://localhost:8501. Stop with `docker compose down`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Unified Streamlit platform (overview, SQL analytics, quality and model evaluation) |\n| `reports/model_comparison_demo.json` | Versioned, measured model comparison for deployment |\n| `Dockerfile`, `compose.yaml` | Reproducible container deployment |
| `src/workforce_intelligence/data_quality.py` | Source data validation |
| `src/workforce_intelligence/analytics.py` | Aggregate workforce KPIs |
| `src/workforce_intelligence/warehouse.py` | Privacy-minimized SQLite ETL |
| `scripts/build_warehouse.py` | Reproducible warehouse build command |
| `tests/` | Unit and dataset integration tests |
| `docs/` | Audit, architecture, model validation and roadmap |
| `Employee_Burnout_and_Turnover_Prediction.ipynb` | Original research notebook |

## Original research notebook

The original notebook is retained as a record of the initial modeling work. Historical results from that notebook are not directly comparable with the current V2 evaluation because the feature sets and evaluation procedures differ. The previous employee-level prediction app and its saved artifacts have been removed from the active repository.

## Development status

See [roadmap](docs/ROADMAP.md), [technical audit](docs/TECHNICAL_AUDIT.md), [data architecture](docs/DATA_ARCHITECTURE.md) and [model validation plan](docs/MODEL_VALIDATION_PLAN.md).

Pull requests should pass the GitHub Actions quality checks before merging.

## Reproducible research baseline

```bash
python scripts/evaluate_baseline.py
```

This writes `data/baseline_metrics.json` locally. The baseline uses a fixed feature whitelist, 60/20/20 stratified splits, train-only preprocessing, validation-only threshold tuning, and untouched test evaluation. It intentionally uses class-weighted logistic regression rather than the legacy SMOTEENN pipeline. Reported metrics must be read from a completed run; no improvement is claimed.
