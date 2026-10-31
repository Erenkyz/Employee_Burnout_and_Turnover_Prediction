# Workforce Intelligence Platform V2 (in development)

A reproducible portfolio project for **Data Engineering, Data Analytics, Business Analysis and Machine Learning** using the public IBM HR attrition demonstration dataset.

> V2 is under active development. The original notebook, model artifacts and Streamlit app remain available. This is a **demo**, not a validated system for real-world employment decisions. Attrition is not a clinical burnout diagnosis.

## Quick start (Python 3.11)

```bash
git clone https://github.com/Erenkyz/Employee_Burnout_and_Turnover_Prediction.git
cd Employee_Burnout_and_Turnover_Prediction
git checkout feature/v2-professional
python -m venv .venv
# Activate the environment for your operating system.
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run dashboard_v2.py
```

Build the local SQL warehouse and print department KPIs:

```bash
python scripts/build_warehouse.py
```

The generated SQLite database is stored at `data/workforce.db` and is git-ignored. No cloud account or paid service is required.

## Repository layout

| Path | Purpose |
| --- | --- |
| `dashboard_v2.py` | V2 aggregate analytics Streamlit app |
| `src/workforce_intelligence/data_quality.py` | Source data validation |
| `src/workforce_intelligence/analytics.py` | Aggregate workforce KPIs |
| `src/workforce_intelligence/warehouse.py` | Privacy-minimized SQLite ETL |
| `scripts/build_warehouse.py` | Reproducible warehouse build command |
| `tests/` | Unit and dataset integration tests |
| `docs/` | Audit, architecture, model validation and roadmap |
| `Project_app.py` | Original legacy prediction demo |
| `Employee_Burnout_and_Turnover_Prediction.ipynb` | Original modeling notebook |

## Legacy model results (not yet independently reproduced)

The original README reported Logistic Regression + SMOTEENN with threshold 0.66, precision 0.45, recall 0.64, and accuracy 0.82 on 294 test observations. Treat these as **historical reported results**, not independently verified V2 performance.

To launch the original demo instead:

```bash
streamlit run Project_app.py
```

The legacy app uses fixed default feature values and heuristic explanations. Its employee-level output must not be used for employment decisions.

## Development status

See [roadmap](docs/ROADMAP.md), [technical audit](docs/TECHNICAL_AUDIT.md), [data architecture](docs/DATA_ARCHITECTURE.md) and [model validation plan](docs/MODEL_VALIDATION_PLAN.md).

Pull requests should pass the GitHub Actions quality checks before merging.
