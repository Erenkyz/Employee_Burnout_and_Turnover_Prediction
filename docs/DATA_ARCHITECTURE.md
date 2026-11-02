# Data architecture

## Pipeline

```text
HR-Employee-Attrition.csv
          |
          v
  Data quality checks
          |
          v
   CSV-to-SQLite ETL
          |
          v
  workforce SQL table
          |
          v
 Aggregate SQL queries
          |
          v
  Streamlit dashboard
```

The dashboard also reads a versioned JSON report for the ML Evaluation tab. It does not retrain models when users open the page.

## Data contract

The source validation requires `Attrition`, `EmployeeNumber`, `Department`, and `JobRole`. Employee numbers are checked for duplicates but are not copied into the SQL warehouse. The `Attrition` column accepts `Yes` and `No` only.

The `workforce` table contains `Department`, `JobRole`, `OverTime`, `JobSatisfaction`, `WorkLifeBalance`, `YearsAtCompany`, and the derived `left_company` indicator.

The ETL stops when source validation fails or missing values are found. The SQLite file is generated locally at `data/workforce.db` and is excluded from version control.

## Query layer

The application uses a fixed list of supported grouping dimensions. The selected dimension is checked against this list before it is included in SQL. Minimum group size is passed as a query parameter, and groups below that threshold are omitted.

The interface exposes aggregate metrics, not individual employee records.

## Run the pipeline

```bash
python -m pip install -r requirements-dev.txt
python scripts/build_warehouse.py
python -m pytest -q
python -m streamlit run app.py
```

SQLite is the implemented backend. PostgreSQL, scheduled ingestion, and historical snapshots are possible future extensions, not current dependencies.

## Limitations

The source is a static demonstration dataset. The pipeline does not provide ongoing ingestion or historical tracking. Segment-level attrition rates describe associations and cannot establish causes or diagnose burnout.
