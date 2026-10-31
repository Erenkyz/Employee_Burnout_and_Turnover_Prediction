"""Reproducible local SQL warehouse for the public demonstration dataset.

SQLite is the zero-cost development backend. PostgreSQL migration can follow.
Only approved analytical columns are persisted; employee identifiers are excluded.
"""
from pathlib import Path
import sqlite3
import pandas as pd
from .data_quality import validate_hr_dataset

FIELDS = ("Department", "JobRole", "OverTime", "JobSatisfaction",
          "WorkLifeBalance", "YearsAtCompany", "Attrition")

def load_warehouse(csv_path: str | Path, db_path: str | Path) -> int:
    frame = pd.read_csv(csv_path)
    report = validate_hr_dataset(frame)
    if not report.passed or report.missing_cells:
        raise ValueError(f"Data quality checks failed: {report}")
    missing = sorted(set(FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing analytics columns: {missing}")
    data = frame.loc[:, FIELDS].copy()
    data["left_company"] = data.pop("Attrition").eq("Yes").astype(int)
    destination = Path(db_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as connection:
        data.to_sql("workforce", connection, if_exists="replace", index=False)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_workforce_department ON workforce (Department)")
        count = connection.execute("SELECT COUNT(*) FROM workforce").fetchone()[0]
    return int(count)

def department_kpis(db_path: str | Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(
            """SELECT Department AS department,
                      COUNT(*) AS employees,
                      SUM(left_company) AS departures,
                      ROUND(1.0 * SUM(left_company) / COUNT(*), 4) AS attrition_rate
               FROM workforce GROUP BY Department
               ORDER BY attrition_rate DESC""", connection)
