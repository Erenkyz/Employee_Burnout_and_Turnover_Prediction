"""SQL-powered aggregate analytics for the local workforce warehouse.

No individual employee rows are returned. Small groups are suppressed in the
presentation layer using a minimum group size.
"""
from pathlib import Path
import sqlite3
import pandas as pd

ALLOWED_DIMENSIONS = {
    "Department": "Department",
    "JobRole": "JobRole",
    "OverTime": "OverTime",
    "JobSatisfaction": "JobSatisfaction",
    "WorkLifeBalance": "WorkLifeBalance",
}


def segment_kpis(db_path: str | Path, dimension: str, min_group_size: int = 10) -> pd.DataFrame:
    if dimension not in ALLOWED_DIMENSIONS:
        raise ValueError(f"Unsupported dimension: {dimension}")
    if not isinstance(min_group_size, int) or isinstance(min_group_size, bool) or min_group_size < 1:
        raise ValueError("min_group_size must be a positive integer")
    column = ALLOWED_DIMENSIONS[dimension]
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(
            f"""SELECT {column} AS segment,
                       COUNT(*) AS employees,
                       SUM(left_company) AS departures,
                       ROUND(1.0 * SUM(left_company) / COUNT(*), 4) AS attrition_rate
                FROM workforce
                WHERE {column} IS NOT NULL
                GROUP BY {column}
                HAVING COUNT(*) >= ?
                ORDER BY attrition_rate DESC, employees DESC""",
            connection,
            params=(min_group_size,),
        )


def warehouse_overview(db_path: str | Path) -> dict:
    with sqlite3.connect(db_path) as connection:
        employees, departures = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(left_company), 0) FROM workforce"
        ).fetchone()
    return {
        "employees": int(employees),
        "departures": int(departures),
        "attrition_rate": round(departures / employees, 4) if employees else 0.0,
    }
