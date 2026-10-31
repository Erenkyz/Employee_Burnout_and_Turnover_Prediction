import sqlite3
import pandas as pd
import pytest
from workforce_intelligence.warehouse import load_warehouse, department_kpis

def test_warehouse_roundtrip(tmp_path):
    source = tmp_path / "input.csv"
    database = tmp_path / "warehouse.db"
    pd.DataFrame({
        "EmployeeNumber": [1, 2, 3],
        "Department": ["Sales", "Sales", "HR"],
        "JobRole": ["Rep", "Rep", "Admin"],
        "OverTime": ["Yes", "No", "No"],
        "JobSatisfaction": [2, 3, 4],
        "WorkLifeBalance": [2, 3, 4],
        "YearsAtCompany": [1, 5, 7],
        "Attrition": ["Yes", "No", "No"]
    }).to_csv(source, index=False)
    assert load_warehouse(source, database) == 3
    result = department_kpis(database).set_index("department")
    assert result.loc["Sales", "attrition_rate"] == pytest.approx(0.5)
    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(workforce)")]
    assert "EmployeeNumber" not in columns

def test_invalid_target_rejected(tmp_path):
    source = tmp_path / "bad.csv"
    pd.DataFrame({"EmployeeNumber":[1], "Department":["HR"],
                  "JobRole":["A"], "Attrition":["Unknown"]}).to_csv(source,index=False)
    with pytest.raises(ValueError, match="Data quality"):
        load_warehouse(source, tmp_path / "bad.db")
