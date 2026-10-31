"""Aggregated workforce analytics, not individual employment decisions."""
import pandas as pd
from .data_quality import validate_hr_dataset

def attrition_summary(frame: pd.DataFrame) -> dict:
    validate_hr_dataset(frame)
    count = len(frame)
    departures = int(frame["Attrition"].eq("Yes").sum())
    return {"employees": count, "departures": departures,
            "attrition_rate": departures / count if count else 0.0}

def department_summary(frame: pd.DataFrame) -> pd.DataFrame:
    validate_hr_dataset(frame)
    data = frame.assign(left=frame["Attrition"].eq("Yes").astype(int))
    result = data.groupby("Department", dropna=False).agg(
        employees=("EmployeeNumber", "count"), departures=("left", "sum")
    ).reset_index()
    result["attrition_rate"] = result["departures"] / result["employees"]
    return result.sort_values("attrition_rate", ascending=False).reset_index(drop=True)
