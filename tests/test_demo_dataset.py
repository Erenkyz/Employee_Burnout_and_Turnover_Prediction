"""Integration checks on the public CSV already committed to this repository."""
from pathlib import Path
import pandas as pd
from workforce_intelligence.data_quality import validate_hr_dataset
from workforce_intelligence.analytics import attrition_summary

ROOT = Path(__file__).resolve().parents[1]

def test_demo_dataset_contract():
    source = ROOT / "HR-Employee-Attrition.csv"
    assert source.exists(), "Expected public demonstration CSV"
    data = pd.read_csv(source)
    report = validate_hr_dataset(data)
    assert report.passed
    assert data["EmployeeNumber"].is_unique
    assert attrition_summary(data)["employees"] == len(data)
