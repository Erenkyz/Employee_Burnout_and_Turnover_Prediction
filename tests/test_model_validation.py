import numpy as np
import pandas as pd
import pytest
from workforce_intelligence.model_validation import FEATURES, NUMERIC, CATEGORICAL, evaluate_baseline

def sample_data(n=100):
    rng = np.random.default_rng(17)
    data = {"EmployeeNumber": np.arange(n), "Attrition": np.where(np.arange(n) % 5 == 0, "Yes", "No")}
    for col in NUMERIC:
        data[col] = rng.integers(1, 20, size=n)
    for col in CATEGORICAL:
        data[col] = np.where(np.arange(n) % 3 == 0, "A", "B")
    return pd.DataFrame(data)

def test_reproducible_baseline_and_split():
    data = sample_data()
    first = evaluate_baseline(data)
    second = evaluate_baseline(data)
    assert first == second
    assert first["split_sizes"] == {"train": 60, "validation": 20, "test": 20}
    assert 0.1 <= first["threshold_selected_on_validation"] <= 0.9
    assert "EmployeeNumber" not in FEATURES
    assert "Gender" not in FEATURES and "MaritalStatus" not in FEATURES
    assert "burnout_risk" not in FEATURES

def test_missing_feature_rejected():
    with pytest.raises(ValueError, match="Missing model features"):
        evaluate_baseline(sample_data().drop(columns=["Age"]))

def test_invalid_target_rejected():
    data = sample_data()
    data.loc[0, "Attrition"] = "Unknown"
    with pytest.raises(ValueError, match="Invalid source data"):
        evaluate_baseline(data)
