import pandas as pd
import pytest
from workforce_intelligence.analytics import attrition_summary, department_summary
from workforce_intelligence.data_quality import validate_hr_dataset

@pytest.fixture
def sample():
    return pd.DataFrame({"EmployeeNumber":[1,2,3], "Attrition":["Yes","No","No"],
                         "Department":["Sales","Sales","HR"], "JobRole":["A","A","B"]})

def test_attrition_summary(sample):
    result = attrition_summary(sample)
    assert result["employees"] == 3
    assert result["departures"] == 1
    assert result["attrition_rate"] == pytest.approx(1/3)

def test_department_summary(sample):
    result = department_summary(sample).set_index("Department")
    assert result.loc["Sales","attrition_rate"] == pytest.approx(0.5)
    assert result.loc["HR","attrition_rate"] == 0

def test_duplicate_ids_fail_quality(sample):
    sample.loc[2,"EmployeeNumber"] = 1
    assert not validate_hr_dataset(sample).passed

def test_missing_columns_raise(sample):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_hr_dataset(sample.drop(columns=["Attrition"]))
