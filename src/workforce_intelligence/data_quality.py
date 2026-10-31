"""Dataset validation utilities; never mutate the input data."""
from dataclasses import dataclass
import pandas as pd

REQUIRED_COLUMNS = ("Attrition", "EmployeeNumber", "Department", "JobRole")
ALLOWED_ATTRITION = frozenset(("Yes", "No"))

@dataclass(frozen=True)
class QualityReport:
    rows: int
    columns: int
    missing_cells: int
    duplicate_employee_ids: int
    invalid_target_rows: int

    @property
    def passed(self) -> bool:
        return self.rows > 0 and self.duplicate_employee_ids == 0 and self.invalid_target_rows == 0

def validate_hr_dataset(frame: pd.DataFrame) -> QualityReport:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return QualityReport(
        rows=len(frame),
        columns=len(frame.columns),
        missing_cells=int(frame.isna().sum().sum()),
        duplicate_employee_ids=int(frame["EmployeeNumber"].duplicated().sum()),
        invalid_target_rows=int((~frame["Attrition"].isin(ALLOWED_ATTRITION)).sum()),
    )
