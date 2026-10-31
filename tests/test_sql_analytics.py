import sqlite3

import pytest

from workforce_intelligence.sql_analytics import segment_kpis, warehouse_overview


@pytest.fixture
def demo_db(tmp_path):
    db = tmp_path / "test.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE workforce (Department TEXT, JobRole TEXT, OverTime TEXT, "
            "JobSatisfaction INTEGER, WorkLifeBalance INTEGER, left_company INTEGER)"
        )
        connection.executemany(
            "INSERT INTO workforce VALUES (?, ?, ?, ?, ?, ?)",
            [("Sales", "A", "Yes", 2, 3, 1)] * 4
            + [("Sales", "A", "No", 3, 3, 0)] * 6
            + [("HR", "B", "No", 4, 4, 0)] * 3,
        )
    return db


def test_overview(demo_db):
    assert warehouse_overview(demo_db) == {
        "employees": 13, "departures": 4, "attrition_rate": round(4 / 13, 4)
    }


def test_group_suppression_and_rates(demo_db):
    groups = segment_kpis(demo_db, "Department", min_group_size=5)
    assert groups["segment"].tolist() == ["Sales"]
    assert groups["employees"].tolist() == [10]
    assert groups["attrition_rate"].tolist() == [0.4]


def test_invalid_dimension_and_group_size(demo_db):
    with pytest.raises(ValueError, match="Unsupported dimension"):
        segment_kpis(demo_db, "Department; DROP TABLE workforce")
    with pytest.raises(ValueError, match="positive integer"):
        segment_kpis(demo_db, "Department", 0)
