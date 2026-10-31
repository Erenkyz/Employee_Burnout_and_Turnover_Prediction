"""Unified SQL-backed workforce intelligence portfolio application.

Aggregate reporting only; not suitable for individual employment decisions.
"""
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from workforce_intelligence.data_quality import validate_hr_dataset
from workforce_intelligence.sql_analytics import segment_kpis, warehouse_overview
from workforce_intelligence.warehouse import load_warehouse

st.set_page_config(
    page_title="Workforce Intelligence Platform", page_icon="📊",
    layout="wide", initial_sidebar_state="expanded",
)
st.title("Workforce Intelligence Platform")
st.caption("Data Engineering · SQL Analytics · Model Validation")
st.info("Portfolio demonstration using the public IBM HR dataset. "
        "Not validated for employment decisions or clinical burnout assessment.")

source = ROOT / "HR-Employee-Attrition.csv"
db = ROOT / "data" / "workforce.db"
if not source.is_file():
    st.error("Required HR-Employee-Attrition.csv not found.")
    st.stop()


@st.cache_data
def read_source(path: str, modified: int):
    return pd.read_csv(path)


@st.cache_resource
def prepare_db(csv_file: str, database_file: str, modified: int):
    load_warehouse(csv_file, database_file)
    return database_file


try:
    data = read_source(str(source), source.stat().st_mtime_ns)
    quality = validate_hr_dataset(data)
    if not quality.passed or quality.missing_cells:
        st.error(f"Source quality checks failed: {quality}")
        st.stop()
    database = prepare_db(str(source), str(db), source.stat().st_mtime_ns)
    overview = warehouse_overview(database)
except Exception as exc:
    st.error(f"Data pipeline failed: {exc}")
    st.stop()

with st.sidebar:
    st.header("Explore the data")
    dimension = st.selectbox(
        "Dimension", ["Department", "JobRole", "OverTime",
                      "JobSatisfaction", "WorkLifeBalance"]
    )
    minimum = st.slider("Minimum group size", 5, 50, 10)
    st.caption("Groups smaller than the threshold are not displayed.")

overview_tab, segments_tab, quality_tab, model_tab = st.tabs(
    ["Overview", "SQL Explorer", "Data Quality", "ML Evaluation"]
)

with overview_tab:
    st.subheader("Workforce overview")
    a, b, c = st.columns(3)
    a.metric("Records", f"{overview['employees']:,}")
    b.metric("Recorded departures", f"{overview['departures']:,}")
    c.metric("Attrition rate", f"{overview['attrition_rate']:.1%}")
    dept = segment_kpis(database, "Department", min_group_size=minimum)
    if not dept.empty:
        st.plotly_chart(
            px.bar(dept, x="segment", y="attrition_rate",
                   hover_data=["employees", "departures"],
                   labels={"segment": "Department", "attrition_rate": "Attrition rate"}),
            key="overview_department_chart",
            width="stretch",
        )
    st.caption("Associations are descriptive and do not establish causality.")

with segments_tab:
    st.subheader(f"SQL explorer — {dimension}")
    groups = segment_kpis(database, dimension, min_group_size=minimum)
    if groups.empty:
        st.info("No groups satisfy the minimum size.")
    else:
        st.plotly_chart(
            px.bar(groups, x="segment", y="attrition_rate",
                   hover_data=["employees", "departures"],
                   labels={"segment": dimension, "attrition_rate": "Attrition rate"}),
            key="sql_explorer_segment_chart",
            width="stretch",
        )
        st.dataframe(groups, width="stretch", hide_index=True)
        st.download_button(
            "Download aggregate CSV",
            groups.to_csv(index=False),
            file_name="aggregate_segment_metrics.csv",
            mime="text/csv",
        )

with quality_tab:
    st.subheader("Data quality")
    a, b, c = st.columns(3)
    a.metric("Source rows", f"{quality.rows:,}")
    b.metric("Source columns", f"{quality.columns:,}")
    c.metric("Missing cells", f"{quality.missing_cells:,}")
    st.success("Data quality validation passed.")
    st.caption("Validated source loaded into a local SQLite warehouse. "
               "The warehouse intentionally excludes employee identifiers.")

with model_tab:
    st.subheader("ML evaluation — documented experiments")
    st.write("A reproducible class-weighted Logistic Regression baseline and "
             "SMOTEENN comparison were evaluated on held-out data.")
    metrics_file = ROOT / "data" / "model_comparison_metrics.json"
    if metrics_file.exists():
        try:
            import json
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            rows = [
                {"model": name, **details["test_metrics"]}
                for name, details in metrics["models"].items()
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption("Local experimental results; not independently externally validated.")
        except (ValueError, KeyError, OSError) as exc:
            st.warning(f"Could not read local metrics: {exc}")
    else:
        st.info("Optional local model metrics not found. Run scripts/compare_models.py "
                "to generate the research report.")
    st.warning("Do not use model outputs to evaluate or make decisions about individual employees.")
