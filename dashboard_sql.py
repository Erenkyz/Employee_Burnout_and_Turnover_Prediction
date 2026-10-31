"""SQL-backed aggregate analytics dashboard; local research demonstration."""
from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from workforce_intelligence.sql_analytics import segment_kpis, warehouse_overview
from workforce_intelligence.warehouse import load_warehouse

st.set_page_config(page_title="Workforce Intelligence | SQL Analytics", layout="wide")
st.title("Workforce Intelligence — SQL Analytics")
st.caption("Aggregate insights from SQLite · Public synthetic demonstration data · Not for employment decisions")

source = ROOT / "HR-Employee-Attrition.csv"
db_path = ROOT / "data" / "workforce.db"
if not source.exists():
    st.error("Dataset missing from repository root.")
    st.stop()

@st.cache_resource
def prepare_database(csv_file: str, db_file: str):
    load_warehouse(csv_file, db_file)
    return db_file

try:
    database = prepare_database(str(source), str(db_path))
    overview = warehouse_overview(database)
except Exception as exc:
    st.error(f"Could not prepare analytics warehouse: {exc}")
    st.stop()

a, b, c = st.columns(3)
a.metric("Employees", f"{overview['employees']:,}")
b.metric("Departures", f"{overview['departures']:,}")
c.metric("Attrition rate", f"{overview['attrition_rate']:.1%}")

dimension = st.selectbox(
    "Group by", ["Department", "JobRole", "OverTime", "JobSatisfaction", "WorkLifeBalance"]
)
min_size = st.slider("Minimum group size", min_value=5, max_value=50, value=10)
groups = segment_kpis(database, dimension, min_group_size=min_size)
if groups.empty:
    st.info("No groups meet the minimum group size.")
else:
    st.plotly_chart(
        px.bar(groups, x="segment", y="attrition_rate",
               hover_data=["employees", "departures"],
               labels={"segment": dimension, "attrition_rate": "Attrition rate"}),
        width="stretch",
    )
    st.dataframe(groups, width="stretch", hide_index=True)
st.caption("Groups below the threshold are hidden. Aggregate associations are not causal; "
           "this dataset is not suitable for decisions about individual workers.")
