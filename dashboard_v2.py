"""V2 aggregate HR analytics demo; not an employee decision engine."""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from workforce_intelligence.analytics import attrition_summary, department_summary
from workforce_intelligence.data_quality import validate_hr_dataset

st.set_page_config(page_title="Workforce Intelligence V2", page_icon="📊", layout="wide")
st.title("Workforce Intelligence | V2")
st.caption("Public demonstration dataset · Descriptive aggregate analytics only · Not for employment decisions")

@st.cache_data
def read_demo_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

source = ROOT / "HR-Employee-Attrition.csv"
if not source.exists():
    st.error("Demo CSV not found. Place HR-Employee-Attrition.csv in the repository root.")
    st.stop()
data = read_demo_data(str(source))
quality = validate_hr_dataset(data)
if not quality.passed:
    st.error(f"Data quality validation failed: {quality}")
    st.stop()

departments = sorted(data["Department"].dropna().unique().tolist())
selected = st.multiselect("Department filter", departments, default=departments)
if not selected:
    st.info("Select at least one department.")
    st.stop()
filtered = data.loc[data["Department"].isin(selected)]
summary = attrition_summary(filtered)
a,b,c = st.columns(3)
a.metric("Employees in selection", f"{summary['employees']:,}")
b.metric("Recorded departures", f"{summary['departures']:,}")
c.metric("Attrition rate", f"{summary['attrition_rate']:.1%}")

st.subheader("Department-level attrition")
groups = department_summary(filtered)
st.plotly_chart(px.bar(groups, x="Department", y="attrition_rate",
                       hover_data=["employees","departures"],
                       labels={"attrition_rate":"Attrition rate","Department":"Department"}),
                use_container_width=True)
st.dataframe(groups, use_container_width=True, hide_index=True)
st.caption("Historical group associations are not causal explanations. Small groups can be misleading.")
with st.expander("Data quality and limitations"):
    st.write(f"Rows: {quality.rows}; columns: {quality.columns}; missing cells: {quality.missing_cells}.")
    st.write("Static demonstration data, no real-time HR feed. No clinical burnout label is validated.")
    st.write("No individual employee scores or automated employment recommendations are displayed.")
