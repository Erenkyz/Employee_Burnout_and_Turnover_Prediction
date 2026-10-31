# SQL Analytics Dashboard

The portfolio now includes a SQL-backed aggregate dashboard in `dashboard_sql.py`. This is separate from the existing `dashboard_v2.py` until the new interface has been validated.

## Local setup (Windows PowerShell)

```powershell
git pull origin feature/v2-professional
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m streamlit run dashboard_sql.py
```

The dashboard builds `data/workforce.db` from `HR-Employee-Attrition.csv` using the existing validated ETL loader, then reads aggregate metrics through SQL. The user can group by department, job role, overtime, job satisfaction, or work-life balance. Groups below a configurable minimum size are suppressed.

## Next product milestones

1. Verify the new dashboard and its SQL-backed metrics locally.
2. Integrate SQL analytics into a unified multi-page dashboard.
3. Add reproducible Docker deployment and environment configuration.
4. Improve ETL observability, data freshness metadata, and CI integration.
5. Document the complete architecture and produce a concise portfolio demo.

The source is a public demonstration dataset. No individual worker scores, causal claims, or employment recommendations are supported.
