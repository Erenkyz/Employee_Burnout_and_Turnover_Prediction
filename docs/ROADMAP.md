# Workforce Intelligence V2 roadmap

- [x] Isolated development branch
- [x] Initial technical audit
- [x] Basic data-quality and aggregated analytics modules
- [x] Test suite and CI definition
- [ ] Reproduce baseline ML evaluation and audit leakage
- [x] Add ingestion and SQL warehouse layer (SQLite first; PostgreSQL pending)
- [x] Add SQL-based department attrition KPI (additional business metrics pending)
- [ ] Refactor Streamlit into multi-page dashboard (initial V2 dashboard created)
- [ ] Add model calibration, explanations and experiment tracking
- [ ] Add scenario analysis with explicit assumptions
- [ ] Add Docker, integration tests and end-user documentation

## Delivery 2
- [x] Add reproducible local SQL warehouse ingestion and departmental KPI query
- [x] Add warehouse round-trip and invalid-target tests
- [ ] Run CI and verify results
- [ ] Migrate warehouse backend to PostgreSQL

## Delivery 3
- [x] Initial aggregate Streamlit dashboard
- [x] Public CSV integration test
- [x] ML validation gate document
- [x] Updated README with correct commands

## Delivery 4
- [x] Add reproducible class-weighted logistic research baseline and tests
- [ ] Run baseline on full CSV and record actual metrics
- [ ] Inspect legacy notebook for leakage and reproduce original SMOTEENN evaluation
