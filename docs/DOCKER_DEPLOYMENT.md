# Containerized local deployment

From the repository root, with Docker Desktop running:

```powershell
docker compose up --build -d
docker compose ps
```

Open http://localhost:8501. To stop:

```powershell
docker compose down
```

The image uses Python 3.11, installs production requirements, includes the demonstration CSV, and runs the unified `app.py` Streamlit application. The SQLite database is created in the container at startup and is ephemeral. No cloud account or paid service is needed.

To run automated tests outside Docker:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Security and scope: the container is for portfolio demonstration only. The included IBM HR data is a public demonstration dataset; the app displays aggregate reporting, not individual employee recommendations.
