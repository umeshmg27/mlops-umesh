# Heart Disease MLOps Pipeline

End-to-end MLOps assignment project for predicting heart disease risk from the UCI Heart Disease
dataset. The project is plain Python and runs locally with either Docker or Podman.

## What This Covers

- Data download, cleaning, preprocessing, and EDA visualizations.
- Logistic Regression and Random Forest training with cross-validation.
- MLflow experiment tracking with logged parameters, metrics, plots, and model artifacts.
- Reproducible scikit-learn preprocessing and model pipeline saved with `joblib`.
- FastAPI model-serving API with Swagger at `/docs`.
- Streamlit UI for prediction, model metrics, EDA figures, and service status.
- Prometheus `/metrics` endpoint and optional Grafana dashboard.
- Unit tests, linting, and GitHub Actions CI.
- Docker/Podman Compose and local Kubernetes manifests.

## Quick Start: Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/download_data.py
python scripts/run_eda.py
python scripts/train_model.py

uvicorn heart_disease_mlops.api:app --host 0.0.0.0 --port 8000
```

Open:

- API health: <http://localhost:8000/health>
- Swagger: <http://localhost:8000/docs>
- Prometheus metrics: <http://localhost:8000/metrics>

In a second terminal:

```bash
source .venv/bin/activate
streamlit run ui/streamlit_app.py --server.port 8501
```

Open the UI at <http://localhost:8501>.

## Quick Start: Docker Or Podman

Docker:

```bash
docker compose up --build
```

Podman:

```bash
podman compose up --build
```

If Podman reports `looking up compose provider failed`, install the Compose provider and run it directly:

```bash
brew install podman-compose
podman-compose -f compose.yaml up --build
```

Services:

- API: <http://localhost:8000>
- Swagger: <http://localhost:8000/docs>
- UI: <http://localhost:8501>
- MLflow: <http://localhost:5001>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000>

Stop the Compose stack:

```bash
docker compose down
podman-compose -f compose.yaml down
```

The API container runs `scripts/bootstrap.py` before serving. If the dataset, EDA figures, or model are
missing, it downloads the data, generates EDA plots, trains the model, and then starts FastAPI.

## Run Tests And CI Checks Locally

```bash
ruff check .
pytest
python scripts/download_data.py
python scripts/train_model.py --cv 3
```

## Report

- PDF report: `docs/final_report.pdf`
- Detailed architecture: `ARCHITECTURE.md`

## Kubernetes: Local Deployment

Build the image locally:

```bash
docker build -t heart-disease-mlops:latest .
```

Apply manifests to Minikube, Docker Desktop Kubernetes, Kind, or another local cluster:

```bash
kubectl apply -f k8s/
kubectl get pods,svc -n heart-disease
```

For Minikube:

```bash
minikube service heart-api -n heart-disease
minikube service heart-ui -n heart-disease
```

## Dataset

The assignment uses the UCI Heart Disease dataset. This project downloads the processed Cleveland file
from the UCI archive and converts the original target values into a binary target:

- `0`: no heart disease
- `1`: heart disease present, converted from original values `1`, `2`, `3`, or `4`

No extra dataset is required.

The repository includes the downloaded raw and cleaned CSVs so the submitted project is immediately
reviewable. The download script still checks locally first and only downloads again if the file is missing
or `--force` is used.

## Included Submission Evidence

- Downloaded dataset: `data/raw/heart.csv` and `data/processed/heart_clean.csv`
- Trained model package and metadata: `models/model.joblib`, `models/metadata.json`
- MLflow file-store runs: `mlruns/`
- EDA and model evaluation figures: `reports/figures/`
- Runtime screenshots: `reports/screenshots/`
- MLflow run summary: `docs/mlflow_runs_summary.md`

## Project Structure

```text
src/heart_disease_mlops/   Reusable Python package
scripts/                   Data, EDA, training, bootstrap, smoke-test scripts
ui/                        Streamlit visual UI
tests/                     Unit and API tests
k8s/                       Local Kubernetes manifests
monitoring/                Prometheus and Grafana config
docs/                      Final report and MLflow summary
data/                      Included raw and processed dataset files
models/                    Included trained model and metadata outputs
reports/                   Included EDA figures and screenshot evidence
```
