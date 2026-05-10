# Heart Disease MLOps Pipeline

## 1. Executive Summary

This project implements a complete end-to-end MLOps pipeline for heart disease risk prediction using the
UCI Heart Disease processed Cleveland dataset. The solution covers the full lifecycle expected in the
assignment: data acquisition, data cleaning, exploratory data analysis, feature engineering, model
training, experiment tracking, model packaging, API serving, Swagger documentation, visual UI access,
containerized local execution, monitoring, logging, CI/CD, and Kubernetes deployment manifests.

The system is intentionally designed to be reproducible. A reviewer can run the project locally with plain
Python, Docker, or Podman. The data download script first checks for a local dataset and downloads only if
the file is missing. The training script creates a complete scikit-learn pipeline that includes both
preprocessing and the classifier, which prevents training-serving mismatch during inference. The Compose
stack starts FastAPI, Streamlit, MLflow, Prometheus, and Grafana so the project can be reviewed as a
working MLOps application rather than only as isolated notebooks or scripts.

The latest model training run selected Logistic Regression as the best model by test ROC-AUC. The latest
metrics were accuracy 0.869, precision 0.812, recall 0.929, and ROC-AUC 0.966. Random Forest was also
trained and tracked for comparison. MLflow run artifacts, the trained model, model metadata, EDA figures,
and runtime evidence screenshots are included in the repository for submission review.

## 2. Problem Statement

The goal is to predict whether a patient is likely to have heart disease based on clinical attributes such
as age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG
result, maximum heart rate, exercise-induced angina, ST depression, slope, number of major vessels, and
thalassemia result.

The original UCI target is multi-class, where 0 indicates absence of heart disease and 1 through 4 indicate
different degrees of disease presence. For this assignment, the pipeline converts the target into a binary
classification problem:

- 0 means no heart disease risk detected by the original label.
- 1 means heart disease risk is present, created from original target values 1, 2, 3, and 4.

This binary framing makes the model output easier to serve through an API and easier to interpret in a
small demonstration UI. The API returns a numeric prediction, a human-readable label, a risk probability,
confidence, and the selected model name.

## 3. Dataset

The project uses the UCI Heart Disease processed Cleveland dataset. The raw dataset is stored as
`data/raw/heart.csv`, and the cleaned dataset is stored as `data/processed/heart_clean.csv`. These files
are included in the repository so the submitted project can be reviewed immediately, but the project still
contains a reproducible download script.

The download workflow is implemented in `scripts/download_data.py`. Its important behavior is that it
checks the local filesystem first. If `data/raw/heart.csv` already exists, the script uses the local copy.
If the file is missing, it downloads the dataset from UCI. This supports both reliable local execution and
repeatable CI or container startup runs.

The raw processed Cleveland file contains 303 records and 14 columns. The feature columns are:

- `age`
- `sex`
- `cp`
- `trestbps`
- `chol`
- `fbs`
- `restecg`
- `thalach`
- `exang`
- `oldpeak`
- `slope`
- `ca`
- `thal`

The target column is `target`. The target transformation is handled during data cleaning so all later
stages work with a consistent binary target.

## 4. Data Cleaning and EDA

The cleaning and EDA workflow is implemented in `scripts/run_eda.py`. The script reads the raw dataset,
normalizes missing-value markers, converts columns to numeric types, removes duplicate rows, writes the
cleaned dataset, and produces EDA figures under `reports/figures/`.

The UCI file uses question marks to represent missing values in some fields. The pipeline converts those
question marks into null values, then relies on the modeling preprocessing pipeline to impute values in a
consistent way. This is preferable to manually dropping rows because the dataset is small. Keeping rows
and imputing inside the modeling pipeline improves reproducibility and ensures the same transformation is
used during training and inference.

The EDA artifacts include:

- Class balance chart showing the distribution of the binary target.
- Missing-value chart showing where input data contains missing fields.
- Numeric histograms showing feature distributions.
- Correlation heatmap showing relationships among numeric variables and the target.

The EDA step is not only decorative. It provides a quick quality check before model training. For example,
class balance helps decide whether metrics like recall and ROC-AUC should be prioritized, and missing
value analysis confirms that imputation is necessary.

## 5. Feature Engineering and Preprocessing

Feature engineering is implemented in `src/heart_disease_mlops/features.py`. The project uses a
scikit-learn `ColumnTransformer` so numeric and categorical columns can be handled differently while still
being packaged as one reproducible pipeline.

Numeric columns are processed with:

- Median imputation for missing values.
- Standard scaling to normalize feature magnitudes.

Categorical columns are processed with:

- Most-frequent imputation for missing categories.
- One-hot encoding for model-readable categorical representation.

The important MLOps decision is that preprocessing is saved together with the model. The final artifact is
not just a classifier; it is a complete preprocessing-plus-model pipeline saved to `models/model.joblib`.
This prevents a common production issue where training transformations differ from serving
transformations. At inference time, FastAPI loads the same joblib pipeline that was produced during
training.

## 6. Model Development

The model training workflow is implemented in `scripts/train_model.py` and reusable training logic is in
`src/heart_disease_mlops/training.py`. The project trains and compares two supervised learning models:

- Logistic Regression
- Random Forest

Logistic Regression is included because it is a strong baseline for tabular clinical data and is relatively
interpretable. Random Forest is included as a non-linear ensemble model that can capture feature
interactions without requiring heavy manual feature engineering.

The evaluation process uses a stratified train/test split so that the target distribution is preserved
between training and testing. It also uses stratified cross-validation on the training data. Metrics are
computed for accuracy, precision, recall, and ROC-AUC. The final model is selected by test ROC-AUC.

The latest run selected Logistic Regression. The latest comparison was:

| Model | Accuracy | Precision | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.869 | 0.812 | 0.929 | 0.966 |
| Random Forest | 0.852 | 0.806 | 0.893 | 0.951 |

Recall is especially important in this type of health-risk screening example because missing a patient who
has risk can be more costly than flagging a patient for further review. The selected Logistic Regression
model had the strongest ROC-AUC and also had high recall on the test split.

## 7. Experiment Tracking with MLflow

MLflow is integrated into the training workflow. Each candidate model is logged as a separate MLflow run.
The repository includes the local MLflow file store under `mlruns/`, so the training history and artifacts
are part of the submitted evidence.

For each model run, MLflow logs:

- Model name.
- Cross-validation fold count.
- Test split size.
- Cross-validation accuracy, precision, recall, and ROC-AUC.
- Test accuracy, precision, recall, and ROC-AUC.
- Confusion matrix artifact.
- ROC curve artifact.
- Serialized model artifact.

The MLflow UI runs at `http://localhost:5001` in the Compose stack. Port 5001 is used on the host to avoid
common conflicts with services that occupy port 5000 on macOS. The container still serves MLflow on port
5000 internally.

A concise MLflow summary is included in `docs/mlflow_runs_summary.md`. This gives reviewers a quick view
of the selected model and the latest metrics without needing to open the full MLflow UI.

## 8. Model Packaging

The selected model is saved to `models/model.joblib`, and metadata is saved to `models/metadata.json`.
The metadata file records the selected model name, training timestamp, selection metric, dataset path,
target column, candidate model metrics, and figure paths.

Packaging the preprocessing and classifier into one joblib artifact is important because the API does not
need to recreate preprocessing manually. FastAPI loads a single artifact and sends incoming patient
records through the same transformation pipeline used during training. This design keeps inference
deterministic and easier to maintain.

The model package is intentionally small enough to include in the repository for submission. In a larger
production system, the model artifact would normally be stored in an artifact registry or object store, but
for this assignment a committed artifact makes local review easier.

## 9. FastAPI Serving and Swagger

The serving layer is implemented in `src/heart_disease_mlops/api.py` using FastAPI. FastAPI was chosen
because it provides a clean Python API framework, request validation through Pydantic, and
Swagger/OpenAPI documentation.

The API exposes these endpoints:

- `GET /health` confirms service status and model availability.
- `GET /model-info` returns metadata about the trained model and metrics.
- `POST /predict` accepts one patient record and returns one prediction.
- `POST /batch-predict` accepts multiple patient records.
- `GET /metrics` exposes Prometheus metrics.

Swagger is available at `http://localhost:8000/docs`. This allows reviewers to test the API from a browser
without writing a custom client. The request schema is defined in `src/heart_disease_mlops/schemas.py`,
so invalid inputs are rejected with clear validation errors.

The prediction response contains:

- `prediction`: numeric class 0 or 1.
- `label`: readable result such as "No heart disease risk" or "Heart disease risk".
- `risk_probability`: model probability for class 1.
- `confidence`: confidence in the predicted class.
- `model_name`: selected trained model.

## 10. Streamlit User Interface

The visual UI is implemented in `ui/streamlit_app.py`. It gives a reviewer an easier way to interact with
the model than sending raw JSON requests. The UI is served at `http://localhost:8501` when the Compose
stack is running.

The UI includes:

- Patient input controls for the model features.
- Prediction output with risk probability.
- API health/status display.
- Model metric display.
- EDA and model artifact display.
- Links to API, Swagger, MLflow, Prometheus, and Grafana.

The UI is not a marketing page. It is a practical model-review dashboard built for the assignment. It makes
the pipeline easier to demonstrate during the video walkthrough and helps show that the model is available
through both programmatic and visual interfaces.

## 11. Containerization with Docker or Podman

The project includes a `Dockerfile` and `compose.yaml`. The same Python image is used for multiple roles:
API, Streamlit UI, and MLflow. The service command determines which process runs inside each container.

The Compose stack includes:

- `api`: bootstraps missing artifacts and serves FastAPI on port 8000.
- `ui`: serves Streamlit on port 8501.
- `mlflow`: serves MLflow UI on host port 5001.
- `prometheus`: scrapes API metrics on port 9090.
- `grafana`: serves the dashboard on port 3000.

The API container runs `scripts/bootstrap.py` before starting the server. If the dataset, figures, or model
are missing, bootstrap runs the download, EDA, and training scripts before launching FastAPI. This makes the
container more robust because a clean checkout can still start the full pipeline.

Docker command:

```bash
docker compose up --build
```

Podman command:

```bash
podman compose up --build
```

If Podman does not have a Compose provider installed, the README explains the fallback:

```bash
brew install podman-compose
podman-compose -f compose.yaml up --build
```

## 12. Monitoring and Logging

The API exposes Prometheus metrics at `/metrics`. The project uses `prometheus-client` to track API and
prediction behavior. Prometheus scrapes the API service using `monitoring/prometheus.yml`, and Grafana is
provisioned with a dashboard from `monitoring/grafana/`.

The monitored signals include:

- API request count.
- Prediction count by predicted class.
- Prediction latency histogram.
- Health of the Prometheus scrape target.

Before capturing the monitoring evidence, prediction traffic was sent through the API so Prometheus and
Grafana were not empty. The metrics included `heart_predictions_total` and
`heart_prediction_latency_seconds`, confirming that inference calls were observed by the monitoring layer.

The API also logs request method, path, status code, and duration. These logs are useful when debugging
local container runs and provide basic operational visibility.

## 13. Kubernetes Deployment Assets

The `k8s/` directory contains local Kubernetes manifests for deploying the API and UI. The manifests define
a namespace, API deployment, API service, UI deployment, and UI service. They are suitable for Minikube,
Kind, Docker Desktop Kubernetes, or another local Kubernetes environment.

The basic deployment command is:

```bash
kubectl apply -f k8s/
kubectl get pods,svc -n heart-disease
```

The Kubernetes files are included as deployment artifacts. A local Kubernetes screenshot still depends on
the review machine or student machine having a Kubernetes cluster enabled. The repository contains the
manifests needed for that evidence step.

## 14. CI/CD

The project includes a GitHub Actions workflow at `.github/workflows/ci.yml`. The workflow performs the
main quality checks expected for this assignment:

- Installs dependencies.
- Runs Ruff linting.
- Runs Pytest.
- Downloads the dataset if needed.
- Runs EDA.
- Runs a training smoke test.

This workflow ensures that code formatting, tests, data acquisition, and model training are checked when
changes are pushed to GitHub. The GitHub Actions screenshot can be captured after the repository is pushed
and the workflow completes.

## 15. Local Execution Instructions

Plain Python:

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

In a second terminal:

```bash
source .venv/bin/activate
streamlit run ui/streamlit_app.py --server.port 8501
```

Docker:

```bash
docker compose up --build
```

Podman:

```bash
podman-compose -f compose.yaml up --build
```

Useful local URLs:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`
- MLflow: `http://localhost:5001`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## 16. Included Evidence

The repository includes the following submission evidence:

- Downloaded raw dataset: `data/raw/heart.csv`
- Cleaned dataset: `data/processed/heart_clean.csv`
- Trained model: `models/model.joblib`
- Model metadata: `models/metadata.json`
- MLflow run store: `mlruns/`
- MLflow summary: `docs/mlflow_runs_summary.md`
- Final report: `docs/final_report.pdf`
- EDA and model figures: `reports/figures/`
- Runtime screenshots: `reports/screenshots/`
- Kubernetes manifests: `k8s/`
- Monitoring configuration: `monitoring/`
- CI/CD workflow: `.github/workflows/ci.yml`

The evidence folder includes API health, Swagger, prediction response, Streamlit UI, MLflow, Prometheus,
Grafana, and Podman container status captured from the running local stack.

## 17. Limitations and Future Improvements

This project is suitable for the assignment scope, but there are realistic improvements that would be made
in a production setting:

- Use a larger and more recent clinical dataset before real-world deployment.
- Add model explainability such as SHAP values for individual predictions.
- Add authentication for the API and UI.
- Store models in a real model registry or object store instead of committing artifacts.
- Use a database-backed MLflow tracking server for long-term experiment history.
- Add drift monitoring for incoming feature distributions.
- Add more complete integration tests for the Compose stack and Kubernetes deployment.
- Add cloud deployment if a public API URL is required.

These limitations do not prevent the assignment from being complete. They identify the boundary between a
local MLOps assignment implementation and a production medical software system.

## 18. Conclusion

The project satisfies the assignment requirements by delivering a complete, executable MLOps pipeline in
Python. It includes data preparation, EDA, feature engineering, model development, experiment tracking,
model packaging, FastAPI serving, Swagger documentation, Streamlit visualization, Docker/Podman execution,
Prometheus/Grafana monitoring, CI/CD, Kubernetes manifests, and final documentation.

The final result can be run locally, inspected visually, tested through Swagger, monitored through
Prometheus and Grafana, and reviewed through included MLflow and report artifacts. The structure is
deliberately practical: scripts are reusable, artifacts are organized, and the README provides direct
commands for both Python and container-based execution.
