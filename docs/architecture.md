# Architecture

## Local Flow

```mermaid
flowchart LR
    A["UCI Heart Disease download"] --> B["Raw data: data/raw/heart.csv"]
    B --> C["Cleaning and binary target conversion"]
    C --> D["Processed data: data/processed/heart_clean.csv"]
    D --> E["EDA figures"]
    D --> F["Training pipeline"]
    F --> G["MLflow runs"]
    F --> H["Packaged model: models/model.joblib"]
    H --> I["FastAPI service"]
    I --> J["Swagger /docs"]
    I --> K["Streamlit UI"]
    I --> L["Prometheus /metrics"]
    L --> M["Grafana dashboard"]
```

## Containers

`compose.yaml` starts five services:

- `api`: bootstraps data/model if needed and serves FastAPI.
- `ui`: Streamlit visual dashboard.
- `mlflow`: experiment tracking UI.
- `prometheus`: scrapes `/metrics` from the API.
- `grafana`: displays the API dashboard.

## Model Reproducibility

The saved model is a single scikit-learn pipeline:

```text
ColumnTransformer
  numeric: median imputation + standard scaling
  categorical: most frequent imputation + one-hot encoding
Estimator
  Logistic Regression or Random Forest
```

Saving the full pipeline ensures the same transformations are applied during training and serving.

