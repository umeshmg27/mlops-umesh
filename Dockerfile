FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_METADATA_PATH=/app/models/metadata.json \
    MLFLOW_TRACKING_URI=file:/app/mlruns \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY ui ./ui
COPY docs ./docs
COPY data ./data
COPY models ./models
COPY reports ./reports

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["sh", "-c", "python scripts/bootstrap.py && uvicorn heart_disease_mlops.api:app --host 0.0.0.0 --port 8000"]
