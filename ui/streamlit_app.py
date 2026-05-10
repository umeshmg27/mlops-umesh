from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", "."))
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5001").rstrip("/")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090").rstrip("/")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000").rstrip("/")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _patient_form() -> dict:
    with st.form("prediction-form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 20, 100, 57)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda value: "Female" if value == 0 else "Male")
            cp = st.selectbox("Chest pain type", [0, 1, 2, 3, 4], index=2)
            trestbps = st.slider("Resting BP", 80, 220, 140)
            chol = st.slider("Cholesterol", 100, 600, 241)
        with col2:
            fbs = st.selectbox("Fasting sugar > 120", [0, 1])
            restecg = st.selectbox("Resting ECG", [0, 1, 2], index=1)
            thalach = st.slider("Max heart rate", 70, 220, 123)
            exang = st.selectbox("Exercise angina", [0, 1], index=1)
        with col3:
            oldpeak = st.slider("Oldpeak", 0.0, 6.5, 0.2, 0.1)
            slope = st.selectbox("Slope", [0, 1, 2, 3], index=2)
            ca = st.selectbox("Major vessels", [0, 1, 2, 3, 4])
            thal = st.selectbox("Thal", [0, 3, 6, 7], index=1)
        submitted = st.form_submit_button("Predict Risk")

    payload = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    return {"submitted": submitted, "payload": payload}


def main() -> None:
    st.set_page_config(page_title="Heart Disease MLOps", layout="wide")
    st.title("Heart Disease Risk MLOps Dashboard")

    health_col, docs_col, metrics_col = st.columns(3)
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        health_col.metric("API", "Online")
        docs_col.link_button("Swagger", f"{API_URL}/docs")
        metrics_col.link_button("Metrics", f"{API_URL}/metrics")
        st.caption(f"Serving from {API_URL}. Model available: {health.get('model_available')}")
    except requests.RequestException:
        health_col.metric("API", "Offline")
        st.warning("The API is not reachable. Start the FastAPI service first.")

    tab_predict, tab_metrics, tab_eda, tab_pipeline = st.tabs(
        ["Predict", "Model Metrics", "EDA", "Pipeline"]
    )

    with tab_predict:
        form = _patient_form()
        if form["submitted"]:
            try:
                response = requests.post(f"{API_URL}/predict", json=form["payload"], timeout=10)
                response.raise_for_status()
                result = response.json()
                risk = result["risk_probability"]
                st.subheader(result["label"])
                st.metric("Risk probability", f"{risk:.1%}")
                st.progress(min(max(risk, 0.0), 1.0))
                st.json(result)
            except requests.RequestException as exc:
                st.error(f"Prediction failed: {exc}")

    with tab_metrics:
        metadata = _load_json(ARTIFACT_ROOT / "models" / "metadata.json")
        if not metadata:
            st.info("Train the model to generate metrics.")
        else:
            st.write(f"Selected model: **{metadata.get('model_name')}**")
            rows = []
            for model_name, details in metadata.get("metrics", {}).items():
                row = {"model": model_name}
                row.update(details.get("metrics", {}))
                rows.append(row)
            st.dataframe(rows, use_container_width=True)

    with tab_eda:
        figure_dir = ARTIFACT_ROOT / "reports" / "figures"
        figures = sorted(figure_dir.rglob("*.png"))
        if not figures:
            st.info("Run `python scripts/run_eda.py` and `python scripts/train_model.py` to create figures.")
        else:
            for figure in figures:
                st.image(str(figure), caption=str(figure.relative_to(ARTIFACT_ROOT)), use_container_width=True)

    with tab_pipeline:
        st.subheader("Pipeline Flow")
        st.markdown(
            """
            `UCI download -> cleaning -> EDA -> preprocessing -> model comparison -> MLflow tracking
            -> packaged model -> FastAPI -> Streamlit UI -> Prometheus/Grafana`
            """
        )
        st.write(
            {
                "API": API_URL,
                "Swagger": f"{API_URL}/docs",
                "MLflow": MLFLOW_URL,
                "Prometheus": PROMETHEUS_URL,
                "Grafana": GRAFANA_URL,
            }
        )


if __name__ == "__main__":
    main()
