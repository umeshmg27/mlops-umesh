# Screenshot Checklist

This folder contains local runtime evidence captured after the Podman Compose stack was running.

Captured screenshots:

- `swagger_docs.png`: `http://localhost:8000/docs`
- `predict_response.png`: successful `/predict` response
- `streamlit_ui.png`: `http://localhost:8501`
- `mlflow_runs.png`: `http://localhost:5001`
- `prometheus_targets.png`: `http://localhost:9090`
- `grafana_dashboard.png`: `http://localhost:3000`
- `podman_containers_running.png`: running local Compose services

Manual browser screenshots captured from the running local stack:

- `manual_streamlit_ui.png`: Streamlit UI at `http://localhost:8501`
- `manual_mlflow_home.png`: MLflow UI at `http://localhost:5001`
- `manual_prometheus_targets.png`: Prometheus target health at `http://localhost:9090/targets`
- `manual_grafana_dashboard.png`: Grafana monitoring dashboard at `http://localhost:3000`

Useful browser pages for additional manual screenshots:

```text
http://localhost:8000/docs
http://localhost:8501
http://localhost:5001
http://localhost:9090/targets
http://localhost:3000
```

Still manual after external setup:

- `github_actions_ci.png`: successful GitHub Actions run
- `kubernetes_pods_services.png`: `kubectl get pods,svc -n heart-disease`
