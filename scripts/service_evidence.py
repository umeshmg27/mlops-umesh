from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
import urllib.error
import urllib.request

SAMPLE_PATIENT = {
    "age": 58,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 2,
    "ca": 0,
    "thal": 2,
}


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def run_command(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        print(f"Command not found: {command[0]}")
        return
    except subprocess.TimeoutExpired:
        print("Command timed out.")
        return

    output = result.stdout.strip() or result.stderr.strip()
    print(output if output else f"Exit code: {result.returncode}")


def get_json(url: str, method: str = "GET", payload: dict | None = None) -> None:
    print(f"$ {method} {url}")
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
            print(json.dumps(parsed, indent=2))
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}")


def get_status(url: str) -> None:
    print(f"$ GET {url}")
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            print(f"HTTP {response.status} {response.reason}")
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}")


def main() -> None:
    print_header("Heart Disease MLOps - Local Service Evidence")
    print(
        textwrap.dedent(
            """
            Use this terminal output as a real submission screenshot.
            It shows the running containers plus live API and monitoring checks.
            """
        ).strip()
    )

    print_header("Container Services")
    engine = "podman" if shutil.which("podman") else "docker"
    run_command(
        [
            engine,
            "ps",
            "--format",
            "table {{.Names}}\t{{.Status}}\t{{.Ports}}",
        ]
    )

    print_header("API Health")
    get_json("http://localhost:8000/health")

    print_header("Prediction Smoke Test")
    get_json("http://localhost:8000/predict", method="POST", payload=SAMPLE_PATIENT)

    print_header("Supporting Service Status")
    for name, url in [
        ("Swagger", "http://localhost:8000/docs"),
        ("Streamlit UI", "http://localhost:8501"),
        ("MLflow", "http://localhost:5001"),
        ("Prometheus", "http://localhost:9090/-/healthy"),
        ("Grafana", "http://localhost:3000/api/health"),
    ]:
        print(f"\n{name}")
        get_status(url)

    print_header("Browser Screenshots To Capture Separately")
    print("Swagger:     http://localhost:8000/docs")
    print("Streamlit:   http://localhost:8501")
    print("MLflow:      http://localhost:5001")
    print("Prometheus:  http://localhost:9090/targets")
    print("Grafana:     http://localhost:3000")


if __name__ == "__main__":
    main()
