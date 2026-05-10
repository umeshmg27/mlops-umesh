from __future__ import annotations

import argparse

import requests

from heart_disease_mlops.data import sample_patient


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a sample request to the API.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()

    response = requests.post(f"{args.api_url.rstrip('/')}/predict", json=sample_patient(), timeout=10)
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()

