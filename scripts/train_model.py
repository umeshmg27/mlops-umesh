from __future__ import annotations

import argparse
import json

from heart_disease_mlops.config import (
    FIGURES_DIR,
    MLRUNS_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
)
from heart_disease_mlops.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and package heart disease models.")
    parser.add_argument("--data", default=str(PROCESSED_DATA_PATH), help="Cleaned dataset CSV path.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Output model path.")
    parser.add_argument("--metadata", default=str(MODEL_METADATA_PATH), help="Output metadata JSON path.")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Output model figure directory.")
    parser.add_argument("--tracking-uri", default=f"file:{MLRUNS_DIR}", help="MLflow tracking URI.")
    parser.add_argument("--experiment-name", default="heart-disease-classification")
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    metadata = run_training(
        data_path=args.data,
        model_path=args.model,
        metadata_path=args.metadata,
        figures_dir=args.figures_dir,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        cv=args.cv,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

