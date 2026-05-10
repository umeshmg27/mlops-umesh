from __future__ import annotations

from run_eda import run_eda

from heart_disease_mlops.config import MODEL_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH
from heart_disease_mlops.data import download_uci_heart_data
from heart_disease_mlops.training import run_training


def main() -> None:
    if not RAW_DATA_PATH.exists():
        print("Raw dataset missing; downloading from UCI.")
        download_uci_heart_data(RAW_DATA_PATH)

    if not PROCESSED_DATA_PATH.exists():
        print("Processed dataset missing; running EDA and cleaning.")
        run_eda(RAW_DATA_PATH, PROCESSED_DATA_PATH, "reports/figures")

    if not MODEL_PATH.exists():
        print("Model missing; training model.")
        run_training(PROCESSED_DATA_PATH)
    else:
        print(f"Model already exists at {MODEL_PATH}")


if __name__ == "__main__":
    main()
