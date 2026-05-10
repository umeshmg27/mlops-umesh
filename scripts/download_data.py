from __future__ import annotations

import argparse

from heart_disease_mlops.config import RAW_DATA_PATH, UCI_CLEVELAND_URL
from heart_disease_mlops.data import download_uci_heart_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCI Heart Disease data.")
    parser.add_argument("--output", default=str(RAW_DATA_PATH), help="Raw CSV output path.")
    parser.add_argument("--url", default=UCI_CLEVELAND_URL, help="Dataset URL.")
    parser.add_argument("--force", action="store_true", help="Redownload even if the file exists.")
    args = parser.parse_args()

    path = download_uci_heart_data(args.output, args.url, force=args.force)
    print(f"Dataset available at {path}")


if __name__ == "__main__":
    main()
