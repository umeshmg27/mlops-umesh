from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from heart_disease_mlops.config import COLUMN_NAMES, TARGET_COLUMN, UCI_CLEVELAND_URL


def download_uci_heart_data(
    output_path: str | Path,
    url: str = UCI_CLEVELAND_URL,
    timeout: int = 30,
    force: bool = False,
) -> Path:
    """Download the processed Cleveland heart disease dataset from UCI."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0 and not force:
        return output

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    text = response.text.strip()
    if not text or "html" in text[:100].lower():
        raise ValueError(f"Unexpected dataset response from {url}")

    output.write_text(text + "\n", encoding="utf-8")
    return output


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw UCI data file with the assignment's 14-column schema."""
    return pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")


def clean_heart_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw UCI data and convert the original multi-class target to binary."""
    missing = set(COLUMN_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    cleaned = df[COLUMN_NAMES].copy()
    cleaned = cleaned.where(cleaned.ne("?"), np.nan)

    for column in COLUMN_NAMES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    cleaned[TARGET_COLUMN] = (cleaned[TARGET_COLUMN] > 0).astype(int)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def save_clean_data(raw_path: str | Path, output_path: str | Path) -> Path:
    """Load, clean, and persist the processed dataset."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(raw_path)
    clean = clean_heart_data(raw)
    clean.to_csv(output, index=False)
    return output


def load_processed_data(path: str | Path) -> pd.DataFrame:
    """Load a cleaned dataset file."""
    df = pd.read_csv(path)
    return clean_heart_data(df)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and binary target vector."""
    clean = clean_heart_data(df)
    return clean.drop(columns=[TARGET_COLUMN]), clean[TARGET_COLUMN]


def sample_patient() -> dict[str, float | int]:
    """Return a realistic sample request for API smoke tests and docs."""
    return {
        "age": 57,
        "sex": 1,
        "cp": 2,
        "trestbps": 140,
        "chol": 241,
        "fbs": 0,
        "restecg": 1,
        "thalach": 123,
        "exang": 1,
        "oldpeak": 0.2,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }
