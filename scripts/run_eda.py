from __future__ import annotations

import argparse
import os
from pathlib import Path

from heart_disease_mlops.config import (
    FIGURES_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from heart_disease_mlops.data import clean_heart_data, load_raw_data


def _configure_matplotlib() -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")


def run_eda(raw_path: str | Path, processed_path: str | Path, figures_dir: str | Path) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data(raw_path)
    clean = clean_heart_data(raw)
    Path(processed_path).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed_path, index=False)

    sns.set_theme(style="whitegrid")

    ax = clean[TARGET_COLUMN].value_counts().sort_index().plot(kind="bar", color=["#4c78a8", "#f58518"])
    ax.set_title("Class Balance")
    ax.set_xlabel("Heart disease present")
    ax.set_ylabel("Patients")
    ax.set_xticklabels(["No", "Yes"], rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(figures_dir / "class_balance.png", dpi=160)
    plt.close(ax.figure)

    numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    clean[numeric_columns].hist(figsize=(12, 8), bins=20, color="#4c78a8", edgecolor="white")
    plt.suptitle("Numeric Feature Distributions")
    plt.tight_layout()
    plt.savefig(figures_dir / "numeric_histograms.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(clean.corr(numeric_only=True), cmap="vlag", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(figures_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    missing = raw.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    missing.plot(kind="bar", ax=ax, color="#72b7b2")
    ax.set_title("Missing Values In Raw Data")
    ax.set_ylabel("Missing count")
    fig.tight_layout()
    fig.savefig(figures_dir / "missing_values.png", dpi=160)
    plt.close(fig)

    print(f"Saved cleaned data to {processed_path}")
    print(f"Saved EDA figures to {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA and produce figures.")
    parser.add_argument("--raw", default=str(RAW_DATA_PATH), help="Raw dataset path.")
    parser.add_argument("--processed", default=str(PROCESSED_DATA_PATH), help="Processed CSV path.")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR), help="Output figure directory.")
    args = parser.parse_args()
    run_eda(args.raw, args.processed, args.figures_dir)


if __name__ == "__main__":
    main()
