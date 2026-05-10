from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from heart_disease_mlops.config import (
    FIGURES_DIR,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from heart_disease_mlops.data import clean_heart_data
from heart_disease_mlops.features import build_model_pipeline, model_candidates


def _configure_matplotlib() -> None:
    project_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> Path:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _save_roc_curve(y_true: pd.Series, y_prob: np.ndarray, output_path: Path) -> Path:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def run_training(
    data_path: str | Path = PROCESSED_DATA_PATH,
    model_path: str | Path = MODEL_PATH,
    metadata_path: str | Path = MODEL_METADATA_PATH,
    figures_dir: str | Path = FIGURES_DIR,
    tracking_uri: str | None = None,
    experiment_name: str = "heart-disease-classification",
    cv: int = 5,
    test_size: float = 0.2,
) -> dict[str, object]:
    """Train two classifiers, track experiments with MLflow, and save the best model."""
    import mlflow
    import mlflow.sklearn

    data_path = Path(data_path)
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    figures_dir = Path(figures_dir)

    df = clean_heart_data(pd.read_csv(data_path))
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    results: dict[str, dict[str, object]] = {}
    best_name = ""
    best_pipeline = None
    best_score = -1.0

    for name, estimator in model_candidates().items():
        pipeline = build_model_pipeline(estimator)
        cv_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=False,
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]

        metrics = _classification_metrics(y_test, y_pred, y_prob)
        cv_metrics = {
            f"cv_{metric}_mean": float(np.mean(cv_scores[f"test_{metric}"]))
            for metric in scoring
        }
        cv_std = {
            f"cv_{metric}_std": float(np.std(cv_scores[f"test_{metric}"]))
            for metric in scoring
        }
        all_metrics = {**cv_metrics, **cv_std, **{f"test_{k}": v for k, v in metrics.items()}}

        run_figures_dir = figures_dir / name
        confusion_path = _save_confusion_matrix(
            y_test, y_pred, run_figures_dir / "confusion_matrix.png"
        )
        roc_path = _save_roc_curve(y_test, y_prob, run_figures_dir / "roc_curve.png")

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("cv_folds", cv)
            mlflow.log_param("test_size", test_size)
            mlflow.log_metrics(all_metrics)
            mlflow.log_artifact(str(confusion_path))
            mlflow.log_artifact(str(roc_path))
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

        results[name] = {
            "metrics": all_metrics,
            "figures": [_portable_path(confusion_path), _portable_path(roc_path)],
        }

        score = metrics["roc_auc"]
        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model was trained")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now(UTC).isoformat(),
        "selection_metric": "test_roc_auc",
        "model_path": _portable_path(model_path),
        "data_path": _portable_path(data_path),
        "target_column": TARGET_COLUMN,
        "metrics": results,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
