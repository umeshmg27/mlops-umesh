from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "heart.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "heart_clean.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SCREENSHOTS_DIR = REPORTS_DIR / "screenshots"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"

RANDOM_STATE = 42
TARGET_COLUMN = "target"

UCI_CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    TARGET_COLUMN,
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

