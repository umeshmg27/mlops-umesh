import pandas as pd

from heart_disease_mlops.config import FEATURE_COLUMNS
from heart_disease_mlops.features import build_model_pipeline, model_candidates


def test_pipeline_fits_and_predicts_on_minimal_data():
    X = pd.DataFrame(
        [
            [57, 1, 2, 140, 241, 0, 1, 123, 1, 0.2, 2, 0, 3],
            [44, 0, 1, 120, 220, 0, 0, 170, 0, 0.0, 1, 0, 3],
            [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 1, 3, 7],
            [39, 0, 0, 110, 180, 0, 0, 175, 0, 0.0, 1, 0, 3],
        ],
        columns=FEATURE_COLUMNS,
    )
    y = [1, 0, 1, 0]

    pipeline = build_model_pipeline(model_candidates()["logistic_regression"])
    pipeline.fit(X, y)

    prediction = pipeline.predict(X.iloc[[0]])
    probabilities = pipeline.predict_proba(X.iloc[[0]])

    assert prediction.shape == (1,)
    assert probabilities.shape == (1, 2)

