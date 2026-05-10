import numpy as np
import pandas as pd

from heart_disease_mlops.config import COLUMN_NAMES, TARGET_COLUMN
from heart_disease_mlops.data import (
    clean_heart_data,
    download_uci_heart_data,
    split_features_target,
)


def test_clean_heart_data_converts_missing_values_and_binary_target():
    raw = pd.DataFrame(
        [
            [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, "?", 6, 0],
            [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 1, 3, 3, 2],
        ],
        columns=COLUMN_NAMES,
    )

    clean = clean_heart_data(raw)

    assert clean[TARGET_COLUMN].tolist() == [0, 1]
    assert np.isnan(clean.loc[0, "ca"])


def test_split_features_target_returns_expected_shapes():
    raw = pd.DataFrame(
        [
            [57, 1, 2, 140, 241, 0, 1, 123, 1, 0.2, 2, 0, 3, 1],
            [44, 0, 1, 120, 220, 0, 0, 170, 0, 0.0, 1, 0, 3, 0],
        ],
        columns=COLUMN_NAMES,
    )

    X, y = split_features_target(raw)

    assert TARGET_COLUMN not in X.columns
    assert X.shape == (2, 13)
    assert y.tolist() == [1, 0]


def test_download_skips_existing_file(tmp_path, monkeypatch):
    existing = tmp_path / "heart.csv"
    existing.write_text("already here\n", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("network should not be called for existing data")

    monkeypatch.setattr("heart_disease_mlops.data.requests.get", fail_if_called)

    result = download_uci_heart_data(existing)

    assert result == existing
    assert existing.read_text(encoding="utf-8") == "already here\n"
