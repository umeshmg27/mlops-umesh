import numpy as np
from fastapi.testclient import TestClient

import heart_disease_mlops.api as api
from heart_disease_mlops.data import sample_patient


class FakeModel:
    classes_ = np.array([0, 1])

    def predict(self, frame):
        return np.ones(len(frame), dtype=int)

    def predict_proba(self, frame):
        return np.tile(np.array([[0.2, 0.8]]), (len(frame), 1))


def test_predict_endpoint_with_fake_model(monkeypatch):
    monkeypatch.setattr(
        api,
        "MODEL_BUNDLE",
        {"model": FakeModel(), "metadata": {"model_name": "fake_model"}},
    )
    client = TestClient(api.app)

    response = client.post("/predict", json=sample_patient())

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["risk_probability"] == 0.8
    assert body["model_name"] == "fake_model"

