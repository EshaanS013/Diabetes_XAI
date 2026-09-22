import numpy as np
from fastapi.testclient import TestClient

from api.app import main as api_main
from api.app.main import app

client = TestClient(app)

VALID_FEATURES = {
    "HighBP": 1,
    "HighChol": 1,
    "CholCheck": 1,
    "BMI": 32,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 0,
    "Fruits": 0,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 4,
    "MentHlth": 5,
    "PhysHlth": 10,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": 9,
    "Education": 4,
    "Income": 3,
}


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert "screening" in r.json()["disclaimer"].lower() or "NOT a medical diagnosis" in r.json()["disclaimer"]


def test_model_info_endpoint():
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "feature_order" in body
    assert body["out_of_scope"]


def test_predict_with_explanations_if_model_present():
    info = client.get("/model-info").json()
    if not info.get("available"):
        return
    payload = {
        "features": VALID_FEATURES,
        "include_explanations": True,
        "audience": "doctor",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "disclaimer" in body
    assert body["explanations"] is not None
    assert body["explanations"]["status"] == "ok_partial_xai"


def test_predict_rejects_fractional_binary_feature():
    features = dict(VALID_FEATURES)
    features["HighBP"] = 0.5
    payload = {"features": features}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_uses_classification_score_for_label_and_calibrated_score_for_risk(monkeypatch):
    class ConstantModel:
        def __init__(self, positive: float):
            self.positive = positive

        def predict_proba(self, rows):
            return np.tile([1.0 - self.positive, self.positive], (len(rows), 1))

    artifact = {
        "model": ConstantModel(0.8),
        "classification_model": ConstantModel(0.8),
        "risk_model": ConstantModel(0.2),
        "feature_order": list(VALID_FEATURES),
        "threshold": 0.5,
        "definition_name": "test_model",
        "model_version": "test",
    }
    monkeypatch.setattr(api_main, "load_production_artifact", lambda: artifact)
    response = client.post("/predict", json={"features": VALID_FEATURES})
    assert response.status_code == 200
    body = response.json()
    assert body["classification_probability"] == 0.8
    assert body["risk_probability"] == 0.2
    assert body["predicted_label"] == 1
