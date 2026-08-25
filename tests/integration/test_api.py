from fastapi.testclient import TestClient

from api.app.main import app

client = TestClient(app)


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
        "features": {
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
        },
        "include_explanations": True,
        "audience": "doctor",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "disclaimer" in body
    assert body["explanations"] is not None
    assert body["explanations"]["status"] == "ok_partial_xai"
