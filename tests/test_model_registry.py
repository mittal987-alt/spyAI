import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.registry import (
    initialize_registry_db,
    register_model,
    list_registered_models,
    update_model_stage,
    compare_registered_models,
)

@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    test_db = tmp_path / "test_registry.db"
    monkeypatch.setattr("app.models.registry.REGISTRY_DB_PATH", test_db)
    initialize_registry_db(test_db)

def test_model_registration_and_stage_transitions(tmp_path):
    test_db = tmp_path / "test_reg.db"
    # Register v1
    m1 = register_model(
        name="FraudDetector",
        version="1.0.0",
        stage="staging",
        framework="xgboost",
        metrics={"f1": 0.88, "precision": 0.90, "recall": 0.86},
        db_path=test_db,
    )
    assert m1["name"] == "FraudDetector"
    assert m1["version"] == "1.0.0"
    assert m1["stage"] == "staging"

    # Register v2 as production
    m2 = register_model(
        name="FraudDetector",
        version="2.0.0",
        stage="production",
        framework="xgboost",
        metrics={"f1": 0.94, "precision": 0.95, "recall": 0.93},
        db_path=test_db,
    )
    assert m2["stage"] == "production"

    # Promote v1 to production (should auto-archive v2)
    m1_promoted = update_model_stage(m1["model_id"], "production", db_path=test_db)
    assert m1_promoted["stage"] == "production"

    # Check that v2 is now archived
    models = list_registered_models(name="FraudDetector", db_path=test_db)
    m2_curr = [m for m in models if m["model_id"] == m2["model_id"]][0]
    assert m2_curr["stage"] == "archived"

    # Compare models
    comp = compare_registered_models(m1["model_id"], m2["model_id"], db_path=test_db)
    assert "f1" in comp["metric_comparisons"]
    assert comp["metric_comparisons"]["f1"]["improved"] is True

def test_model_registry_api(tmp_path, monkeypatch):
    test_db = tmp_path / "test_reg_api.db"
    monkeypatch.setattr("app.models.registry.REGISTRY_DB_PATH", test_db)
    client = TestClient(app)

    # 1. Register model
    payload = {
        "name": "ChurnPredictor",
        "version": "1.0.0",
        "stage": "development",
        "framework": "pytorch",
        "description": "Customer churn deep learning model",
        "metrics": {"accuracy": 0.92, "roc_auc": 0.96},
        "tags": ["churn", "retention"],
    }
    r = client.post("/api/v1/models/registry", json=payload)
    assert r.status_code == 201
    mid = r.json()["model_id"]

    # 2. Get model
    r_get = client.get(f"/api/v1/models/registry/{mid}")
    assert r_get.status_code == 200
    assert r_get.json()["name"] == "ChurnPredictor"

    # 3. Update stage
    r_stage = client.patch(f"/api/v1/models/registry/{mid}/stage", json={"stage": "staging"})
    assert r_stage.status_code == 200
    assert r_stage.json()["stage"] == "staging"

    # 4. List models
    r_list = client.get("/api/v1/models/registry?stage=staging")
    assert r_list.status_code == 200
    assert r_list.json()["count"] == 1

    # 5. Delete model
    r_del = client.delete(f"/api/v1/models/registry/{mid}")
    assert r_del.status_code == 200
    assert r_del.json()["deleted"] is True
