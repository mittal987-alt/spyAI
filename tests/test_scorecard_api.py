import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.monitoring.scorecard import initialize_scorecard_db

@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    test_db = tmp_path / "test_scorecard.db"
    monkeypatch.setattr("app.monitoring.scorecard.SCORECARD_DB", test_db)
    initialize_scorecard_db(test_db)

def test_scorecard_crud_and_leaderboard():
    client = TestClient(app)

    # 1. Get metadata
    meta_resp = client.get("/api/v1/scorecard/metadata")
    assert meta_resp.status_code == 200
    assert "dimensions" in meta_resp.json()

    # 2. Add entry
    payload = {
        "model_name": "TestModel-Alpha",
        "model_version": "1.0.0",
        "source": "manual",
        "notes": "Testing entry creation",
        "metrics": {
            "precision_at_k": 0.85,
            "recall_at_k": 0.80,
            "answer_relevance_score": 0.90,
            "ndcg_at_k": 0.88,
            "context_support_coverage": 0.75,
            "hallucination_risk_score": 15.0,
            "safety_score": 95.0,
            "max_effect_size": 0.05,
        },
    }
    create_resp = client.post("/api/v1/scorecard/entries", json=payload)
    assert create_resp.status_code == 201
    data = create_resp.json()
    assert data["model_name"] == "TestModel-Alpha"
    assert data["composite_score"] > 80.0
    entry_id = data["entry_id"]

    # 3. Get entry by ID
    get_resp = client.get(f"/api/v1/scorecard/entries/{entry_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["entry_id"] == entry_id

    # 4. List entries
    list_resp = client.get("/api/v1/scorecard/entries")
    assert list_resp.status_code == 200
    assert len(list_resp.json()["entries"]) == 1

    # 5. Leaderboard
    lb_resp = client.get("/api/v1/scorecard/leaderboard")
    assert lb_resp.status_code == 200
    lb = lb_resp.json()
    assert lb["total_models"] == 1
    assert lb["leaderboard"][0]["model_name"] == "TestModel-Alpha"
    assert lb["leaderboard"][0]["rank"] == 1

    # 6. Delete entry
    del_resp = client.delete(f"/api/v1/scorecard/entries/{entry_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True
