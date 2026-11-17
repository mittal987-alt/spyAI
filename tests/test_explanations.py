from fastapi.testclient import TestClient
from app.main import app
from app.diagnostics.explanations import generate_heuristic_explanation, explain_drift

def test_heuristic_drift_explanation():
    summary = {
        "total_features": 10,
        "drifted_features_count": 4,
    }
    drifts = [
        {"feature_name": "feat_0", "ks_statistic": 0.35, "psi_score": 0.45, "drift_detected": True, "severity": "high"},
        {"feature_name": "feat_1", "ks_statistic": 0.22, "psi_score": 0.18, "drift_detected": True, "severity": "medium"},
    ]

    res = generate_heuristic_explanation(summary, drifts)
    assert res["health_status"] == "CRITICAL"
    assert "recommended_actions" in res
    assert len(res["recommended_actions"]) > 0
    assert len(res["key_findings"]) > 0

def test_explain_drift_api():
    client = TestClient(app)
    payload = {
        "drift_summary": {"total_features": 5, "drifted_features_count": 1},
        "feature_drifts": [
            {"feature_name": "sensor_x", "ks_statistic": 0.12, "psi_score": 0.08, "drift_detected": True, "severity": "medium"}
        ],
    }
    r = client.post("/api/v1/explain/drift", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "headline" in data
    assert "recommended_actions" in data
