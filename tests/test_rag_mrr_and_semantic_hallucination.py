import pytest
from app.diagnostics.rag_evaluation import evaluate_retrieval
from app.diagnostics.hallucination import evaluate_hallucination
from app.monitoring.alerts import save_alerts, list_alerts, cleanup_old_alerts, initialize_alert_db
from fastapi.testclient import TestClient
from app.main import app

def test_rag_mrr_and_hit_rate():
    retrieved = [
        {"id": "doc-1", "text": "Unrelated topic information."},
        {"id": "doc-2", "text": "This document is highly relevant to query."},
        {"id": "doc-3", "text": "Another document here."},
    ]
    # doc-2 is at rank 2
    res = evaluate_retrieval(
        query="test query",
        retrieved_documents=retrieved,
        relevant_document_ids=["doc-2"],
        k=2,
    )
    assert res["mrr"] == 0.5
    assert res["hit_rate_at_k"] == 1.0

    # Test when relevant is not in top-k
    res_k1 = evaluate_retrieval(
        query="test query",
        retrieved_documents=retrieved,
        relevant_document_ids=["doc-2"],
        k=1,
    )
    assert res_k1["hit_rate_at_k"] == 0.0

def test_hallucination_semantic_similarity():
    # Closely matching texts
    ref = "The quick brown fox jumps over the lazy dog near the river bank."
    gen = "A quick brown fox jumped across the lazy dog by the river."
    eval_res = evaluate_hallucination(generated=gen, reference=ref)
    assert "semantic_similarity" in eval_res
    assert eval_res["semantic_similarity"]["score"] > 0.0

def test_alert_deduplication_and_cleanup(tmp_path, monkeypatch):
    test_db = tmp_path / "test_alerts.db"
    initialize_alert_db(test_db)
    monkeypatch.setattr("app.monitoring.alerts.ALERTS_DB_PATH", test_db)

    alert1 = {
        "alert_id": "alert-1",
        "experiment": "test",
        "feature": "feat_a",
        "severity": "high",
        "title": "High drift in feat_a",
        "message": "Drift detected",
        "effect_size": 0.45,
        "adjusted_p_value": 0.001,
        "created_at": "2026-11-16T10:00:00Z",
    }
    # Duplicate alert
    alert2 = {
        "alert_id": "alert-2",
        "experiment": "test",
        "feature": "feat_a",
        "severity": "high",
        "title": "High drift in feat_a duplicate",
        "message": "Drift detected again",
        "effect_size": 0.46,
        "adjusted_p_value": 0.001,
        "created_at": "2026-11-16T11:00:00Z",
    }

    inserted = save_alerts([alert1, alert2], db_path=test_db)
    # The duplicate should be skipped
    assert inserted == 1

    alerts = list_alerts(db_path=test_db)
    assert len(alerts) == 1

    # Test cleanup
    deleted = cleanup_old_alerts(retention_days=0, db_path=test_db)
    assert deleted >= 0

def test_alert_cleanup_api_endpoint(tmp_path, monkeypatch):
    test_db = tmp_path / "test_alerts_api.db"
    initialize_alert_db(test_db)
    monkeypatch.setattr("app.monitoring.alerts.ALERTS_DB_PATH", test_db)

    client = TestClient(app)
    resp = client.delete("/api/v1/alerts/cleanup?retention_days=30")
    assert resp.status_code == 200
    assert "deleted" in resp.json()
