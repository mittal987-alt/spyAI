
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_report(precision=0.8, relevance=0.9):
    return {
        "service": "ModelAutopsy AI RAG Evaluation",
        "query": "What is the answer?",
        "retrieved_document_count": 1,
        "relevant_document_count": 1,
        "evaluation": {
            "retrieval": {"precision_at_k": precision},
            "answer": {"answer_relevance_score": relevance},
        },
        "summary": {
            "precision_at_k": precision,
            "recall_at_k": 1.0,
            "reciprocal_rank": 1.0,
            "ndcg_at_k": 1.0,
            "answer_relevance_score": relevance,
            "context_support_coverage": 1.0,
            "potentially_unsupported_sentence_count": 0,
            "finding_count": 0,
        },
        "findings": [],
        "limitations": [],
    }


def test_rag_evaluate_saves_history():
    with (
        patch(
            "app.main.run_rag_evaluation",
            return_value=make_report(),
        ),
        patch(
            "app.main.save_rag_evaluation",
            return_value={
                "run_id": "run-123",
                "created_at": "2026-10-14T10:00:00+00:00",
                "query": "What is the answer?",
            },
        ) as save_mock,
    ):
        response = client.post(
            "/api/v1/rag/evaluate",
            json={
                "query": "What is the answer?",
                "answer": "Here is the answer.",
                "retrieved_documents": [
                    {"id": "doc-1", "text": "Supporting context."}
                ],
                "relevant_document_ids": ["doc-1"],
            },
        )

    assert response.status_code == 200
    assert response.json()["history"]["run_id"] == "run-123"
    save_mock.assert_called_once()


def test_rag_history_returns_evaluations():
    with patch(
        "app.main.list_rag_evaluations",
        return_value=[
            {
                "run_id": "run-123",
                "created_at": "2026-10-14T10:00:00+00:00",
                "query": "What is the answer?",
                "summary": make_report()["summary"],
            }
        ],
    ):
        response = client.get("/api/v1/rag/history")

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["evaluations"][0]["run_id"] == "run-123"


def test_rag_history_rejects_invalid_limit():
    response = client.get("/api/v1/rag/history?limit=0")

    assert response.status_code == 422


def test_rag_history_item_returns_404_for_unknown_run():
    with patch(
        "app.main.get_rag_evaluation",
        return_value=None,
    ):
        response = client.get("/api/v1/rag/history/unknown-run")

    assert response.status_code == 404


def test_rag_history_item_returns_saved_report():
    saved = {
        "run_id": "run-123",
        "created_at": "2026-10-14T10:00:00+00:00",
        "query": "What is the answer?",
        "report": make_report(),
    }

    with patch(
        "app.main.get_rag_evaluation",
        return_value=saved,
    ):
        response = client.get("/api/v1/rag/history/run-123")

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-123"
    assert response.json()["report"]["summary"]["precision_at_k"] == 0.8


def test_rag_comparison_endpoint():
    comparison = {
        "baseline_run_id": "baseline-1",
        "candidate_run_id": "candidate-1",
        "regression_threshold": 0.1,
        "metrics": {
            "precision_at_k": {
                "baseline": 0.9,
                "candidate": 0.7,
                "change": -0.2,
                "relative_change": -0.222,
                "status": "regression",
            }
        },
        "regressions": ["precision_at_k"],
        "improvements": [],
        "regression_detected": True,
        "limitations": [],
    }

    with patch(
        "app.main.compare_rag_evaluations",
        return_value=comparison,
    ):
        response = client.post(
            "/api/v1/rag/compare",
            json={
                "baseline_run_id": "baseline-1",
                "candidate_run_id": "candidate-1",
                "regression_threshold": 0.1,
            },
        )

    assert response.status_code == 200
    assert response.json()["regression_detected"] is True
    assert "precision_at_k" in response.json()["regressions"]


def test_rag_comparison_missing_run_returns_404():
    with patch(
        "app.main.compare_rag_evaluations",
        side_effect=ValueError("Baseline evaluation run was not found"),
    ):
        response = client.post(
            "/api/v1/rag/compare",
            json={
                "baseline_run_id": "missing-baseline",
                "candidate_run_id": "candidate-1",
            },
        )

    assert response.status_code == 404


def test_rag_comparison_rejects_invalid_threshold():
    response = client.post(
        "/api/v1/rag/compare",
        json={
            "baseline_run_id": "baseline-1",
            "candidate_run_id": "candidate-1",
            "regression_threshold": 0,
        },
    )

    assert response.status_code == 422
