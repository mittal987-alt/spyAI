
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_dashboard_endpoint_returns_metrics():
    dashboard = {
        "service": "ModelAutopsy AI RAG Dashboard",
        "runs_included": 2,
        "finding_summary": {
            "runs_with_findings": 1,
            "runs_without_findings": 1,
            "runs_with_missing_finding_data": 0,
        },
        "metrics": {
            "precision_at_k": {
                "available_runs": 2,
                "average": 0.75,
                "latest_value": 0.8,
            }
        },
        "recent_runs": [],
        "limitations": [],
    }

    with patch(
        "app.main.get_rag_dashboard",
        return_value=dashboard,
    ) as dashboard_mock:
        response = client.get("/api/v1/rag/dashboard?limit=10")

    assert response.status_code == 200
    assert response.json()["runs_included"] == 2
    assert response.json()["metrics"]["precision_at_k"]["average"] == 0.75
    dashboard_mock.assert_called_once_with(limit=10)


def test_dashboard_endpoint_handles_empty_history():
    dashboard = {
        "service": "ModelAutopsy AI RAG Dashboard",
        "runs_included": 0,
        "finding_summary": {
            "runs_with_findings": 0,
            "runs_without_findings": 0,
            "runs_with_missing_finding_data": 0,
        },
        "metrics": {
            "precision_at_k": {
                "available_runs": 0,
                "average": None,
                "latest_value": None,
            }
        },
        "recent_runs": [],
        "limitations": [],
    }

    with patch(
        "app.main.get_rag_dashboard",
        return_value=dashboard,
    ):
        response = client.get("/api/v1/rag/dashboard")

    assert response.status_code == 200
    assert response.json()["runs_included"] == 0
    assert response.json()["recent_runs"] == []


def test_dashboard_endpoint_rejects_invalid_limit():
    response = client.get("/api/v1/rag/dashboard?limit=0")

    assert response.status_code == 422


def test_dashboard_endpoint_handles_internal_error():
    with patch(
        "app.main.get_rag_dashboard",
        side_effect=RuntimeError("Database unavailable"),
    ):
        response = client.get("/api/v1/rag/dashboard")

    assert response.status_code == 500
    assert response.json()["detail"] == "Could not build RAG dashboard"
