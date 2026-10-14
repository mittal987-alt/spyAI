
from fastapi.testclient import TestClient

from app.main import app
from app.monitoring.alerts import (
    generate_drift_alerts,
    save_alerts,
)

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_alert_list_endpoint():
    response = client.get("/api/v1/alerts")

    assert response.status_code == 200
    assert "alerts" in response.json()
    assert "count" in response.json()


def test_alert_summary_endpoint():
    response = client.get("/api/v1/alerts/summary")

    assert response.status_code == 200

    result = response.json()
    assert "total" in result
    assert "high" in result
    assert "medium" in result
    assert "low" in result


def test_invalid_alert_severity():
    response = client.get(
        "/api/v1/alerts",
        params={"severity": "critical"},
    )

    assert response.status_code == 422


def test_unknown_alert_endpoint():
    response = client.get(
        "/api/v1/alerts/nonexistent-alert-id"
    )

    assert response.status_code == 404
