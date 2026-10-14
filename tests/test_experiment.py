
from fastapi.testclient import TestClient

from app.main import app
from app.experiments.run import run_experiment

client = TestClient(app)


def test_experiment_returns_expected_structure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.experiments.run.REPORTS_DIR",
        tmp_path,
    )

    report = run_experiment()

    assert report["experiment"] == "synthetic_feature_drift"
    assert report["dataset"]["training_samples"] > 0
    assert 0 <= report["baseline_metrics"]["accuracy"] <= 1
    assert 0 <= report["changed_metrics"]["accuracy"] <= 1
    assert "feature_drift" in report
    assert (tmp_path / "latest_experiment.json").exists()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_drift_endpoint():
    response = client.post("/api/v1/experiments/drift")

    assert response.status_code == 200
    assert response.json()["experiment"] == "synthetic_feature_drift"
