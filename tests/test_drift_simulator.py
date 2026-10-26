import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.diagnostics.drift_simulator import (
    get_drift_simulation_presets,
    simulate_drift_and_perturbations,
)

client = TestClient(app)


def test_get_drift_simulation_presets():
    presets = get_drift_simulation_presets()
    assert len(presets) >= 4
    preset_ids = {p["id"] for p in presets}
    assert "sensor_drift" in preset_ids
    assert "outlier_shock" in preset_ids
    assert "concept_drift" in preset_ids
    assert "clean_baseline" in preset_ids


def test_drift_simulation_validation():
    with pytest.raises(ValueError, match="n_samples must be at least 200"):
        simulate_drift_and_perturbations(n_samples=50, n_features=6)

    with pytest.raises(ValueError, match="n_features must be at least 4"):
        simulate_drift_and_perturbations(n_samples=300, n_features=2)

    with pytest.raises(ValueError, match="variance_scale must be strictly positive"):
        simulate_drift_and_perturbations(n_samples=300, n_features=6, variance_scale=0)

    with pytest.raises(ValueError, match="outlier_rate must be between 0.0 and 0.40"):
        simulate_drift_and_perturbations(n_samples=300, n_features=6, outlier_rate=0.75)


def test_drift_simulation_clean_control():
    # Clean baseline: no shift, variance=1.0, 0 noise, 0 outliers, 0 flips
    result = simulate_drift_and_perturbations(
        n_samples=400,
        n_features=6,
        mean_shift=0.0,
        variance_scale=1.0,
        gaussian_noise=0.0,
        outlier_rate=0.0,
        label_flip_rate=0.0,
        target_features=[],
        random_state=42,
    )

    assert result["status"] == "success"
    assert result["drift_summary"]["drifted_features_count"] == 0
    assert result["model_stress_test"]["resilience_rating"] == "highly_resilient"
    assert result["model_stress_test"]["accuracy_drop"] <= 0.05
    assert len(result["feature_drifts"]) == 6


def test_drift_simulation_severe_perturbation():
    # Severe shift: mean shift +3.5, variance 2.0x on features 0 and 1
    result = simulate_drift_and_perturbations(
        n_samples=500,
        n_features=6,
        mean_shift=3.5,
        variance_scale=2.0,
        gaussian_noise=0.5,
        outlier_rate=0.05,
        label_flip_rate=0.15,
        target_features=[0, 1],
        random_state=42,
    )

    assert result["status"] == "success"
    assert result["drift_summary"]["drifted_features_count"] >= 1
    assert result["drift_summary"]["max_psi_score"] > 0.25
    assert len(result["recommendations"]) > 0

    # Ensure distribution visual data generated for top features
    assert len(result["distribution_visuals"]) > 0
    top_vis = result["distribution_visuals"][0]
    assert "density_bins" in top_vis
    assert len(top_vis["density_bins"]) > 0


def test_drift_api_endpoints():
    # Presets endpoint
    res_presets = client.get("/api/v1/drift/presets")
    assert res_presets.status_code == 200
    assert "presets" in res_presets.json()

    # Simulation endpoint
    payload = {
        "n_samples": 400,
        "n_features": 6,
        "mean_shift": 2.0,
        "variance_scale": 1.5,
        "gaussian_noise": 0.2,
        "outlier_rate": 0.02,
        "label_flip_rate": 0.0,
        "target_features": [0],
        "random_state": 42,
    }
    res_sim = client.post("/api/v1/drift/simulate", json=payload)
    assert res_sim.status_code == 200, res_sim.text
    data = res_sim.json()
    assert "drift_summary" in data
    assert "model_stress_test" in data
    assert "feature_drifts" in data
