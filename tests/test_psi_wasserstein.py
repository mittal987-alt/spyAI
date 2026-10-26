"""
Unit tests for Population Stability Index (PSI) and Wasserstein Distance.
"""

import numpy as np
import pytest
from app.diagnostics.drift import (
    calculate_psi,
    calculate_wasserstein,
    detect_feature_drift,
)


def test_psi_identical_distributions():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 1000)

    result = calculate_psi(data, data)
    assert result["psi_score"] == pytest.approx(0.0, abs=1e-3)
    assert result["risk_level"] == "stable"
    assert len(result["bins"]) > 0


def test_psi_large_shift():
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, 1000)
    current = rng.normal(3.0, 1, 1000)  # Significant shift

    result = calculate_psi(reference, current)
    assert result["psi_score"] > 0.25
    assert result["risk_level"] == "significant"


def test_wasserstein_distance():
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cur = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    w_dist = calculate_wasserstein(ref, cur)
    assert w_dist == pytest.approx(1.0, abs=1e-3)


def test_detect_feature_drift_includes_psi_and_wasserstein():
    rng = np.random.default_rng(42)
    reference = rng.normal(0, 1, (500, 2))
    current = reference.copy()
    current[:, 0] += 2.5

    results = detect_feature_drift(reference, current, ["feature_a", "feature_b"])
    assert len(results) == 2

    # Check feature_a (shifted)
    feat_a = results[0]
    assert "psi" in feat_a
    assert "wasserstein_distance" in feat_a
    assert feat_a["psi"]["risk_level"] == "significant"
    assert feat_a["wasserstein_distance"] > 1.5
    assert "summary_stats" in feat_a
    assert feat_a["summary_stats"]["current_mean"] > feat_a["summary_stats"]["reference_mean"]

    # Check feature_b (unchanged)
    feat_b = results[1]
    assert feat_b["psi"]["risk_level"] == "stable"
    assert feat_b["wasserstein_distance"] < 0.2
