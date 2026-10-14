
import numpy as np
import pytest

from app.diagnostics.drift import detect_feature_drift


def test_detects_large_distribution_shift():
    rng = np.random.default_rng(42)

    reference = rng.normal(0, 1, size=(1000, 2))
    current = reference.copy()
    current[:, 0] += 5

    result = detect_feature_drift(
        reference,
        current,
        ["shifted", "unchanged"],
    )

    assert result[0]["drift_detected"] is True
    assert result[1]["drift_detected"] is False


def test_rejects_mismatched_dimensions():
    with pytest.raises(ValueError):
        detect_feature_drift(
            np.zeros((10, 2)),
            np.zeros((10, 3)),
            ["a", "b"],
        )


def test_rejects_empty_data():
    with pytest.raises(ValueError):
        detect_feature_drift(
            np.empty((0, 1)),
            np.ones((10, 1)),
            ["feature_0"],
        )


def test_rejects_non_finite_data():
    with pytest.raises(ValueError):
        detect_feature_drift(
            np.array([[1.0], [float("nan")]]),
            np.array([[2.0], [3.0]]),
            ["feature_0"],
        )
