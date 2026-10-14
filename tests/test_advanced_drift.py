
import numpy as np
import pytest

from app.diagnostics.drift import (
    benjamini_hochberg,
    detect_feature_drift,
)


def test_benjamini_hochberg_adjusts_p_values():
    adjusted, rejected = benjamini_hochberg(
        [0.01, 0.04, 0.03, 0.20],
        alpha=0.05,
    )

    assert len(adjusted) == 4
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[1] == pytest.approx(0.053333, abs=1e-5)
    assert adjusted[2] == pytest.approx(0.053333, abs=1e-5)
    assert adjusted[3] == pytest.approx(0.20)
    assert rejected == [True, False, False, False]


def test_empty_p_values():
    adjusted, rejected = benjamini_hochberg([])

    assert adjusted == []
    assert rejected == []


def test_invalid_p_values_are_rejected():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.01, 1.5])


def test_large_shift_is_detected_after_correction():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(1000, 3))
    current = reference.copy()
    current[:, 0] += 5.0

    results = detect_feature_drift(
        reference,
        current,
        ["shifted", "stable_a", "stable_b"],
    )

    shifted = results[0]

    assert shifted["drift_detected"] is True
    assert shifted["adjusted_p_value"] < 0.05
    assert shifted["effect_size"] >= 0.10
    assert shifted["severity"] == "high"


def test_small_effect_can_be_filtered():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(1000, 1))
    current = reference + 0.01

    results = detect_feature_drift(
        reference,
        current,
        ["feature"],
        effect_threshold=0.10,
    )

    assert results[0]["drift_detected"] is False


def test_invalid_threshold_order_is_rejected():
    with pytest.raises(ValueError):
        detect_feature_drift(
            np.ones((10, 1)),
            np.ones((10, 1)),
            ["feature"],
            medium_threshold=0.50,
            high_threshold=0.20,
        )
