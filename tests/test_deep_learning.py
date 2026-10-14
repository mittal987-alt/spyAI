
import pytest

from app.deep_learning.trainer import train_model
from app.deep_learning.diagnostics import analyze_training_history


def test_training_produces_expected_history():
    result = train_model(
        epochs=3,
        n_samples=300,
        n_features=8,
        batch_size=32,
    )

    assert len(result["history"]["train_loss"]) == 3
    assert len(result["history"]["val_loss"]) == 3
    assert len(result["history"]["train_accuracy"]) == 3
    assert len(result["history"]["val_accuracy"]) == 3


def test_training_metrics_are_finite():
    result = train_model(
        epochs=3,
        n_samples=300,
        n_features=8,
    )

    for values in result["history"].values():
        assert all(
            value == value and abs(value) != float("inf")
            for value in values
        )


def test_detects_overfitting_from_accuracy_gap():
    history = {
        "train_loss": [0.8, 0.5, 0.2],
        "val_loss": [0.9, 1.0, 1.2],
        "train_accuracy": [0.60, 0.80, 0.95],
        "val_accuracy": [0.58, 0.65, 0.70],
    }

    report = analyze_training_history(history)

    assert report["overfitting_detected"] is True
    assert any(
        finding["type"] == "overfitting"
        for finding in report["findings"]
    )


def test_rejects_missing_history_fields():
    with pytest.raises(ValueError):
        analyze_training_history({
            "train_loss": [0.5, 0.4],
        })


def test_rejects_non_finite_history():
    history = {
        "train_loss": [0.5, float("nan")],
        "val_loss": [0.6, 0.5],
        "train_accuracy": [0.7, 0.8],
        "val_accuracy": [0.6, 0.7],
    }

    with pytest.raises(ValueError):
        analyze_training_history(history)
