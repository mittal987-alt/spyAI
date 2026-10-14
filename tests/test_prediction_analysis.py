
import pytest
import torch
from torch import nn

from app.deep_learning.prediction_analysis import (
    analyze_predictions,
    generate_prediction_findings,
)


class FixedLogitModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer(
            "fixed_logits",
            torch.tensor(logits, dtype=torch.float32),
        )

    def forward(self, x):
        return self.fixed_logits[: len(x)]


def test_prediction_analysis_returns_class_metrics():
    model = FixedLogitModel([
        [5.0, 0.1],
        [0.2, 4.0],
        [3.0, 0.1],
        [0.1, 4.0],
    ])

    X = torch.zeros((4, 2))
    y = torch.tensor([0, 1, 1, 1])

    report = analyze_predictions(model, X, y)

    assert report["sample_count"] == 4
    assert report["class_count"] == 2
    assert len(report["confusion_matrix"]) == 2
    assert "precision" in report["class_metrics"]["0"]
    assert 0 <= report["accuracy"] <= 1


def test_detects_high_confidence_incorrect_prediction():
    model = FixedLogitModel([
        [8.0, 0.0],
        [0.0, 8.0],
    ])

    X = torch.zeros((2, 2))
    y = torch.tensor([1, 1])

    report = analyze_predictions(model, X, y)

    assert report["errors"]["high_confidence_error_count"] == 1

    findings = generate_prediction_findings(report)

    assert any(
        finding["type"] == "high_confidence_errors"
        for finding in findings
    )


def test_rejects_mismatched_sample_counts():
    model = FixedLogitModel([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    with pytest.raises(ValueError):
        analyze_predictions(
            model,
            torch.zeros((2, 2)),
            torch.tensor([0]),
        )


def test_rejects_invalid_confidence_threshold():
    model = FixedLogitModel([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    with pytest.raises(ValueError):
        analyze_predictions(
            model,
            torch.zeros((2, 2)),
            torch.tensor([0, 1]),
            high_confidence_threshold=0,
        )
