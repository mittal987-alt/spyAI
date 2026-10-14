
import pytest
import torch
from torch import nn

from app.deep_learning.explainability import explain_predictions


class SmallNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.layers(x)


def test_shap_explanation_returns_feature_importance():
    torch.manual_seed(42)

    model = SmallNetwork()
    background = torch.randn(12, 4)
    samples = torch.randn(3, 4)

    report = explain_predictions(
        model,
        background,
        samples,
        feature_names=["age", "income", "score", "history"],
        max_background=10,
        max_samples=3,
    )

    assert report["method"] == "SHAP GradientExplainer"
    assert report["feature_count"] == 4
    assert len(report["global_feature_importance"]) == 4
    assert len(report["local_explanations"]) == 3


def test_explanation_rejects_mismatched_features():
    model = SmallNetwork()

    with pytest.raises(ValueError):
        explain_predictions(
            model,
            torch.randn(10, 4),
            torch.randn(2, 3),
        )


def test_explanation_rejects_invalid_feature_names():
    model = SmallNetwork()

    with pytest.raises(ValueError):
        explain_predictions(
            model,
            torch.randn(10, 4),
            torch.randn(2, 4),
            feature_names=["only_one_name"],
        )


def test_explanation_rejects_non_finite_samples():
    model = SmallNetwork()
    samples = torch.randn(2, 4)
    samples[0, 0] = float("nan")

    with pytest.raises(ValueError):
        explain_predictions(
            model,
            torch.randn(10, 4),
            samples,
        )
