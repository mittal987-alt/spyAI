
from app.evaluation.metrics import compare_metrics


def test_metric_deltas_use_changed_minus_baseline():
    baseline = {
        "accuracy": 0.90,
        "precision": 0.80,
        "recall": 0.70,
        "f1": 0.75,
    }

    changed = {
        "accuracy": 0.80,
        "precision": 0.85,
        "recall": 0.60,
        "f1": 0.70,
    }

    result = compare_metrics(baseline, changed)

    assert result["accuracy"] == -0.10
    assert result["precision"] == 0.05
    assert result["recall"] == -0.10
    assert result["f1"] == -0.05
