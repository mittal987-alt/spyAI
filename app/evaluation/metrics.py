
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X, y):
    predictions = model.predict(X)

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "sample_count": int(len(y)),
    }


def compare_metrics(baseline, changed):
    return {
        metric: round(changed[metric] - baseline[metric], 6)
        for metric in ("accuracy", "precision", "recall", "f1")
    }
