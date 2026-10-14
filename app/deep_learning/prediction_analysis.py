
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def analyze_predictions(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    high_confidence_threshold: float = 0.90,
) -> dict:
    """
    Analyze model predictions, confidence, and class-wise errors.

    Confidence is the maximum softmax probability for each prediction.
    High-confidence errors are predictions that are wrong despite having
    confidence at or above the configured threshold.
    """
    if not 0 < high_confidence_threshold <= 1:
        raise ValueError(
            "high_confidence_threshold must be in (0, 1]"
        )

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional tensor")

    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional tensor")

    if len(X) == 0 or len(y) == 0:
        raise ValueError("X and y must not be empty")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")

    if not torch.isfinite(X).all():
        raise ValueError("X contains non-finite values")

    if y.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("y must contain integer class labels")

    model.eval()

    with torch.no_grad():
        logits = model(X)

        if logits.ndim != 2:
            raise ValueError("Model output must have shape (samples, classes)")

        if logits.shape[0] != len(y):
            raise ValueError("Model output count does not match labels")

        if logits.shape[1] < 2:
            raise ValueError("Model must output at least two classes")

        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite logits")

        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)

    y_true = y.detach().cpu().numpy().astype(int)
    y_pred = predictions.detach().cpu().numpy().astype(int)
    confidence_values = confidence.detach().cpu().numpy()

    num_classes = logits.shape[1]
    labels = list(range(num_classes))

    if np.any(y_true < 0) or np.any(y_true >= num_classes):
        raise ValueError("Labels must match the model's output classes")

    correct = y_true == y_pred
    incorrect = ~correct
    high_confidence = confidence_values >= high_confidence_threshold
    high_confidence_errors = incorrect & high_confidence

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    class_metrics = {}

    for class_index in labels:
        class_report = report[str(class_index)]

        class_metrics[str(class_index)] = {
            "precision": float(class_report["precision"]),
            "recall": float(class_report["recall"]),
            "f1_score": float(class_report["f1-score"]),
            "support": int(class_report["support"]),
        }

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    incorrect_indices = np.flatnonzero(incorrect)
    high_confidence_error_indices = np.flatnonzero(
        high_confidence_errors
    )

    return {
        "sample_count": int(len(y_true)),
        "class_count": int(num_classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": matrix.tolist(),
        "class_metrics": class_metrics,
        "confidence": {
            "mean": float(np.mean(confidence_values)),
            "median": float(np.median(confidence_values)),
            "minimum": float(np.min(confidence_values)),
            "maximum": float(np.max(confidence_values)),
            "low_confidence_count": int(
                np.sum(confidence_values < 0.60)
            ),
            "high_confidence_threshold": float(
                high_confidence_threshold
            ),
            "high_confidence_count": int(
                np.sum(high_confidence)
            ),
        },
        "errors": {
            "incorrect_count": int(np.sum(incorrect)),
            "error_rate": float(np.mean(incorrect)),
            "high_confidence_error_count": int(
                np.sum(high_confidence_errors)
            ),
            "high_confidence_error_rate": float(
                np.mean(high_confidence_errors)
            ),
            "incorrect_sample_indices": incorrect_indices[:100].tolist(),
            "high_confidence_error_indices": (
                high_confidence_error_indices[:100].tolist()
            ),
        },
    }


def generate_prediction_findings(
    report: dict,
    low_recall_threshold: float = 0.60,
) -> list[dict]:
    """Convert prediction-analysis results into diagnostic findings."""
    if not 0 <= low_recall_threshold <= 1:
        raise ValueError("low_recall_threshold must be between 0 and 1")

    findings = []
    errors = report["errors"]
    confidence = report["confidence"]

    if errors["high_confidence_error_count"] > 0:
        findings.append({
            "type": "high_confidence_errors",
            "severity": "high",
            "message": (
                f'{errors["high_confidence_error_count"]} incorrect '
                "predictions were made with high confidence."
            ),
            "recommendation": (
                "Inspect these samples for mislabeled data, ambiguous "
                "examples, class imbalance, and train-validation mismatch."
            ),
        })

    for class_name, metrics in report["class_metrics"].items():
        if (
            metrics["support"] > 0
            and metrics["recall"] < low_recall_threshold
        ):
            findings.append({
                "type": "low_class_recall",
                "severity": "medium",
                "class": class_name,
                "message": (
                    f"Class {class_name} has recall "
                    f'{metrics["recall"]:.3f}.'
                ),
                "recommendation": (
                    "Review false negatives for this class and examine "
                    "class balance, features, and label quality."
                ),
            })

    if confidence["mean"] < 0.60:
        findings.append({
            "type": "low_prediction_confidence",
            "severity": "medium",
            "message": (
                "Average prediction confidence is below 0.60."
            ),
            "recommendation": (
                "Inspect ambiguous samples and evaluate probability "
                "calibration on a separate validation set."
            ),
        })

    return findings
