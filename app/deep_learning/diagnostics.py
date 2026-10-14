
from __future__ import annotations

import numpy as np


def analyze_training_history(
    history: dict,
    overfit_gap_threshold: float = 0.10,
    patience: int = 5,
) -> dict:
    """
    Analyze training history for overfitting and convergence issues.

    Accuracy gap is measured as training accuracy minus validation
    accuracy. Loss and accuracy use the final epoch for the final gap.
    """
    required = {
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy",
    }

    missing = required - set(history)
    if missing:
        raise ValueError(
            f"Missing history fields: {sorted(missing)}"
        )

    arrays = {
        key: np.asarray(history[key], dtype=float)
        for key in required
    }

    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        raise ValueError(
            "All history arrays must have the same length of at least 2"
        )

    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(
                f"History field '{key}' contains non-finite values"
            )

    if not 0 <= overfit_gap_threshold <= 1:
        raise ValueError(
            "overfit_gap_threshold must be between 0 and 1"
        )

    if patience < 1:
        raise ValueError("patience must be positive")

    train_loss = arrays["train_loss"]
    val_loss = arrays["val_loss"]
    train_accuracy = arrays["train_accuracy"]
    val_accuracy = arrays["val_accuracy"]

    if np.any(train_loss < 0) or np.any(val_loss < 0):
        raise ValueError("Loss values cannot be negative")

    for key in ("train_accuracy", "val_accuracy"):
        if np.any((arrays[key] < 0) | (arrays[key] > 1)):
            raise ValueError(f"{key} must be between 0 and 1")

    accuracy_gap = float(
        train_accuracy[-1] - val_accuracy[-1]
    )
    loss_gap = float(val_loss[-1] - train_loss[-1])

    # Detect a rising validation loss while training loss keeps falling.
    recent_val = val_loss[-patience:]
    recent_train = train_loss[-patience:]

    validation_loss_increasing = (
        len(recent_val) >= 2
        and recent_val[-1] > recent_val[0]
    )
    training_loss_decreasing = (
        len(recent_train) >= 2
        and recent_train[-1] < recent_train[0]
    )

    overfitting = (
        accuracy_gap >= overfit_gap_threshold
        or (
            validation_loss_increasing
            and training_loss_decreasing
        )
    )

    # Large epoch-to-epoch loss jumps are a warning, not proof of instability.
    loss_changes = np.diff(train_loss)
    loss_scale = max(float(np.median(train_loss)), 1e-8)
    unstable_training = bool(
        np.any(np.abs(loss_changes) > 0.5 * loss_scale)
    )

    best_epoch = int(np.argmin(val_loss)) + 1
    final_epoch = len(train_loss)

    if best_epoch < final_epoch and validation_loss_increasing:
        convergence_status = "validation_loss_increasing"
    elif np.all(np.abs(loss_changes[-min(patience, len(loss_changes)):]) < 1e-4):
        convergence_status = "loss_plateau"
    else:
        convergence_status = "still_learning_or_stable"

    findings = []

    if overfitting:
        findings.append({
            "type": "overfitting",
            "severity": "high" if accuracy_gap >= 0.20 else "medium",
            "message": (
                "Training performance is stronger than validation "
                "performance, or validation loss is worsening."
            ),
            "recommendation": (
                "Consider early stopping, stronger regularization, "
                "dropout tuning, or more representative training data."
            ),
        })

    if unstable_training:
        findings.append({
            "type": "training_instability",
            "severity": "medium",
            "message": (
                "Large epoch-to-epoch training-loss changes were observed."
            ),
            "recommendation": (
                "Inspect the learning rate, batch size, gradients, "
                "and data preprocessing."
            ),
        })

    if convergence_status == "loss_plateau":
        findings.append({
            "type": "loss_plateau",
            "severity": "low",
            "message": "Training loss has changed very little recently.",
            "recommendation": (
                "Review whether more epochs or a learning-rate schedule "
                "could improve validation performance."
            ),
        })

    return {
        "epochs": final_epoch,
        "final_train_loss": float(train_loss[-1]),
        "final_validation_loss": float(val_loss[-1]),
        "final_train_accuracy": float(train_accuracy[-1]),
        "final_validation_accuracy": float(val_accuracy[-1]),
        "accuracy_gap": accuracy_gap,
        "validation_minus_training_loss": loss_gap,
        "best_validation_epoch": best_epoch,
        "best_validation_loss": float(val_loss[best_epoch - 1]),
        "overfitting_detected": bool(overfitting),
        "training_instability_detected": unstable_training,
        "convergence_status": convergence_status,
        "findings": findings,
    }
