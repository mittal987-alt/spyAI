
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_dataset(
    data: Any,
    feature_names: list[str],
    *,
    expected_features: int | None = None,
    max_duplicate_rate: float = 0.10,
) -> dict:
    """
    Validate a feature dataset before model training or inference.

    Errors block the experiment.
    Warnings are recorded but do not automatically block it.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not 0 <= max_duplicate_rate <= 1:
        raise ValueError("max_duplicate_rate must be between 0 and 1.")

    if not feature_names:
        raise ValueError("feature_names must not be empty.")

    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique.")

    try:
        frame = pd.DataFrame(data, columns=feature_names)
    except (TypeError, ValueError) as exc:
        return {
            "valid": False,
            "errors": [f"Cannot construct feature table: {exc}"],
            "warnings": [],
            "summary": {},
            "features": [],
        }

    if frame.empty:
        errors.append("Dataset must not be empty.")

    if expected_features is not None:
        if frame.shape[1] != expected_features:
            errors.append(
                f"Expected {expected_features} features, "
                f"received {frame.shape[1]}."
            )

    if frame.shape[1] != len(feature_names):
        errors.append("Feature count does not match feature names.")

    non_numeric = [
        str(column)
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]

    if non_numeric:
        errors.append(f"Non-numeric features: {non_numeric}")

    feature_results = []

    for column in frame.columns:
        series = frame[column]

        missing_count = int(series.isna().sum())
        numeric = pd.to_numeric(series, errors="coerce")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)

        non_finite_count = int((~np.isfinite(values)).sum())
        finite_values = values[np.isfinite(values)]

        feature_errors = []

        if missing_count:
            feature_errors.append("missing_values")

        if non_finite_count:
            feature_errors.append("non_finite_values")

        feature_results.append({
            "feature": str(column),
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "non_finite_count": non_finite_count,
            "minimum": (
                float(finite_values.min())
                if finite_values.size else None
            ),
            "maximum": (
                float(finite_values.max())
                if finite_values.size else None
            ),
            "errors": feature_errors,
        })

    if any(item["missing_count"] for item in feature_results):
        errors.append("Dataset contains missing values.")

    if any(item["non_finite_count"] for item in feature_results):
        errors.append("Dataset contains NaN or infinite numeric values.")

    duplicate_count = int(frame.duplicated().sum())
    duplicate_rate = (
        duplicate_count / len(frame) if len(frame) else 0.0
    )

    if duplicate_rate > max_duplicate_rate:
        warnings.append(
            f"Duplicate-row rate {duplicate_rate:.2%} exceeds "
            f"the configured threshold {max_duplicate_rate:.2%}."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "row_count": int(len(frame)),
            "feature_count": int(frame.shape[1]),
            "duplicate_rows": duplicate_count,
            "duplicate_rate": round(duplicate_rate, 6),
            "missing_values": int(frame.isna().sum().sum()),
            "non_finite_values": int(
                sum(item["non_finite_count"] for item in feature_results)
            ),
        },
        "features": feature_results,
    }


def require_valid_dataset(report: dict) -> None:
    """Stop the pipeline when a dataset fails validation."""
    if not report.get("valid", False):
        errors = "; ".join(report.get("errors", []))
        raise ValueError(f"Dataset validation failed: {errors}")
