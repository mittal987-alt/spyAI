
import numpy as np
import pytest

from app.validation.data_quality import (
    validate_dataset,
    require_valid_dataset,
)


def test_valid_dataset_passes():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    report = validate_dataset(X, ["a", "b"])

    assert report["valid"] is True
    assert report["errors"] == []
    assert report["summary"]["row_count"] == 3


def test_missing_values_fail_validation():
    X = np.array([
        [1.0, np.nan],
        [2.0, 3.0],
    ])

    report = validate_dataset(X, ["a", "b"])

    assert report["valid"] is False
    assert any("missing values" in e for e in report["errors"])


def test_infinite_values_fail_validation():
    X = np.array([
        [1.0, 2.0],
        [3.0, np.inf],
    ])

    report = validate_dataset(X, ["a", "b"])

    assert report["valid"] is False
    assert any("infinite" in e for e in report["errors"])


def test_duplicate_rows_generate_warning():
    X = np.array([
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    report = validate_dataset(
        X,
        ["a", "b"],
        max_duplicate_rate=0.10,
    )

    assert report["valid"] is True
    assert report["summary"]["duplicate_rows"] == 2
    assert report["warnings"]


def test_wrong_feature_count_is_rejected():
    X = np.ones((4, 2))

    report = validate_dataset(
        X,
        ["a", "b"],
        expected_features=3,
    )

    assert report["valid"] is False


def test_invalid_report_raises():
    with pytest.raises(ValueError, match="validation failed"):
        require_valid_dataset({
            "valid": False,
            "errors": ["Dataset must not be empty."],
        })


def test_empty_dataset_is_rejected():
    report = validate_dataset(
        np.empty((0, 2)),
        ["a", "b"],
    )

    assert report["valid"] is False
