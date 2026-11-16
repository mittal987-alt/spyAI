import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data.explorer import (
    DatasetExplorerError,
    profile_dataset,
    read_dataset_csv,
)
from app.main import app


client = TestClient(app)


def test_csv_reader_accepts_valid_csv():
    frame = read_dataset_csv(
        "sample.csv",
        b"age,score\n20,80\n21,90\n",
    )

    assert frame.shape == (2, 2)
    assert list(frame.columns) == ["age", "score"]


def test_csv_reader_rejects_non_csv_filename():
    with pytest.raises(DatasetExplorerError):
        read_dataset_csv("sample.txt", b"a,b\n1,2\n")


def test_csv_reader_rejects_empty_file():
    with pytest.raises(DatasetExplorerError):
        read_dataset_csv("sample.csv", b"")


def test_csv_reader_rejects_oversized_file():
    with pytest.raises(DatasetExplorerError):
        read_dataset_csv(
            "sample.csv",
            b"x" * (10 * 1024 * 1024 + 1),
        )


def test_profile_reports_missing_values_and_duplicates():
    frame = pd.DataFrame({
        "age": [20, 21, 21],
        "score": [80.0, None, None],
    })

    report = profile_dataset(frame)

    assert report["dataset"]["row_count"] == 3
    assert report["dataset"]["column_count"] == 2
    assert report["dataset"]["duplicate_rows"] == 1
    assert report["dataset"]["missing_cells"] == 2


def test_profile_detects_constant_columns():
    frame = pd.DataFrame({
        "constant": [5, 5, 5],
        "varying": [1, 2, 3],
    })

    report = profile_dataset(frame)

    assert report["constant_columns"] == ["constant"]


def test_profile_rejects_invalid_preview_size():
    frame = pd.DataFrame({"x": [1, 2, 3]})

    with pytest.raises(DatasetExplorerError):
        profile_dataset(frame, preview_rows=0)


def test_profile_api_returns_dataset_summary():
    response = client.post(
        "/api/v1/datasets/profile",
        files={
            "file": (
                "sample.csv",
                io.BytesIO(b"age,score\n20,80\n21,90\n"),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["dataset"]["row_count"] == 2
    assert body["dataset"]["column_count"] == 2
    assert "columns" in body
    assert "preview" in body


def test_profile_api_rejects_non_csv():
    response = client.post(
        "/api/v1/datasets/profile",
        files={
            "file": (
                "sample.txt",
                io.BytesIO(b"hello"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_report_api_returns_json_attachment():
    response = client.post(
        "/api/v1/datasets/report",
        files={
            "file": (
                "sample.csv",
                io.BytesIO(b"age,score\n20,80\n21,90\n"),
                "text/csv",
            )
        },
    )

    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert response.json()["dataset"]["row_count"] == 2