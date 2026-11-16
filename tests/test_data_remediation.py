import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data.remediation import (
    apply_dataset_remediation,
    detect_dataset_outliers,
    export_dataframe_to_csv_bytes,
    generate_remediation_plan,
)
from app.main import app

client = TestClient(app)


def test_detect_dataset_outliers_finds_iqr_and_z_outliers():
    # 20 standard values and 2 extreme outliers
    data = {"val": [10.0] * 20 + [1000.0, -500.0], "cat": ["A"] * 22}
    df = pd.DataFrame(data)

    outliers = detect_dataset_outliers(df)
    assert outliers["has_numeric_data"] is True
    assert outliers["outlier_rows_count"] >= 2
    assert "val" in outliers["feature_outliers"]
    assert outliers["feature_outliers"]["val"]["iqr_outliers_count"] >= 2


def test_generate_remediation_plan_identifies_duplicates_missing_constants():
    df = pd.DataFrame({
        "num": [1.0, 2.0, None, 1.0],
        "constant": [99, 99, 99, 99],
        "cat": ["X", "Y", "X", "X"],
    })

    plan = generate_remediation_plan(df)
    action_types = [a["type"] for a in plan["actions"]]

    assert "imputation" in action_types
    assert "pruning" in action_types  # constant column


def test_apply_dataset_remediation_imputes_and_prunes():
    df = pd.DataFrame({
        "num": [10.0, 20.0, None, 20.0],
        "constant": [1, 1, 1, 1],
        "cat": ["Apple", None, "Banana", "Banana"],
    })

    cleaned, report = apply_dataset_remediation(
        df,
        drop_duplicates=False,
        drop_constant_columns=True,
        impute_numeric_strategy="median",
        impute_categorical_strategy="mode",
    )

    # Constant column should be dropped
    assert "constant" not in cleaned.columns
    # Missing cells should be 0
    assert cleaned.isna().sum().sum() == 0
    assert report["cleaned"]["missing_cells"] == 0
    assert len(report["dropped_constant_columns"]) == 1


def test_apply_dataset_remediation_clips_outliers():
    df = pd.DataFrame({
        "val": [10.0, 11.0, 10.5, 9.5, 10.2, 10.1, 9.9, 10.0, 10.3, 100.0]
    })

    cleaned, report = apply_dataset_remediation(
        df,
        clip_outliers=True,
        iqr_multiplier=1.5,
    )

    # Extreme value 100.0 should be clipped
    assert cleaned["val"].max() < 50.0
    assert any("Clipped" in change for change in report["changes_made"])


def test_remediation_api_endpoints():
    csv_bytes = b"age,income,const,gender\n25,50000,1,M\n30,,1,F\n35,70000,1,F\n25,50000,1,M\n"

    # 1. Outlier detection endpoint
    res_out = client.post(
        "/api/v1/datasets/outliers",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert res_out.status_code == 200
    assert "outlier_percentage" in res_out.json()

    # 2. Remediation plan endpoint
    res_plan = client.post(
        "/api/v1/datasets/remediation-plan",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert res_plan.status_code == 200
    assert "actions" in res_plan.json()

    # 3. Apply remediation endpoint
    res_clean = client.post(
        "/api/v1/datasets/remediate",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"drop_duplicates": "true", "drop_constant_columns": "true"},
    )
    assert res_clean.status_code == 200
    clean_data = res_clean.json()
    assert clean_data["report"]["cleaned"]["missing_cells"] == 0
    assert "preview" in clean_data

    # 4. Export cleaned CSV endpoint
    res_export = client.post(
        "/api/v1/datasets/remediate/export",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"drop_duplicates": "true", "drop_constant_columns": "true"},
    )
    assert res_export.status_code == 200
    assert "attachment" in res_export.headers.get("content-disposition", "")
    assert "const" not in res_export.text  # constant column dropped
