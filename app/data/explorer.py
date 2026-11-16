import csv
from io import BytesIO, StringIO
from typing import Any

import numpy as np
import pandas as pd

from app.config import (
    MAX_DATASET_COLUMNS,
    MAX_DATASET_ROWS,
    MAX_UPLOAD_SIZE_BYTES,
)

MAX_FILE_SIZE = MAX_UPLOAD_SIZE_BYTES
MAX_ROWS = MAX_DATASET_ROWS
MAX_COLUMNS = MAX_DATASET_COLUMNS
MAX_PREVIEW_ROWS = 100


class DatasetExplorerError(ValueError):
    """Raised when an uploaded dataset cannot be analyzed."""


def read_dataset_csv(
    filename: str | None,
    content: bytes,
) -> pd.DataFrame:
    """Read and validate a CSV uploaded by the user with strict security bounds."""

    if not filename or not filename.lower().endswith(".csv"):
        raise DatasetExplorerError(
            "Please upload a file with a .csv extension."
        )

    if not content:
        raise DatasetExplorerError("The uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise DatasetExplorerError("The maximum file size is 10 MB.")

    if b"\x00" in content[:65536]:
        raise DatasetExplorerError("The CSV file contains invalid null bytes.")

    # 1. Validate raw column headers before Pandas can mangle or rename duplicate columns
    try:
        header_text = content[:131072].decode("utf-8-sig", errors="replace")
        first_line = next((line for line in header_text.splitlines() if line.strip()), None)
        if first_line:
            reader = csv.reader(StringIO(first_line))
            raw_headers = next(reader, [])
            if any(not str(h).strip() for h in raw_headers):
                raise DatasetExplorerError("Column names must not be empty.")
            trimmed = [str(h).strip() for h in raw_headers]
            if len(trimmed) != len(set(trimmed)):
                raise DatasetExplorerError("Duplicate column names are not allowed.")
    except UnicodeDecodeError as exc:
        raise DatasetExplorerError("The CSV file encoding is invalid.") from exc

    # 2. Parse CSV with bounded row limit
    try:
        # Read one extra row so we can detect files exceeding MAX_ROWS.
        frame = pd.read_csv(
            BytesIO(content),
            nrows=MAX_ROWS + 1,
        )
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise DatasetExplorerError(
            "Could not parse this CSV. Check its format and encoding."
        ) from exc

    if frame.empty:
        raise DatasetExplorerError(
            "The CSV must contain at least one data row."
        )

    if len(frame) > MAX_ROWS:
        raise DatasetExplorerError(
            f"The dataset exceeds the {MAX_ROWS:,}-row limit."
        )

    if frame.shape[1] == 0:
        raise DatasetExplorerError(
            "The CSV must contain at least one column."
        )

    if frame.shape[1] > MAX_COLUMNS:
        raise DatasetExplorerError(
            f"The dataset exceeds the {MAX_COLUMNS}-column limit."
        )

    if frame.columns.duplicated().any():
        raise DatasetExplorerError(
            "Duplicate column names are not allowed."
        )

    if any(not str(column).strip() for column in frame.columns):
        raise DatasetExplorerError(
            "Column names must not be empty."
        )

    return frame


def _json_value(value: Any) -> Any:
    """Convert Pandas and NumPy values to JSON-compatible values."""

    if value is None or value is pd.NA:
        return None

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]

    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]

    if isinstance(value, (str, int, bool)):
        return value

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def profile_dataset(
    frame: pd.DataFrame,
    *,
    preview_rows: int = 10,
) -> dict[str, Any]:
    """Generate a dataset summary, feature statistics, and preview."""

    if not 1 <= preview_rows <= MAX_PREVIEW_ROWS:
        raise DatasetExplorerError(
            f"preview_rows must be between 1 and {MAX_PREVIEW_ROWS}."
        )

    row_count, column_count = frame.shape

    if row_count == 0 or column_count == 0:
        raise DatasetExplorerError(
            "The dataset must contain at least one row and one column."
        )

    missing_by_column = frame.isna().sum()
    duplicate_rows = int(frame.duplicated().sum())

    columns: list[dict[str, Any]] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    constant_columns: list[str] = []

    for column in frame.columns:
        series = frame[column]
        name = str(column)
        unique_count = int(series.nunique(dropna=True))
        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric:
            numeric_columns.append(name)
        else:
            categorical_columns.append(name)

        is_constant = unique_count <= 1

        if is_constant:
            constant_columns.append(name)

        result: dict[str, Any] = {
            "name": name,
            "dtype": str(series.dtype),
            "kind": "numeric" if is_numeric else "categorical",
            "missing_count": int(missing_by_column[column]),
            "missing_percentage": round(
                float(missing_by_column[column]) / row_count * 100,
                2,
            ),
            "unique_count": unique_count,
            "constant": is_constant,
        }

        if is_numeric:
            numeric = pd.to_numeric(series, errors="coerce")
            finite = numeric[np.isfinite(numeric)]

            result["statistics"] = {
                "count": int(finite.count()),
                "mean": _json_value(finite.mean()),
                "median": _json_value(finite.median()),
                "std": _json_value(finite.std()),
                "min": _json_value(finite.min()),
                "max": _json_value(finite.max()),
            }
        else:
            counts = series.astype("string").value_counts(dropna=True)

            result["top_values"] = [
                {
                    "value": str(value),
                    "count": int(count),
                }
                for value, count in counts.head(10).items()
            ]

        columns.append(result)

    numeric_frame = frame.select_dtypes(include="number")
    correlation = numeric_frame.corr().replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return {
        "dataset": {
            "row_count": int(row_count),
            "column_count": int(column_count),
            "memory_usage_bytes": int(
                frame.memory_usage(index=True, deep=True).sum()
            ),
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": round(
                duplicate_rows / row_count * 100,
                2,
            ),
            "missing_cells": int(missing_by_column.sum()),
            "missing_percentage": round(
                float(missing_by_column.sum())
                / (row_count * column_count)
                * 100,
                2,
            ),
            "numeric_column_count": len(numeric_columns),
            "categorical_column_count": len(categorical_columns),
            "constant_column_count": len(constant_columns),
        },
        "columns": columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "constant_columns": constant_columns,
        "correlation": {
            str(column): {
                str(other): _json_value(correlation.loc[column, other])
                for other in correlation.columns
            }
            for column in correlation.columns
        },
        "preview": {
            "columns": [str(column) for column in frame.columns],
            "rows": [
                [_json_value(value) for value in row]
                for row in frame.head(preview_rows).itertuples(
                    index=False,
                    name=None,
                )
            ],
        },
    }