"""
Smart Outlier Detection & Data Remediation Engine.

Provides multi-method outlier detection (Z-Score, IQR, Isolation Forest),
automated data quality diagnostics, intelligent imputation recommendations,
and sanitized dataset generation/export.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_dataset_outliers(
    df: pd.DataFrame,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> dict[str, Any]:
    """
    Detect outliers using statistical (Z-Score, IQR) and ensemble (Isolation Forest) methods.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna(how="all")
    if numeric_df.empty or len(numeric_df) < 5:
        return {
            "has_numeric_data": False,
            "total_rows": len(df),
            "outlier_rows_count": 0,
            "outlier_percentage": 0.0,
            "feature_outliers": {},
            "anomaly_indices": [],
        }

    total_rows = len(df)
    feature_outliers: dict[str, dict[str, Any]] = {}
    iqr_flagged_indices: set[int] = set()
    z_flagged_indices: set[int] = set()

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.nunique() <= 1 or len(series) < 5:
            continue

        # 1. IQR Method
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - (iqr_multiplier * iqr)
        upper_bound = q75 + (iqr_multiplier * iqr)

        iqr_mask = (series < lower_bound) | (series > upper_bound)
        iqr_outliers = series[iqr_mask]
        iqr_count = int(iqr_mask.sum())
        iqr_flagged_indices.update(iqr_outliers.index.tolist())

        # 2. Z-Score Method
        mean = float(series.mean())
        std = float(series.std())
        z_count = 0
        if std > 1e-9:
            z_scores = np.abs((series - mean) / std)
            z_mask = z_scores > z_threshold
            z_count = int(z_mask.sum())
            z_flagged_indices.update(series[z_mask].index.tolist())

        feature_outliers[str(col)] = {
            "iqr_outliers_count": iqr_count,
            "iqr_lower_bound": round(lower_bound, 4),
            "iqr_upper_bound": round(upper_bound, 4),
            "z_outliers_count": z_count,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "outlier_rate": round((iqr_count / len(series)) * 100, 2),
        }

    # 3. Multivariate Isolation Forest
    clean_numeric = numeric_df.fillna(numeric_df.median())
    isolation_forest_count = 0
    isolation_indices: list[int] = []

    if clean_numeric.shape[1] >= 1 and len(clean_numeric) >= 10:
        try:
            iso = IsolationForest(
                contamination=0.05,
                random_state=42,
                n_estimators=50,
            )
            predictions = iso.fit_predict(clean_numeric)
            iso_mask = predictions == -1
            isolation_forest_count = int(iso_mask.sum())
            isolation_indices = clean_numeric.index[iso_mask].tolist()
        except Exception:
            pass

    all_outlier_indices = sorted(list(iqr_flagged_indices | set(isolation_indices)))

    return {
        "has_numeric_data": True,
        "total_rows": total_rows,
        "outlier_rows_count": len(all_outlier_indices),
        "outlier_percentage": round((len(all_outlier_indices) / total_rows) * 100, 2) if total_rows > 0 else 0.0,
        "isolation_forest_anomalies": isolation_forest_count,
        "feature_outliers": feature_outliers,
        "sample_anomaly_indices": all_outlier_indices[:50],
    }


def generate_remediation_plan(df: pd.DataFrame) -> dict[str, Any]:
    """
    Diagnose dataset quality issues and generate actionable remediation recommendations.
    """
    total_rows, total_cols = df.shape
    actions = []

    # 1. Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = round((dup_count / total_rows) * 100, 2)
        actions.append({
            "action_id": "drop_duplicates",
            "type": "deduplication",
            "severity": "medium" if dup_pct > 5 else "low",
            "title": f"Remove {dup_count:,} duplicate rows",
            "description": f"{dup_pct}% of the dataset consists of redundant identical rows.",
            "recommended": True,
            "target": "rows",
            "count": dup_count,
        })

    # 2. Missing values
    missing_by_col = df.isna().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0]
    if len(cols_with_missing) > 0:
        for col_name, count in cols_with_missing.items():
            pct = round((count / total_rows) * 100, 2)
            is_num = pd.api.types.is_numeric_dtype(df[col_name])
            strategy = "median" if is_num else "mode"
            actions.append({
                "action_id": f"impute_{col_name}",
                "type": "imputation",
                "severity": "high" if pct > 20 else "medium",
                "title": f"Impute missing values in '{col_name}'",
                "description": f"{count:,} cells ({pct}%) missing. Recommended strategy: {strategy}.",
                "recommended": True,
                "target": str(col_name),
                "strategy": strategy,
                "missing_count": int(count),
                "missing_pct": pct,
            })

    # 3. Constant columns (zero variance)
    constant_cols = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if constant_cols:
        actions.append({
            "action_id": "drop_constant_columns",
            "type": "pruning",
            "severity": "medium",
            "title": f"Drop {len(constant_cols)} constant column(s)",
            "description": f"Columns {constant_cols} provide zero predictive variance and inflate model dimensionality.",
            "recommended": True,
            "target": "columns",
            "columns": constant_cols,
        })

    # 4. Multicollinearity (High Correlation)
    numeric_df = df.select_dtypes(include=[np.number])
    collinear_pairs = []
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr().abs()
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                r_val = corr_matrix.iloc[i, j]
                if r_val > 0.90 and np.isfinite(r_val):
                    c1, c2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    collinear_pairs.append({
                        "feature_a": str(c1),
                        "feature_b": str(c2),
                        "correlation": round(float(r_val), 4),
                    })

    if collinear_pairs:
        actions.append({
            "action_id": "prune_collinear_features",
            "type": "feature_selection",
            "severity": "low",
            "title": f"Found {len(collinear_pairs)} highly collinear feature pairs (|r| > 0.90)",
            "description": "Redundant feature pairs can distort model interpretability and gradient weights.",
            "recommended": False,
            "pairs": collinear_pairs,
        })

    # 5. Outliers
    outlier_info = detect_dataset_outliers(df)
    if outlier_info["outlier_rows_count"] > 0:
        actions.append({
            "action_id": "clip_outliers",
            "type": "outlier_capping",
            "severity": "low",
            "title": f"Clip extreme numeric outliers (IQR bounds)",
            "description": f"{outlier_info['outlier_rows_count']:,} rows ({outlier_info['outlier_percentage']}%) contain values outside 1.5x IQR boundaries.",
            "recommended": False,
            "outlier_count": outlier_info["outlier_rows_count"],
        })

    return {
        "dataset_shape": {"rows": total_rows, "columns": total_cols},
        "total_actions_recommended": len(actions),
        "actions": actions,
        "outlier_summary": outlier_info,
    }


def apply_dataset_remediation(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    drop_constant_columns: bool = True,
    impute_numeric_strategy: str = "median",  # "median" | "mean" | "zero" | "none"
    impute_categorical_strategy: str = "mode",  # "mode" | "missing_token" | "none"
    clip_outliers: bool = False,
    iqr_multiplier: float = 1.5,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Apply chosen data cleaning and remediation transformations to a DataFrame.
    Returns (cleaned_df, remediation_report).
    """
    initial_rows, initial_cols = df.shape
    initial_missing = int(df.isna().sum().sum())
    cleaned = df.copy()
    changes_made: list[str] = []

    # 1. Deduplication
    if drop_duplicates:
        dups_before = int(cleaned.duplicated().sum())
        if dups_before > 0:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            changes_made.append(f"Dropped {dups_before:,} duplicate rows.")

    # 2. Drop constant columns
    dropped_cols = []
    if drop_constant_columns:
        for col in list(cleaned.columns):
            if cleaned[col].nunique(dropna=True) <= 1:
                dropped_cols.append(str(col))
                cleaned = cleaned.drop(columns=[col])
        if dropped_cols:
            changes_made.append(f"Dropped {len(dropped_cols)} constant column(s): {', '.join(dropped_cols)}.")

    # 3. Impute numeric features
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    num_imputed = 0
    if impute_numeric_strategy != "none":
        for col in numeric_cols:
            missing_count = int(cleaned[col].isna().sum())
            if missing_count > 0:
                if impute_numeric_strategy == "median":
                    fill_val = cleaned[col].median()
                elif impute_numeric_strategy == "mean":
                    fill_val = cleaned[col].mean()
                else:
                    fill_val = 0.0

                if pd.notna(fill_val):
                    cleaned[col] = cleaned[col].fillna(fill_val)
                    num_imputed += missing_count

        if num_imputed > 0:
            changes_made.append(f"Imputed {num_imputed:,} numeric missing values using {impute_numeric_strategy}.")

    # 4. Impute categorical features
    cat_cols = cleaned.select_dtypes(exclude=[np.number]).columns
    cat_imputed = 0
    if impute_categorical_strategy != "none":
        for col in cat_cols:
            missing_count = int(cleaned[col].isna().sum())
            if missing_count > 0:
                if impute_categorical_strategy == "mode":
                    mode_series = cleaned[col].mode(dropna=True)
                    fill_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                else:
                    fill_val = "Missing"

                cleaned[col] = cleaned[col].fillna(fill_val)
                cat_imputed += missing_count

        if cat_imputed > 0:
            changes_made.append(f"Imputed {cat_imputed:,} categorical missing values using {impute_categorical_strategy}.")

    # 5. Clip numeric outliers
    clipped_count = 0
    if clip_outliers:
        for col in cleaned.select_dtypes(include=[np.number]).columns:
            series = cleaned[col]
            if series.nunique() > 1:
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lower = q25 - (iqr_multiplier * iqr)
                upper = q75 + (iqr_multiplier * iqr)
                was_outlier = (series < lower) | (series > upper)
                clipped_count += int(was_outlier.sum())
                cleaned[col] = series.clip(lower=lower, upper=upper)

        if clipped_count > 0:
            changes_made.append(f"Clipped {clipped_count:,} numeric outlier values to {iqr_multiplier}x IQR bounds.")

    final_rows, final_cols = cleaned.shape
    final_missing = int(cleaned.isna().sum().sum())

    report = {
        "initial": {
            "rows": initial_rows,
            "columns": initial_cols,
            "missing_cells": initial_missing,
        },
        "cleaned": {
            "rows": final_rows,
            "columns": final_cols,
            "missing_cells": final_missing,
            "rows_removed": initial_rows - final_rows,
            "columns_removed": initial_cols - final_cols,
            "missing_resolved": initial_missing - final_missing,
        },
        "dropped_constant_columns": dropped_cols,
        "changes_made": changes_made,
    }

    return cleaned, report


def export_dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to CSV bytes for client download."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
