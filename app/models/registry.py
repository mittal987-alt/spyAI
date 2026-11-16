"""
Model Registry & Lifecycle Management for ModelAutopsy AI.

Allows versioning, tagging, lifecycle stage transitions (development -> staging -> production -> archived),
and head-to-head metric comparisons.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

REGISTRY_DB_PATH = PROJECT_ROOT / "model_registry.db"
VALID_STAGES = {"development", "staging", "production", "archived"}


def initialize_registry_db(db_path: str | Path = REGISTRY_DB_PATH) -> None:
    """Initialize SQLite tables for model registry."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS registered_models (
                model_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                stage TEXT NOT NULL DEFAULT 'development',
                framework TEXT NOT NULL DEFAULT 'sklearn',
                description TEXT NOT NULL DEFAULT '',
                artifact_uri TEXT NOT NULL DEFAULT '',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                parameters_json TEXT NOT NULL DEFAULT '{}',
                tags_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_name_ver
            ON registered_models(name, version)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_stage
            ON registered_models(stage)
            """
        )


def register_model(
    name: str,
    version: str,
    stage: str = "development",
    framework: str = "sklearn",
    description: str = "",
    artifact_uri: str = "",
    metrics: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    db_path: str | Path = REGISTRY_DB_PATH,
) -> dict[str, Any]:
    """Register a new model or model version."""
    initialize_registry_db(db_path)

    name = name.strip()
    version = version.strip()
    stage = stage.strip().lower()

    if not name:
        raise ValueError("Model name must not be empty.")
    if not version:
        raise ValueError("Model version must not be empty.")
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of: {sorted(VALID_STAGES)}")

    model_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    metrics_str = json.dumps(metrics or {}, default=str)
    params_str = json.dumps(parameters or {}, default=str)
    tags_str = json.dumps(tags or [], default=str)

    with sqlite3.connect(db_path) as conn:
        # Check if version exists for this model
        existing = conn.execute(
            "SELECT model_id FROM registered_models WHERE name = ? AND version = ?",
            (name, version),
        ).fetchone()
        if existing:
            raise ValueError(f"Model '{name}' with version '{version}' is already registered.")

        # If promoting to production, downgrade any current production version for this model name
        if stage == "production":
            conn.execute(
                "UPDATE registered_models SET stage = 'archived', updated_at = ? WHERE name = ? AND stage = 'production'",
                (now, name),
            )

        conn.execute(
            """
            INSERT INTO registered_models
                (model_id, name, version, stage, framework, description,
                 artifact_uri, metrics_json, parameters_json, tags_json,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                name,
                version,
                stage,
                framework,
                description,
                artifact_uri,
                metrics_str,
                params_str,
                tags_str,
                now,
                now,
            ),
        )

    return get_registered_model(model_id, db_path=db_path)  # type: ignore[return-value]


def list_registered_models(
    name: str | None = None,
    stage: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db_path: str | Path = REGISTRY_DB_PATH,
) -> list[dict[str, Any]]:
    """List registered models with optional filtering."""
    initialize_registry_db(db_path)

    query = "SELECT * FROM registered_models"
    clauses: list[str] = []
    params: list[Any] = []

    if name:
        clauses.append("name = ?")
        params.append(name.strip())
    if stage:
        clauses.append("stage = ?")
        params.append(stage.strip().lower())

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    return [_deserialize_model(dict(r)) for r in rows]


def get_registered_model(
    model_id: str,
    db_path: str | Path = REGISTRY_DB_PATH,
) -> dict[str, Any] | None:
    """Retrieve a registered model by ID."""
    initialize_registry_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM registered_models WHERE model_id = ?",
            (model_id,),
        ).fetchone()

    return _deserialize_model(dict(row)) if row else None


def update_model_stage(
    model_id: str,
    new_stage: str,
    db_path: str | Path = REGISTRY_DB_PATH,
) -> dict[str, Any]:
    """Promote or transition a model's stage."""
    initialize_registry_db(db_path)
    new_stage = new_stage.strip().lower()
    if new_stage not in VALID_STAGES:
        raise ValueError(f"new_stage must be one of: {sorted(VALID_STAGES)}")

    model = get_registered_model(model_id, db_path=db_path)
    if not model:
        raise ValueError(f"Model ID '{model_id}' not found.")

    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as conn:
        # If promoting to production, archive any prior production model of the same name
        if new_stage == "production":
            conn.execute(
                "UPDATE registered_models SET stage = 'archived', updated_at = ? WHERE name = ? AND stage = 'production' AND model_id != ?",
                (now, model["name"], model_id),
            )

        conn.execute(
            "UPDATE registered_models SET stage = ?, updated_at = ? WHERE model_id = ?",
            (new_stage, now, model_id),
        )

    return get_registered_model(model_id, db_path=db_path)  # type: ignore[return-value]


def delete_registered_model(
    model_id: str,
    db_path: str | Path = REGISTRY_DB_PATH,
) -> bool:
    """Delete a registered model."""
    initialize_registry_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("DELETE FROM registered_models WHERE model_id = ?", (model_id,))
        return cursor.rowcount > 0


def compare_registered_models(
    baseline_id: str,
    candidate_id: str,
    db_path: str | Path = REGISTRY_DB_PATH,
) -> dict[str, Any]:
    """Compare performance metrics between two model versions."""
    base = get_registered_model(baseline_id, db_path=db_path)
    cand = get_registered_model(candidate_id, db_path=db_path)

    if not base:
        raise ValueError(f"Baseline model ID '{baseline_id}' not found.")
    if not cand:
        raise ValueError(f"Candidate model ID '{candidate_id}' not found.")

    base_metrics = base.get("metrics", {})
    cand_metrics = cand.get("metrics", {})

    all_keys = set(base_metrics.keys()) | set(cand_metrics.keys())
    deltas = {}

    for k in sorted(all_keys):
        b_val = base_metrics.get(k)
        c_val = cand_metrics.get(k)
        if isinstance(b_val, (int, float)) and isinstance(c_val, (int, float)):
            diff = round(float(c_val) - float(b_val), 4)
            deltas[k] = {
                "baseline": b_val,
                "candidate": c_val,
                "delta": diff,
                "improved": diff > 0,
            }
        else:
            deltas[k] = {
                "baseline": b_val,
                "candidate": c_val,
                "delta": None,
                "improved": None,
            }

    return {
        "baseline": {
            "model_id": base["model_id"],
            "name": base["name"],
            "version": base["version"],
            "stage": base["stage"],
        },
        "candidate": {
            "model_id": cand["model_id"],
            "name": cand["name"],
            "version": cand["version"],
            "stage": cand["stage"],
        },
        "metric_comparisons": deltas,
    }


def _deserialize_model(row: dict[str, Any]) -> dict[str, Any]:
    row["metrics"] = json.loads(row.pop("metrics_json", "{}"))
    row["parameters"] = json.loads(row.pop("parameters_json", "{}"))
    row["tags"] = json.loads(row.pop("tags_json", "[]"))
    return row
