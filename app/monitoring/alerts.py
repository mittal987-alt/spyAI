
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ALERTS_DB_PATH


def initialize_alert_db(
    db_path: str | Path = ALERTS_DB_PATH,
) -> None:
    """Create the alert database and table if they do not exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                experiment TEXT NOT NULL,
                run_id TEXT,
                feature TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                effect_size REAL NOT NULL,
                adjusted_p_value REAL NOT NULL,
                created_at TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_alerts_created_at
            ON alerts(created_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_alerts_severity
            ON alerts(severity)
            """
        )


def generate_drift_alerts(
    drift_results: list[dict[str, Any]],
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Convert detected feature drift into actionable alerts."""
    alerts = []

    for result in drift_results:
        if not result.get("drift_detected", False):
            continue

        feature = str(result["feature"])
        severity = str(result["severity"]).lower()

        if severity not in {"low", "medium", "high"}:
            continue

        effect_size = float(result["effect_size"])
        adjusted_p_value = float(result["adjusted_p_value"])

        if severity == "high":
            recommendation = (
                "Investigate the feature pipeline, data source, and "
                "recent upstream changes before relying on model predictions."
            )
        elif severity == "medium":
            recommendation = (
                "Review recent feature distributions and compare them "
                "with the reference dataset."
            )
        else:
            recommendation = (
                "Monitor this feature and investigate if drift persists "
                "or model performance declines."
            )

        alert = {
            "alert_id": str(uuid.uuid4()),
            "experiment": "synthetic_feature_drift",
            "run_id": run_id,
            "feature": feature,
            "severity": severity,
            "title": f"{severity.upper()} drift detected: {feature}",
            "message": (
                f"Feature '{feature}' differs from its reference "
                f"distribution. KS effect size={effect_size:.4f}; "
                f"adjusted p-value={adjusted_p_value:.6g}. "
                f"{recommendation}"
            ),
            "effect_size": effect_size,
            "adjusted_p_value": adjusted_p_value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "details": result,
            "recommendation": recommendation,
        }

        alerts.append(alert)

    return alerts


def save_alerts(
    alerts: list[dict[str, Any]],
    db_path: str | Path = ALERTS_DB_PATH,
) -> int:
    """Persist alerts. Returns the number of inserted alerts."""
    initialize_alert_db(db_path)
    path = Path(db_path)
    inserted = 0

    with sqlite3.connect(path) as connection:
        for alert in alerts:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO alerts (
                    alert_id,
                    experiment,
                    run_id,
                    feature,
                    severity,
                    title,
                    message,
                    effect_size,
                    adjusted_p_value,
                    created_at,
                    details_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert["alert_id"],
                    alert["experiment"],
                    alert.get("run_id"),
                    alert["feature"],
                    alert["severity"],
                    alert["title"],
                    alert["message"],
                    alert["effect_size"],
                    alert["adjusted_p_value"],
                    alert["created_at"],
                    json.dumps(alert, default=str),
                ),
            )
            inserted += cursor.rowcount

    return inserted


def list_alerts(
    limit: int = 50,
    severity: str | None = None,
    db_path: str | Path = ALERTS_DB_PATH,
) -> list[dict[str, Any]]:
    """Return the newest alerts, optionally filtered by severity."""
    if not 1 <= limit <= 500:
        raise ValueError("limit must be between 1 and 500")

    if severity is not None:
        severity = severity.lower()
        if severity not in {"low", "medium", "high"}:
            raise ValueError("severity must be low, medium, or high")

    initialize_alert_db(db_path)

    query = """
        SELECT alert_id, experiment, run_id, feature, severity,
               title, message, effect_size, adjusted_p_value, created_at,
               details_json
        FROM alerts
    """
    parameters: list[Any] = []

    if severity is not None:
        query += " WHERE severity = ?"
        parameters.append(severity)

    query += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
    parameters.append(limit)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, parameters).fetchall()

    results = []
    for row in rows:
        item = dict(row)
        item["details"] = json.loads(item.pop("details_json"))
        results.append(item)

    return results


def get_alert(
    alert_id: str,
    db_path: str | Path = ALERTS_DB_PATH,
) -> dict[str, Any] | None:
    """Return one alert by ID, or None if it does not exist."""
    initialize_alert_db(db_path)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT alert_id, experiment, run_id, feature, severity,
                   title, message, effect_size, adjusted_p_value,
                   created_at, details_json
            FROM alerts
            WHERE alert_id = ?
            """,
            (alert_id,),
        ).fetchone()

    if row is None:
        return None

    result = dict(row)
    result["details"] = json.loads(result.pop("details_json"))
    return result
