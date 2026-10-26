"""
Webhook Notification Manager for Drift Alerts & System Diagnostics.

Supports Slack, Discord, and Generic JSON Webhooks with automated dispatch,
severity filtering, and live delivery logs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ALERTS_DB_PATH

logger = logging.getLogger(__name__)

VALID_SERVICES = {"slack", "discord", "generic"}
VALID_SEVERITIES = {"low", "medium", "high"}
SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}


def initialize_webhook_tables(db_path: str | Path = ALERTS_DB_PATH) -> None:
    """Initialize webhooks and webhook_logs tables in the SQLite database."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS webhooks (
                webhook_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                url TEXT NOT NULL,
                service_type TEXT NOT NULL,
                min_severity TEXT NOT NULL,
                is_enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS webhook_logs (
                log_id TEXT PRIMARY KEY,
                webhook_id TEXT NOT NULL,
                webhook_name TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                status_code INTEGER,
                success INTEGER NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_webhook_logs_created
            ON webhook_logs(created_at)
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_webhook(
    name: str,
    url: str,
    service_type: str = "generic",
    min_severity: str = "medium",
    db_path: str | Path = ALERTS_DB_PATH,
) -> dict[str, Any]:
    """Register a new webhook endpoint."""
    initialize_webhook_tables(db_path)

    name = name.strip()
    url = url.strip()
    service_type = service_type.strip().lower()
    min_severity = min_severity.strip().lower()

    if not name:
        raise ValueError("Webhook name must not be empty.")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("Webhook URL must start with http:// or https://")
    if service_type not in VALID_SERVICES:
        raise ValueError(f"service_type must be one of: {VALID_SERVICES}")
    if min_severity not in VALID_SEVERITIES:
        raise ValueError(f"min_severity must be one of: {VALID_SEVERITIES}")

    webhook_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO webhooks (
                webhook_id, name, url, service_type, min_severity, is_enabled, created_at
            )
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (webhook_id, name, url, service_type, min_severity, created_at),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "webhook_id": webhook_id,
        "name": name,
        "url": url,
        "service_type": service_type,
        "min_severity": min_severity,
        "is_enabled": True,
        "created_at": created_at,
    }


def list_webhooks(db_path: str | Path = ALERTS_DB_PATH) -> list[dict[str, Any]]:
    """List all configured webhooks."""
    initialize_webhook_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT webhook_id, name, url, service_type, min_severity, is_enabled, created_at
            FROM webhooks
            ORDER BY created_at DESC
            """
        )
        return [
            {
                "webhook_id": str(row["webhook_id"]),
                "name": str(row["name"]),
                "url": str(row["url"]),
                "service_type": str(row["service_type"]),
                "min_severity": str(row["min_severity"]),
                "is_enabled": bool(row["is_enabled"]),
                "created_at": str(row["created_at"]),
            }
            for row in cursor.fetchall()
        ]
    finally:
        conn.close()


def toggle_webhook(
    webhook_id: str,
    is_enabled: bool,
    db_path: str | Path = ALERTS_DB_PATH,
) -> bool:
    """Enable or disable a webhook."""
    initialize_webhook_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "UPDATE webhooks SET is_enabled = ? WHERE webhook_id = ?",
            (1 if is_enabled else 0, webhook_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_webhook(webhook_id: str, db_path: str | Path = ALERTS_DB_PATH) -> bool:
    """Delete a webhook by ID."""
    initialize_webhook_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            "DELETE FROM webhooks WHERE webhook_id = ?",
            (webhook_id,),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()



def format_webhook_payload(
    service_type: str,
    alert: dict[str, Any],
) -> dict[str, Any]:
    """Format payload for Slack, Discord, or Generic webhooks."""
    title = alert.get("title", "ModelAutopsy Alert")
    message = alert.get("message", "")
    severity = str(alert.get("severity", "medium")).lower()
    feature = alert.get("feature", "N/A")
    effect_size = alert.get("effect_size", 0.0)
    adjusted_p_value = alert.get("adjusted_p_value", 0.0)

    # Color codes
    color_hex = 0x10B981 if severity == "low" else (0xF59E0B if severity == "medium" else 0xE11D48)
    color_str = "#10B981" if severity == "low" else ("#F59E0B" if severity == "medium" else "#E11D48")

    if service_type == "slack":
        return {
            "text": f"[{severity.upper()}] {title}",
            "attachments": [
                {
                    "color": color_str,
                    "title": title,
                    "text": message,
                    "fields": [
                        {"title": "Feature", "value": str(feature), "short": True},
                        {"title": "Severity", "value": severity.upper(), "short": True},
                        {"title": "Effect Size", "value": f"{effect_size:.4f}", "short": True},
                        {"title": "Adjusted p-value", "value": f"{adjusted_p_value:.6g}", "short": True},
                    ],
                    "footer": "ModelAutopsy AI Platform",
                    "ts": int(datetime.now(timezone.utc).timestamp()),
                }
            ],
        }

    if service_type == "discord":
        return {
            "content": f"🚨 **ModelAutopsy AI Alert** ({severity.upper()})",
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": color_hex,
                    "fields": [
                        {"name": "Feature", "value": str(feature), "inline": True},
                        {"name": "Severity", "value": severity.upper(), "inline": True},
                        {"name": "Effect Size", "value": f"{effect_size:.4f}", "inline": True},
                        {"name": "Adjusted p-val", "value": f"{adjusted_p_value:.6g}", "inline": True},
                    ],
                    "footer": {"text": "ModelAutopsy AI Drift Monitoring"},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }

    # Default generic JSON
    return {
        "event": "model_drift_alert",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "alert": alert,
    }


def deliver_webhook(
    url: str,
    payload: dict[str, Any],
    timeout: float = 6.0,
) -> tuple[bool, int, str]:
    """Execute an HTTP POST request to deliver the webhook payload."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ModelAutopsy-AI-Webhook/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8", errors="ignore")[:300]
            return True, status_code, body
    except urllib.error.HTTPError as err:
        return False, err.code, str(err.reason)
    except urllib.error.URLError as err:
        return False, 0, str(err.reason)
    except Exception as exc:
        return False, 0, str(exc)


def log_webhook_delivery(
    webhook_id: str,
    webhook_name: str,
    event_type: str,
    severity: str,
    status_code: int,
    success: bool,
    message: str,
    db_path: str | Path = ALERTS_DB_PATH,
) -> None:
    """Record webhook delivery attempt."""
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO webhook_logs (
                    log_id, webhook_id, webhook_name, event_type, severity,
                    status_code, success, message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    webhook_id,
                    webhook_name,
                    event_type,
                    severity,
                    status_code,
                    1 if success else 0,
                    message[:400],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Could not write webhook delivery log: %s", exc)


def list_webhook_logs(
    limit: int = 50,
    db_path: str | Path = ALERTS_DB_PATH,
) -> list[dict[str, Any]]:
    """Fetch recent webhook delivery logs."""
    initialize_webhook_tables(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT log_id, webhook_id, webhook_name, event_type, severity,
                   status_code, success, message, created_at
            FROM webhook_logs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        )
        return [
            {
                "log_id": str(row["log_id"]),
                "webhook_id": str(row["webhook_id"]),
                "webhook_name": str(row["webhook_name"]),
                "event_type": str(row["event_type"]),
                "severity": str(row["severity"]),
                "status_code": row["status_code"],
                "success": bool(row["success"]),
                "message": str(row["message"] or ""),
                "created_at": str(row["created_at"]),
            }
            for row in cursor.fetchall()
        ]
    finally:
        conn.close()


def dispatch_alerts_to_webhooks(
    alerts: list[dict[str, Any]],
    db_path: str | Path = ALERTS_DB_PATH,
) -> int:
    """
    Deliver alerts to all matching active webhooks based on minimum severity.
    Returns the number of successful deliveries.
    """
    if not alerts:
        return 0

    webhooks = list_webhooks(db_path)
    active_webhooks = [w for w in webhooks if w["is_enabled"]]
    if not active_webhooks:
        return 0

    successful_deliveries = 0

    for alert in alerts:
        alert_sev = str(alert.get("severity", "low")).lower()
        alert_rank = SEVERITY_ORDER.get(alert_sev, 1)

        for webhook in active_webhooks:
            min_rank = SEVERITY_ORDER.get(webhook["min_severity"], 2)
            if alert_rank >= min_rank:
                payload = format_webhook_payload(webhook["service_type"], alert)
                success, code, msg = deliver_webhook(webhook["url"], payload)
                log_webhook_delivery(
                    webhook_id=webhook["webhook_id"],
                    webhook_name=webhook["name"],
                    event_type="drift_alert",
                    severity=alert_sev,
                    status_code=code,
                    success=success,
                    message=msg,
                    db_path=db_path,
                )
                if success:
                    successful_deliveries += 1

    return successful_deliveries


def test_webhook_connection(
    url: str,
    service_type: str = "generic",
    name: str = "Test Webhook",
) -> dict[str, Any]:
    """Dispatch a test ping alert to verify connectivity."""
    sample_alert = {
        "title": "ModelAutopsy AI — Test Notification",
        "message": "This is a test notification verifying your webhook integration with ModelAutopsy AI platform.",
        "severity": "medium",
        "feature": "latency_ms",
        "effect_size": 0.1425,
        "adjusted_p_value": 0.0034,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    payload = format_webhook_payload(service_type, sample_alert)
    success, status_code, response_body = deliver_webhook(url, payload, timeout=7.0)

    return {
        "success": success,
        "status_code": status_code,
        "message": response_body if response_body else ("OK" if success else "Failed to connect to webhook URL"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
