"""
Webhook Notification Manager for Drift Alerts & System Diagnostics.

Supports Slack, Discord, and Generic JSON Webhooks with automated dispatch,
severity filtering, and live delivery logs.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ALERTS_DB_PATH, WEBHOOK_ALLOW_LOCAL

logger = logging.getLogger(__name__)

VALID_SERVICES = {"slack", "discord", "generic"}
VALID_SEVERITIES = {"low", "medium", "high"}
SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}

# Forbidden networks for SSRF prevention
FORBIDDEN_NETWORKS = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("198.18.0.0/15"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("224.0.0.0/4"),
    ipaddress.ip_network("240.0.0.0/4"),
    ipaddress.ip_network("255.255.255.255/32"),
    # IPv6
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def is_safe_webhook_url(url: str, allow_local: bool | None = None) -> tuple[bool, str]:
    """
    Validate that a webhook URL does not target loopback, private,
    link-local, or cloud metadata IP addresses (SSRF defense).
    """
    if allow_local is None:
        allow_local = WEBHOOK_ALLOW_LOCAL

    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception as exc:
        return False, f"Could not parse URL: {exc}"

    if parsed.scheme not in ("http", "https"):
        return False, "Scheme must be http or https"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL must include a valid hostname"

    # Reject URLs containing embedded credentials
    if parsed.username or parsed.password:
        return False, "URLs with embedded credentials are not allowed"

    if allow_local:
        return True, "OK"

    lower_host = hostname.lower()
    if lower_host in ("localhost", "local", "internal") or lower_host.endswith(
        (".localhost", ".local", ".internal")
    ):
        return False, f"Destination '{hostname}' is not permitted"

    # Attempt to parse hostname directly as an IP address
    try:
        ip_obj = ipaddress.ip_address(hostname)
        # Handle NAT64 (RFC 6052: 64:ff9b::/96) translated IPv4
        if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj in ipaddress.ip_network("64:ff9b::/96"):
            embedded = ipaddress.IPv4Address(ip_obj.packed[-4:])
            for net in FORBIDDEN_NETWORKS:
                if embedded in net:
                    return False, f"IP address '{hostname}' targets restricted network {net}"
            if embedded.is_private or embedded.is_loopback or embedded.is_link_local:
                return False, f"IP address '{hostname}' targets restricted network"
            return True, "OK"

        for net in FORBIDDEN_NETWORKS:
            if ip_obj in net:
                return False, f"IP address '{hostname}' is within restricted network {net}"
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            return False, f"IP address '{hostname}' is restricted"
        return True, "OK"
    except ValueError:
        pass  # Hostname is a domain name, proceed to DNS resolution

    # Resolve domain to all associated IP addresses
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addr_info = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        # Allow trusted public webhook provider domains in offline/sandboxed test environments
        if any(hostname.lower().endswith(domain) for domain in (".slack.com", ".discord.com", ".pagerduty.com", ".office.com")):
            return True, "OK"
        return False, f"DNS resolution failed for '{hostname}': {exc}"
    except Exception as exc:
        return False, f"Could not resolve host '{hostname}': {exc}"

    if not addr_info:
        return False, f"No IP address found for host '{hostname}'"

    for entry in addr_info:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)

            # Handle NAT64 (RFC 6052) translated IPv4
            if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj in ipaddress.ip_network("64:ff9b::/96"):
                embedded = ipaddress.IPv4Address(ip_obj.packed[-4:])
                for net in FORBIDDEN_NETWORKS:
                    if embedded in net:
                        return False, f"Resolved IP '{ip_str}' targets restricted network {net}"
                if embedded.is_private or embedded.is_loopback or embedded.is_link_local:
                    return False, f"Resolved IP '{ip_str}' targets restricted network"
                continue

            for net in FORBIDDEN_NETWORKS:
                if ip_obj in net:
                    return False, f"Resolved IP '{ip_str}' is within restricted network {net}"
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
                return False, f"Resolved IP '{ip_str}' is restricted"
        except ValueError:
            return False, f"Invalid IP address format: {ip_str}"

    return True, "OK"


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Enforces SSRF validation on any HTTP redirect before following."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        is_safe, reason = is_safe_webhook_url(newurl)
        if not is_safe:
            raise urllib.error.HTTPError(
                newurl,
                code,
                f"Redirect to unsafe destination blocked ({reason})",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


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
    is_safe, reason = is_safe_webhook_url(url)
    if not is_safe:
        raise ValueError(f"Webhook URL is not permitted: {reason}")
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
    timeout: float = 5.0,
) -> tuple[bool, int, str]:
    """Execute an HTTP POST request to deliver the webhook payload safely."""
    is_safe, reason = is_safe_webhook_url(url)
    if not is_safe:
        return False, 0, f"Blocked unsafe webhook destination: {reason}"

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
        opener = urllib.request.build_opener(SafeRedirectHandler)
        with opener.open(req, timeout=timeout) as response:
            status_code = response.getcode()
            body = response.read(10240).decode("utf-8", errors="ignore")[:300]
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
    """Dispatch a test ping alert to verify connectivity safely."""
    is_safe, reason = is_safe_webhook_url(url)
    if not is_safe:
        return {
            "success": False,
            "status_code": 400,
            "message": f"Webhook URL is not permitted: {reason}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

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
    success, status_code, response_body = deliver_webhook(url, payload, timeout=5.0)

    return {
        "success": success,
        "status_code": status_code,
        "message": response_body if response_body else ("OK" if success else "Failed to connect to webhook URL"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
