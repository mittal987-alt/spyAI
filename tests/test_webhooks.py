"""
Unit tests for Webhook Alerts & Notification Dispatcher.
"""

import tempfile
from pathlib import Path
from app.monitoring.webhooks import (
    create_webhook,
    delete_webhook,
    format_webhook_payload,
    initialize_webhook_tables,
    list_webhooks,
    toggle_webhook,
)


def test_webhook_crud_lifecycle():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_alerts.db"
        initialize_webhook_tables(db_path)

        # 1. Create Webhook
        hook = create_webhook(
            name="Slack Production Alerts",
            url="https://hooks.slack.com/services/T00/B00/X00",
            service_type="slack",
            min_severity="high",
            db_path=db_path,
        )

        assert hook["name"] == "Slack Production Alerts"
        assert hook["service_type"] == "slack"
        assert hook["min_severity"] == "high"
        assert hook["is_enabled"] is True

        # 2. List Webhooks
        hooks = list_webhooks(db_path)
        assert len(hooks) == 1
        assert hooks[0]["webhook_id"] == hook["webhook_id"]

        # 3. Toggle Webhook
        toggled = toggle_webhook(hook["webhook_id"], False, db_path)
        assert toggled is True

        updated_hooks = list_webhooks(db_path)
        assert updated_hooks[0]["is_enabled"] is False

        # 4. Delete Webhook
        deleted = delete_webhook(hook["webhook_id"], db_path)
        assert deleted is True

        assert len(list_webhooks(db_path)) == 0


def test_format_webhook_payloads():
    sample_alert = {
        "title": "HIGH drift detected: feature_3",
        "message": "Distribution shifted significantly.",
        "severity": "high",
        "feature": "feature_3",
        "effect_size": 0.354,
        "adjusted_p_value": 0.0001,
    }

    # Slack format
    slack_payload = format_webhook_payload("slack", sample_alert)
    assert "attachments" in slack_payload
    assert slack_payload["attachments"][0]["color"] == "#E11D48"

    # Discord format
    discord_payload = format_webhook_payload("discord", sample_alert)
    assert "embeds" in discord_payload
    assert len(discord_payload["embeds"][0]["fields"]) == 4

    # Generic format
    generic_payload = format_webhook_payload("generic", sample_alert)
    assert generic_payload["event"] == "model_drift_alert"
    assert generic_payload["alert"]["feature"] == "feature_3"
