
from app.monitoring.alerts import (
    generate_drift_alerts,
    save_alerts,
    list_alerts,
    get_alert,
)


def sample_drift_results():
    return [
        {
            "feature": "feature_0",
            "ks_statistic": 0.45,
            "p_value": 0.00001,
            "adjusted_p_value": 0.00008,
            "effect_size": 0.45,
            "statistically_significant": True,
            "drift_detected": True,
            "severity": "high",
        },
        {
            "feature": "feature_1",
            "ks_statistic": 0.02,
            "p_value": 0.50,
            "adjusted_p_value": 0.50,
            "effect_size": 0.02,
            "statistically_significant": False,
            "drift_detected": False,
            "severity": "none",
        },
    ]


def test_only_detected_drift_generates_alerts():
    alerts = generate_drift_alerts(sample_drift_results())

    assert len(alerts) == 1
    assert alerts[0]["feature"] == "feature_0"
    assert alerts[0]["severity"] == "high"
    assert alerts[0]["recommendation"]


def test_alerts_are_saved_and_retrieved(tmp_path):
    db_path = tmp_path / "test_alerts.db"
    alerts = generate_drift_alerts(
        sample_drift_results(),
        run_id="test-run-123",
    )

    inserted = save_alerts(alerts, db_path=db_path)

    assert inserted == 1

    saved = list_alerts(db_path=db_path)
    assert len(saved) == 1
    assert saved[0]["run_id"] == "test-run-123"
    assert saved[0]["feature"] == "feature_0"


def test_alert_can_be_retrieved_by_id(tmp_path):
    db_path = tmp_path / "test_alerts.db"
    alerts = generate_drift_alerts(sample_drift_results())

    save_alerts(alerts, db_path=db_path)
    result = get_alert(
        alerts[0]["alert_id"],
        db_path=db_path,
    )

    assert result is not None
    assert result["alert_id"] == alerts[0]["alert_id"]


def test_unknown_alert_returns_none(tmp_path):
    result = get_alert(
        "does-not-exist",
        db_path=tmp_path / "test_alerts.db",
    )

    assert result is None


def test_alerts_can_be_filtered_by_severity(tmp_path):
    db_path = tmp_path / "test_alerts.db"

    alerts = generate_drift_alerts(sample_drift_results())
    save_alerts(alerts, db_path=db_path)

    assert len(list_alerts(
        severity="high",
        db_path=db_path,
    )) == 1

    assert len(list_alerts(
        severity="low",
        db_path=db_path,
    )) == 0
