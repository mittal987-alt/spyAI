"""
Autonomous Retraining Trigger Agent (Agentic MLOps).
Monitors continuous data/concept drift and automatically queues retraining jobs when thresholds breach.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path("artifacts") / "retraining.db"


def initialize_retraining_db(db_path: Path | None = None) -> None:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retraining_runs (
            run_id              TEXT PRIMARY KEY,
            triggered_at        TEXT,
            reason              TEXT,
            status              TEXT,
            metrics_snapshot    TEXT,
            candidate_model_id  TEXT,
            completion_time     TEXT,
            notes               TEXT
        )
    """)
    conn.commit()
    conn.close()


def evaluate_retraining_need(
    metrics: dict[str, Any] | None = None,
    threshold_psi: float = 0.25,
    threshold_ks_pvalue: float = 0.05,
) -> dict[str, Any]:
    """
    Decide autonomously whether current drift telemetry warrants retraining.
    """
    m = metrics or {}
    psi = float(m.get("psi", 0.0))
    ks_p = float(m.get("ks_p_value", 1.0))
    accuracy_drop = float(m.get("accuracy_drop_pct", 0.0))

    reasons: list[str] = []
    urgency = "low"

    if psi >= threshold_psi:
        reasons.append(f"Population Stability Index (PSI={psi:.3f}) exceeded threshold ({threshold_psi}). Severe feature distribution shift.")
        urgency = "high"

    if ks_p < threshold_ks_pvalue:
        reasons.append(f"Kolmogorov-Smirnov p-value ({ks_p:.4f}) indicates statistically significant drift.")
        urgency = "medium" if urgency != "high" else "critical"

    if accuracy_drop >= 10.0:
        reasons.append(f"Observed model accuracy dropped by {accuracy_drop:.1f}% on production verification.")
        urgency = "critical"

    retraining_recommended = len(reasons) > 0

    return {
        "retraining_recommended": retraining_recommended,
        "urgency": urgency if retraining_recommended else "nominal",
        "psi": psi,
        "ks_p_value": ks_p,
        "reasons": reasons if reasons else ["Telemetry nominal. Feature distributions match baseline."],
        "action_recommended": "Queue Autonomous Pipeline Retraining" if retraining_recommended else "Maintain current production checkpoint",
    }


def trigger_autonomous_retraining(
    reason: str,
    metrics: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """
    Queue an autonomous retraining job and record in ledger.
    """
    path = db_path or DB_PATH
    initialize_retraining_db(path)

    run_id = f"retrain-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    candidate_model_id = f"model-v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}-healed"

    conn = sqlite3.connect(path)
    conn.execute(
        """
        INSERT INTO retraining_runs VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            run_id,
            now,
            reason,
            "QUEUED",
            json.dumps(metrics or {}),
            candidate_model_id,
            "",
            "Autonomous self-healing triggered by ModelAutopsy Agent.",
        ),
    )
    conn.commit()
    conn.close()

    return {
        "run_id": run_id,
        "status": "QUEUED",
        "triggered_at": now,
        "reason": reason,
        "candidate_model_id": candidate_model_id,
        "message": f"Autonomous retraining pipeline activated for candidate {candidate_model_id}.",
    }


def get_retraining_history(limit: int = 50, db_path: Path | None = None) -> list[dict[str, Any]]:
    """Retrieve history of autonomous retraining runs."""
    path = db_path or DB_PATH
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM retraining_runs ORDER BY triggered_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        d["metrics_snapshot"] = json.loads(d.get("metrics_snapshot") or "{}")
        results.append(d)
    return results
