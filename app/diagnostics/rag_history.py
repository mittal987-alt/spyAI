
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT


RAG_HISTORY_DB = PROJECT_ROOT / "rag_history.db"

COMPARABLE_METRICS = (
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "answer_relevance_score",
    "context_support_coverage",
)


def _connect(db_path: str | Path = RAG_HISTORY_DB):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_rag_history_db(
    db_path: str | Path = RAG_HISTORY_DB,
) -> None:
    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_evaluations (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                query TEXT NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_evaluations_created_at
            ON rag_evaluations(created_at)
            """
        )


def save_rag_evaluation(
    report: dict[str, Any],
    db_path: str | Path = RAG_HISTORY_DB,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("report must be a dictionary")

    if not isinstance(report.get("query"), str) or not report["query"].strip():
        raise ValueError("report must contain a non-empty query")

    if not isinstance(report.get("summary"), dict):
        raise ValueError("report must contain a summary dictionary")

    initialize_rag_history_db(db_path)

    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    stored_report = dict(report)
    stored_report["history_run_id"] = run_id
    stored_report["history_created_at"] = created_at

    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO rag_evaluations (
                run_id, created_at, query, report_json
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                report["query"],
                json.dumps(stored_report, allow_nan=False),
            ),
        )

    return {
        "run_id": run_id,
        "created_at": created_at,
        "query": report["query"],
    }


def get_rag_evaluation(
    run_id: str,
    db_path: str | Path = RAG_HISTORY_DB,
) -> dict[str, Any] | None:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty text")

    initialize_rag_history_db(db_path)

    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT run_id, created_at, query, report_json
            FROM rag_evaluations
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "run_id": row["run_id"],
        "created_at": row["created_at"],
        "query": row["query"],
        "report": json.loads(row["report_json"]),
    }


def list_rag_evaluations(
    limit: int = 20,
    db_path: str | Path = RAG_HISTORY_DB,
) -> list[dict[str, Any]]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    initialize_rag_history_db(db_path)

    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT run_id, created_at, query, report_json
            FROM rag_evaluations
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    history = []

    for row in rows:
        report = json.loads(row["report_json"])

        history.append(
            {
                "run_id": row["run_id"],
                "created_at": row["created_at"],
                "query": row["query"],
                "summary": report.get("summary", {}),
            }
        )

    return history


def compare_rag_evaluations(
    baseline_run_id: str,
    candidate_run_id: str,
    regression_threshold: float = 0.10,
    db_path: str | Path = RAG_HISTORY_DB,
) -> dict[str, Any]:
    if not 0 < regression_threshold <= 1:
        raise ValueError(
            "regression_threshold must be greater than 0 and at most 1"
        )

    if baseline_run_id == candidate_run_id:
        raise ValueError("Baseline and candidate runs must be different")

    baseline = get_rag_evaluation(baseline_run_id, db_path)
    candidate = get_rag_evaluation(candidate_run_id, db_path)

    if baseline is None:
        raise ValueError("Baseline evaluation run was not found")

    if candidate is None:
        raise ValueError("Candidate evaluation run was not found")

    baseline_summary = baseline["report"].get("summary", {})
    candidate_summary = candidate["report"].get("summary", {})

    metrics = {}
    regressions = []
    improvements = []

    for metric in COMPARABLE_METRICS:
        old_value = baseline_summary.get(metric)
        new_value = candidate_summary.get(metric)

        if (
            not isinstance(old_value, (int, float))
            or isinstance(old_value, bool)
            or not isinstance(new_value, (int, float))
            or isinstance(new_value, bool)
        ):
            metrics[metric] = {
                "baseline": old_value,
                "candidate": new_value,
                "change": None,
                "relative_change": None,
                "status": "unavailable",
            }
            continue

        old_value = float(old_value)
        new_value = float(new_value)

        if not all(map(lambda value: value == value and abs(value) != float("inf"),
                       (old_value, new_value))):
            metrics[metric] = {
                "baseline": old_value,
                "candidate": new_value,
                "change": None,
                "relative_change": None,
                "status": "unavailable",
            }
            continue

        change = new_value - old_value
        relative_change = (
            change / abs(old_value) if old_value != 0 else None
        )

        # These six metrics are quality metrics where higher is better.
        if old_value > 0:
            regressed = change < 0 and abs(change) / abs(old_value) >= regression_threshold
            improved = change > 0 and change / abs(old_value) >= regression_threshold
        else:
            # Relative changes are undefined when the baseline is zero.
            # Use an absolute drop/rise threshold in that case.
            regressed = change <= -regression_threshold
            improved = change >= regression_threshold

        status = (
            "regression" if regressed
            else "improvement" if improved
            else "stable"
        )

        metrics[metric] = {
            "baseline": old_value,
            "candidate": new_value,
            "change": change,
            "relative_change": relative_change,
            "status": status,
        }

        if regressed:
            regressions.append(metric)
        elif improved:
            improvements.append(metric)

    return {
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "regression_threshold": regression_threshold,
        "metrics": metrics,
        "regressions": regressions,
        "improvements": improvements,
        "regression_detected": bool(regressions),
        "limitations": [
            "Regression thresholds are heuristic and should be calibrated.",
            "Runs are most meaningful to compare on the same evaluation dataset.",
            "Embedding-based relevance and support metrics are proxies, not proof of correctness.",
            "Changes in query composition or relevance labels can invalidate direct comparisons.",
        ],
    }
