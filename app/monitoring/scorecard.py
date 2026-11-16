"""
Model Performance Scorecard Aggregator
=======================================
Reads data from all existing stores (RAG history, drift alerts, hallucination
runs saved in the scorecard DB, red-team campaigns saved in the scorecard DB)
and computes a per-model composite score + leaderboard.

Each "model entry" is identified by a user-supplied model name. Users register
scorecard entries via the API; this module rolls up the metrics.

Scorecard entry schema (persisted):
  - entry_id       : UUID
  - model_name     : human label
  - model_version  : semver string
  - created_at     : ISO timestamp
  - source         : "rag" | "hallucination" | "red_team" | "drift" | "manual"
  - metrics_json   : JSON blob of raw metric values
  - composite_score: 0–100 (higher = better)
  - notes          : optional free text

Composite score weights:
  RAG:              precision_at_k (25%) + recall_at_k (20%) + answer_relevance_score (20%)
                    + ndcg_at_k (10%) + context_support_coverage (10%) - hallucination_risk penalty (15%)
  Red team:         (100 - vulnerability_rate*100)  → safety_score (10% of composite)
  Drift:            1 - max_effect_size (capped 0-1)  → stability_score (10%)
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

SCORECARD_DB = PROJECT_ROOT / "scorecard.db"


# ── DB init ─────────────────────────────────────────────────────────────────

def _connect(db_path: str | Path = SCORECARD_DB):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_scorecard_db(db_path: str | Path = SCORECARD_DB) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scorecard_entries (
                entry_id        TEXT PRIMARY KEY,
                model_name      TEXT NOT NULL,
                model_version   TEXT NOT NULL DEFAULT '1.0.0',
                source          TEXT NOT NULL DEFAULT 'manual',
                composite_score REAL NOT NULL DEFAULT 0.0,
                metrics_json    TEXT NOT NULL DEFAULT '{}',
                notes           TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scorecard_model
            ON scorecard_entries(model_name)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scorecard_created
            ON scorecard_entries(created_at)
            """
        )


# ── Composite scorer ─────────────────────────────────────────────────────────

WEIGHTS = {
    "precision_at_k":           0.20,
    "recall_at_k":              0.15,
    "answer_relevance_score":   0.15,
    "ndcg_at_k":                0.10,
    "context_support_coverage": 0.10,
    # Penalty: hallucination_risk_score is 0-100, higher = worse
    # We invert: (100 - score) / 100
    "_hallucination_safety":    0.15,
    # Derived fields
    "_red_team_safety":         0.10,
    "_drift_stability":         0.05,
}


def compute_composite(metrics: dict[str, Any]) -> float:
    """
    Compute a 0–100 composite quality score from a metrics dict.
    Missing dimensions default to 0 (conservative — don't inflate unknown models).
    """
    score = 0.0

    def _get(key: str, default: float = 0.0) -> float:
        v = metrics.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    # RAG quality (0-1 each)
    score += _get("precision_at_k")           * WEIGHTS["precision_at_k"]           * 100
    score += _get("recall_at_k")              * WEIGHTS["recall_at_k"]               * 100
    score += _get("answer_relevance_score")   * WEIGHTS["answer_relevance_score"]    * 100
    score += _get("ndcg_at_k")                * WEIGHTS["ndcg_at_k"]                 * 100
    score += _get("context_support_coverage") * WEIGHTS["context_support_coverage"]  * 100

    # Hallucination safety: (100 - risk) / 100 → 0-1
    hal_risk = _get("hallucination_risk_score", 50.0)  # default 50 = unknown
    hal_safety = max(0.0, (100.0 - hal_risk) / 100.0)
    score += hal_safety * WEIGHTS["_hallucination_safety"] * 100

    # Red-team safety: provide vulnerability_rate (0-1) or safety_score (0-100)
    if "safety_score" in metrics:
        rt_safety = _get("safety_score") / 100.0
    elif "vulnerability_rate" in metrics:
        rt_safety = max(0.0, 1.0 - _get("vulnerability_rate"))
    else:
        rt_safety = 0.5  # unknown
    score += rt_safety * WEIGHTS["_red_team_safety"] * 100

    # Drift stability: provide max_effect_size (0-1), lower = more stable
    if "max_effect_size" in metrics:
        stability = max(0.0, 1.0 - _get("max_effect_size"))
    elif "drift_effect_size" in metrics:
        stability = max(0.0, 1.0 - _get("drift_effect_size"))
    else:
        stability = 0.5
    score += stability * WEIGHTS["_drift_stability"] * 100

    return round(min(100.0, max(0.0, score)), 2)


# ── CRUD ─────────────────────────────────────────────────────────────────────

def add_scorecard_entry(
    model_name: str,
    metrics: dict[str, Any],
    model_version: str = "1.0.0",
    source: str = "manual",
    notes: str = "",
    db_path: str | Path = SCORECARD_DB,
) -> dict[str, Any]:
    """Save a new scorecard entry and return it."""
    initialize_scorecard_db(db_path)

    model_name = model_name.strip()
    if not model_name:
        raise ValueError("model_name must not be empty.")
    if source not in {"rag", "hallucination", "red_team", "drift", "manual"}:
        raise ValueError("source must be: rag | hallucination | red_team | drift | manual")

    composite = compute_composite(metrics)
    entry_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO scorecard_entries
                (entry_id, model_name, model_version, source, composite_score,
                 metrics_json, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                model_name,
                model_version.strip(),
                source,
                composite,
                json.dumps(metrics, default=str),
                notes,
                created_at,
            ),
        )

    return get_scorecard_entry(entry_id, db_path=db_path)  # type: ignore[return-value]


def list_scorecard_entries(
    model_name: str | None = None,
    source: str | None = None,
    limit: int = 100,
    db_path: str | Path = SCORECARD_DB,
) -> list[dict[str, Any]]:
    """Return all entries, optionally filtered."""
    initialize_scorecard_db(db_path)

    query = "SELECT * FROM scorecard_entries"
    params: list[Any] = []
    clauses: list[str] = []

    if model_name:
        clauses.append("model_name = ?")
        params.append(model_name.strip())
    if source:
        clauses.append("source = ?")
        params.append(source.strip())

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [_deserialize(dict(r)) for r in rows]


def get_scorecard_entry(
    entry_id: str,
    db_path: str | Path = SCORECARD_DB,
) -> dict[str, Any] | None:
    initialize_scorecard_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM scorecard_entries WHERE entry_id = ?", (entry_id,)
        ).fetchone()
    return _deserialize(dict(row)) if row else None


def delete_scorecard_entry(
    entry_id: str,
    db_path: str | Path = SCORECARD_DB,
) -> bool:
    initialize_scorecard_db(db_path)
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM scorecard_entries WHERE entry_id = ?", (entry_id,)
        )
    return cursor.rowcount > 0


# ── Leaderboard aggregation ───────────────────────────────────────────────────

def build_leaderboard(db_path: str | Path = SCORECARD_DB) -> dict[str, Any]:
    """
    Aggregate all entries by model_name.
    For each model, compute:
      - latest_score        : composite score of the most recent entry
      - average_score       : mean composite across all entries
      - best_score          : highest composite achieved
      - entry_count         : total entries
      - trend               : "up" | "down" | "stable" (based on last 2 entries)
      - trend_delta         : latest - previous composite
      - metric_averages     : averaged raw metrics across all entries
      - sources             : distinct data sources recorded
      - last_entry_at       : ISO timestamp of newest entry
    """
    initialize_scorecard_db(db_path)
    entries = list_scorecard_entries(limit=5000, db_path=db_path)

    # Group by model_name
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        grouped.setdefault(entry["model_name"], []).append(entry)

    leaderboard: list[dict[str, Any]] = []

    for model_name, model_entries in grouped.items():
        # Sort oldest→newest for trend calc
        sorted_entries = sorted(model_entries, key=lambda e: e["created_at"])

        scores = [e["composite_score"] for e in sorted_entries]
        latest = scores[-1] if scores else 0.0
        previous = scores[-2] if len(scores) >= 2 else None

        if previous is None:
            trend = "stable"
            trend_delta = 0.0
        else:
            delta = latest - previous
            if abs(delta) < 0.5:
                trend = "stable"
            elif delta > 0:
                trend = "up"
            else:
                trend = "down"
            trend_delta = round(delta, 2)

        # Average raw metrics across all entries
        all_metrics: dict[str, list[float]] = {}
        for entry in model_entries:
            for k, v in entry.get("metrics", {}).items():
                try:
                    all_metrics.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    pass

        metric_averages = {
            k: round(sum(vs) / len(vs), 4)
            for k, vs in all_metrics.items()
            if vs
        }

        sources = sorted({e["source"] for e in model_entries})
        versions = sorted({e["model_version"] for e in model_entries})

        leaderboard.append(
            {
                "model_name": model_name,
                "model_versions": versions,
                "latest_score": round(latest, 2),
                "average_score": round(sum(scores) / len(scores), 2),
                "best_score": round(max(scores), 2),
                "entry_count": len(model_entries),
                "trend": trend,
                "trend_delta": trend_delta,
                "metric_averages": metric_averages,
                "sources": sources,
                "last_entry_at": sorted_entries[-1]["created_at"],
                "score_history": [
                    {"created_at": e["created_at"], "score": e["composite_score"]}
                    for e in sorted_entries
                ],
            }
        )

    # Rank by latest_score descending
    leaderboard.sort(key=lambda x: x["latest_score"], reverse=True)
    for rank, entry in enumerate(leaderboard, start=1):
        entry["rank"] = rank

    return {
        "leaderboard": leaderboard,
        "total_models": len(leaderboard),
        "total_entries": len(entries),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "score_weights": WEIGHTS,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _deserialize(row: dict[str, Any]) -> dict[str, Any]:
    row["metrics"] = json.loads(row.pop("metrics_json", "{}"))
    return row


def get_score_metadata() -> dict[str, Any]:
    """Return the scoring formula and dimension descriptions."""
    return {
        "composite_score_range": "0–100 (higher = better)",
        "dimensions": {
            "precision_at_k": {"weight": WEIGHTS["precision_at_k"], "range": "0–1", "higher_is_better": True},
            "recall_at_k": {"weight": WEIGHTS["recall_at_k"], "range": "0–1", "higher_is_better": True},
            "answer_relevance_score": {"weight": WEIGHTS["answer_relevance_score"], "range": "0–1", "higher_is_better": True},
            "ndcg_at_k": {"weight": WEIGHTS["ndcg_at_k"], "range": "0–1", "higher_is_better": True},
            "context_support_coverage": {"weight": WEIGHTS["context_support_coverage"], "range": "0–1", "higher_is_better": True},
            "hallucination_risk_score": {"weight": WEIGHTS["_hallucination_safety"], "range": "0–100", "higher_is_better": False,
                                          "note": "Inverted: (100-score)/100 used in composite"},
            "safety_score": {"weight": WEIGHTS["_red_team_safety"], "range": "0–100", "higher_is_better": True,
                              "note": "Or supply vulnerability_rate (0-1)"},
            "max_effect_size": {"weight": WEIGHTS["_drift_stability"], "range": "0–1", "higher_is_better": False,
                                 "note": "Drift KS effect size; inverted for stability"},
        },
    }
