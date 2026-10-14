
from __future__ import annotations

import math
from typing import Any

from app.diagnostics.rag_history import list_rag_evaluations


DASHBOARD_METRICS = (
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "answer_relevance_score",
    "context_support_coverage",
)


def _valid_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def get_rag_dashboard(limit: int = 20) -> dict[str, Any]:
    """Build a dashboard summary from persisted RAG evaluations."""

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    recent_runs = list_rag_evaluations(limit=limit)

    metric_values: dict[str, list[float]] = {
        metric: [] for metric in DASHBOARD_METRICS
    }

    runs_with_findings = 0
    runs_without_findings = 0
    runs_with_missing_findings = 0

    for run in recent_runs:
        summary = run.get("summary", {})

        for metric in DASHBOARD_METRICS:
            value = summary.get(metric)

            if _valid_number(value):
                metric_values[metric].append(float(value))

        finding_count = summary.get("finding_count")

        if (
            _valid_number(finding_count)
            and finding_count >= 0
        ):
            if finding_count > 0:
                runs_with_findings += 1
            else:
                runs_without_findings += 1
        else:
            runs_with_missing_findings += 1

    metric_summary = {}

    for metric, values in metric_values.items():
        metric_summary[metric] = {
            "available_runs": len(values),
            "average": (
                round(sum(values) / len(values), 6)
                if values
                else None
            ),
            "latest_value": (
                values[0] if values else None
            ),
        }

    return {
        "service": "ModelAutopsy AI RAG Dashboard",
        "runs_included": len(recent_runs),
        "finding_summary": {
            "runs_with_findings": runs_with_findings,
            "runs_without_findings": runs_without_findings,
            "runs_with_missing_finding_data": runs_with_missing_findings,
        },
        "metrics": metric_summary,
        "recent_runs": recent_runs,
        "limitations": [
            "Averages only include runs where each metric is available.",
            "The dashboard summarizes the most recent requested runs, not necessarily all historical runs.",
            "Runs may use different queries or relevance labels, so aggregate metrics may not be directly comparable.",
            "Finding counts are diagnostic indicators, not calibrated risk scores.",
            "This endpoint does not independently calculate regression counts.",
        ],
    }
