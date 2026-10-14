
from unittest.mock import patch

import pytest

from app.diagnostics.rag_dashboard import get_rag_dashboard


def test_dashboard_aggregates_metrics():
    runs = [
        {
            "run_id": "run-2",
            "created_at": "2026-10-14T10:00:00+00:00",
            "query": "Question two",
            "summary": {
                "precision_at_k": 0.8,
                "recall_at_k": 0.6,
                "answer_relevance_score": 0.9,
                "finding_count": 0,
            },
        },
        {
            "run_id": "run-1",
            "created_at": "2026-10-13T10:00:00+00:00",
            "query": "Question one",
            "summary": {
                "precision_at_k": 0.6,
                "recall_at_k": 0.8,
                "answer_relevance_score": 0.7,
                "finding_count": 2,
            },
        },
    ]

    with patch(
        "app.diagnostics.rag_dashboard.list_rag_evaluations",
        return_value=runs,
    ):
        result = get_rag_dashboard(limit=20)

    assert result["runs_included"] == 2
    assert result["metrics"]["precision_at_k"]["average"] == pytest.approx(0.7)
    assert result["metrics"]["precision_at_k"]["latest_value"] == 0.8
    assert result["metrics"]["recall_at_k"]["average"] == pytest.approx(0.7)
    assert result["finding_summary"]["runs_with_findings"] == 1
    assert result["finding_summary"]["runs_without_findings"] == 1


def test_dashboard_handles_empty_history():
    with patch(
        "app.diagnostics.rag_dashboard.list_rag_evaluations",
        return_value=[],
    ):
        result = get_rag_dashboard()

    assert result["runs_included"] == 0
    assert result["metrics"]["precision_at_k"]["average"] is None
    assert result["recent_runs"] == []


def test_dashboard_ignores_missing_and_invalid_metrics():
    runs = [
        {
            "run_id": "run-1",
            "created_at": "2026-10-14T10:00:00+00:00",
            "query": "Question",
            "summary": {
                "precision_at_k": float("nan"),
                "recall_at_k": None,
                "finding_count": None,
            },
        }
    ]

    with patch(
        "app.diagnostics.rag_dashboard.list_rag_evaluations",
        return_value=runs,
    ):
        result = get_rag_dashboard()

    assert result["metrics"]["precision_at_k"]["average"] is None
    assert result["metrics"]["recall_at_k"]["average"] is None
    assert result["finding_summary"]["runs_with_missing_finding_data"] == 1


def test_dashboard_rejects_invalid_limit():
    with pytest.raises(ValueError, match="limit"):
        get_rag_dashboard(limit=0)
