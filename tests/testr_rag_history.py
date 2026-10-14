
import pytest

from app.diagnostics.rag_history import (
    save_rag_evaluation,
    get_rag_evaluation,
    list_rag_evaluations,
    compare_rag_evaluations,
)


def make_report(
    precision=0.8,
    recall=0.7,
    reciprocal_rank=0.8,
    ndcg=0.75,
    relevance=0.9,
    support=1.0,
):
    return {
        "query": "What is the answer?",
        "summary": {
            "precision_at_k": precision,
            "recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank,
            "ndcg_at_k": ndcg,
            "answer_relevance_score": relevance,
            "context_support_coverage": support,
        },
        "findings": [],
    }


def test_save_and_get_evaluation(tmp_path):
    db_path = tmp_path / "history.db"
    saved = save_rag_evaluation(make_report(), db_path)

    retrieved = get_rag_evaluation(saved["run_id"], db_path)

    assert retrieved is not None
    assert retrieved["run_id"] == saved["run_id"]
    assert retrieved["report"]["summary"]["precision_at_k"] == 0.8


def test_list_evaluations(tmp_path):
    db_path = tmp_path / "history.db"

    first = save_rag_evaluation(make_report(), db_path)
    second = save_rag_evaluation(make_report(precision=0.9), db_path)

    history = list_rag_evaluations(limit=10, db_path=db_path)

    assert len(history) == 2
    assert {item["run_id"] for item in history} == {
        first["run_id"],
        second["run_id"],
    }


def test_detects_quality_regression(tmp_path):
    db_path = tmp_path / "history.db"

    baseline = save_rag_evaluation(
        make_report(precision=0.9, recall=0.9),
        db_path,
    )
    candidate = save_rag_evaluation(
        make_report(precision=0.7, recall=0.9),
        db_path,
    )

    result = compare_rag_evaluations(
        baseline["run_id"],
        candidate["run_id"],
        regression_threshold=0.10,
        db_path=db_path,
    )

    assert result["regression_detected"] is True
    assert "precision_at_k" in result["regressions"]
    assert "recall_at_k" not in result["regressions"]


def test_missing_run_raises(tmp_path):
    db_path = tmp_path / "history.db"

    with pytest.raises(ValueError, match="Baseline"):
        compare_rag_evaluations(
            "missing-baseline",
            "missing-candidate",
            db_path=db_path,
        )


def test_invalid_regression_threshold_raises(tmp_path):
    with pytest.raises(ValueError, match="regression_threshold"):
        compare_rag_evaluations(
            "baseline",
            "candidate",
            regression_threshold=0,
            db_path=tmp_path / "history.db",
        )
