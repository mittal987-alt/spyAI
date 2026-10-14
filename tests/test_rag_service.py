
import pytest

import app.diagnostics.rag_service as rag_service


def test_rag_service_combines_evaluation_reports(monkeypatch):
    monkeypatch.setattr(
        rag_service,
        "evaluate_retrieval",
        lambda **kwargs: {
            "precision_at_k": 1.0,
            "recall_at_k": 0.5,
            "reciprocal_rank": 1.0,
            "ndcg_at_k": 1.0,
        },
    )

    monkeypatch.setattr(
        rag_service,
        "evaluate_rag_answer",
        lambda **kwargs: {
            "answer_relevance_score": 0.9,
            "context_support_coverage": 1.0,
            "potentially_unsupported_sentence_count": 0,
        },
    )

    report = rag_service.run_rag_evaluation(
        query="How does billing work?",
        answer="Billing is explained in the invoice.",
        retrieved_documents=[
            {"id": "doc-1", "text": "Billing information."}
        ],
        relevant_document_ids=["doc-1"],
    )

    assert report["summary"]["precision_at_k"] == 1.0
    assert report["summary"]["answer_relevance_score"] == 0.9
    assert report["summary"]["finding_count"] == 0


def test_rag_service_rejects_empty_query():
    with pytest.raises(ValueError):
        rag_service.run_rag_evaluation(
            query=" ",
            answer="An answer.",
            retrieved_documents=[
                {"id": "doc-1", "text": "Example context."}
            ],
            relevant_document_ids=["doc-1"],
        )


def test_rag_service_rejects_oversized_document():
    with pytest.raises(ValueError):
        rag_service.run_rag_evaluation(
            query="Example question?",
            answer="An answer.",
            retrieved_documents=[
                {"id": "doc-1", "text": "x" * 10001}
            ],
            relevant_document_ids=["doc-1"],
        )


def test_rag_service_rejects_duplicate_relevant_ids():
    with pytest.raises(ValueError):
        rag_service.run_rag_evaluation(
            query="Example question?",
            answer="An answer.",
            retrieved_documents=[
                {"id": "doc-1", "text": "Example context."}
            ],
            relevant_document_ids=["doc-1", "doc-1"],
        )
