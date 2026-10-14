
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.diagnostics.rag_evaluation import evaluate_retrieval
from app.diagnostics.rag_answer_evaluation import evaluate_rag_answer
from app.diagnostics.rag_service import run_rag_evaluation


client = TestClient(app)


def test_retrieval_metrics_are_consistent():
    documents = [
        {"id": "doc-1", "text": "Relevant document"},
        {"id": "doc-2", "text": "Irrelevant document"},
        {"id": "doc-3", "text": "Another relevant document"},
    ]

    with patch(
        "app.diagnostics.rag_evaluation.embed_texts",
        return_value=np.array(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
    ):
        result = evaluate_retrieval(
            query="test query",
            retrieved_documents=documents,
            relevant_document_ids=["doc-1", "doc-3"],
            k=2,
        )

    assert result["precision_at_k"] == pytest.approx(0.5)
    assert result["recall_at_k"] == pytest.approx(0.5)
    assert result["reciprocal_rank"] == pytest.approx(1.0)
    assert 0.0 <= result["ndcg_at_k"] <= 1.0


def test_retrieval_rejects_duplicate_document_ids():
    documents = [
        {"id": "doc-1", "text": "First"},
        {"id": "doc-1", "text": "Duplicate"},
    ]

    with pytest.raises(ValueError, match="unique"):
        evaluate_retrieval(
            query="test query",
            retrieved_documents=documents,
            relevant_document_ids=["doc-1"],
            k=2,
        )


def test_answer_evaluation_reports_unsupported_claims():
    vectors = np.array(
        [
            [1.0, 0.0, 0.0],  # query
            [0.0, 1.0, 0.0],  # complete answer
            [0.0, 1.0, 0.0],  # answer sentence
            [1.0, 0.0, 0.0],  # retrieved context
        ],
        dtype=np.float32,
    )

    with patch(
        "app.diagnostics.rag_answer_evaluation.embed_texts",
        return_value=vectors,
    ):
        result = evaluate_rag_answer(
            query="What is the product?",
            answer="The product costs one hundred dollars.",
            retrieved_contexts=["The product is available today."],
            support_threshold=0.5,
        )

    assert result["potentially_unsupported_sentence_count"] == 1
    assert result["context_support_coverage"] == pytest.approx(0.0)
    assert len(result["potentially_unsupported_sentences"]) == 1


def test_rag_service_combines_evaluation_results():
    retrieval = {
        "precision_at_k": 1.0,
        "recall_at_k": 1.0,
        "reciprocal_rank": 1.0,
        "ndcg_at_k": 1.0,
        "mean_pairwise_similarity": 0.2,
    }

    answer_evaluation = {
        "answer_relevance_score": 0.9,
        "context_support_coverage": 1.0,
        "potentially_unsupported_sentence_count": 0,
    }

    with (
        patch(
            "app.diagnostics.rag_service.evaluate_retrieval",
            return_value=retrieval,
        ),
        patch(
            "app.diagnostics.rag_service.evaluate_rag_answer",
            return_value=answer_evaluation,
        ),
    ):
        result = run_rag_evaluation(
            query="What is documented?",
            answer="The documented answer.",
            retrieved_documents=[
                {"id": "doc-1", "text": "The documented answer."}
            ],
            relevant_document_ids=["doc-1"],
            k=1,
        )

    assert result["evaluation"]["retrieval"] == retrieval
    assert result["evaluation"]["answer"] == answer_evaluation
    assert result["summary"]["precision_at_k"] == 1.0
    assert result["summary"]["answer_relevance_score"] == 0.9
    assert result["summary"]["finding_count"] == 0
    assert result["findings"] == []


def test_rag_endpoint_rejects_empty_query():
    response = client.post(
        "/api/v1/rag/evaluate",
        json={
            "query": "",
            "answer": "An answer.",
            "retrieved_documents": [
                {"id": "doc-1", "text": "A document."}
            ],
            "relevant_document_ids": ["doc-1"],
        },
    )

    assert response.status_code == 422


def test_rag_endpoint_rejects_empty_document_list():
    response = client.post(
        "/api/v1/rag/evaluate",
        json={
            "query": "A question?",
            "answer": "An answer.",
            "retrieved_documents": [],
            "relevant_document_ids": ["doc-1"],
        },
    )

    assert response.status_code == 422
