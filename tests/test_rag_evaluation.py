
import pytest

import app.diagnostics.rag_evaluation as rag_evaluation


def test_retrieval_metrics(monkeypatch):
    def fake_embeddings(texts, model_name=None):
        import numpy as np

        return np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(
        rag_evaluation,
        "embed_texts",
        fake_embeddings,
    )

    documents = [
        {"id": "doc-1", "text": "Relevant billing information"},
        {"id": "doc-2", "text": "Unrelated weather information"},
        {"id": "doc-3", "text": "Another billing document"},
    ]

    report = rag_evaluation.evaluate_retrieval(
        query="How does billing work?",
        retrieved_documents=documents,
        relevant_document_ids=["doc-1", "doc-3"],
        k=2,
    )

    assert report["precision_at_k"] == 0.5
    assert report["recall_at_k"] == 0.5
    assert report["reciprocal_rank"] == 1.0
    assert 0.0 <= report["ndcg_at_k"] <= 1.0
    assert report["mean_pairwise_similarity"] is not None


def test_retrieval_rejects_empty_query():
    with pytest.raises(ValueError):
        rag_evaluation.evaluate_retrieval(
            query=" ",
            retrieved_documents=[
                {"id": "doc-1", "text": "Example document"}
            ],
            relevant_document_ids=["doc-1"],
        )


def test_retrieval_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        rag_evaluation.evaluate_retrieval(
            query="Example question",
            retrieved_documents=[
                {"id": "doc-1", "text": "First document"},
                {"id": "doc-1", "text": "Duplicate ID"},
            ],
            relevant_document_ids=["doc-1"],
        )


def test_retrieval_rejects_missing_relevance_labels():
    with pytest.raises(ValueError):
        rag_evaluation.evaluate_retrieval(
            query="Example question",
            retrieved_documents=[
                {"id": "doc-1", "text": "Example document"}
            ],
            relevant_document_ids=[],
        )


def test_single_document_has_no_pairwise_similarity():
    report = rag_evaluation.evaluate_retrieval(
        query="Example question",
        retrieved_documents=[
            {"id": "doc-1", "text": "Relevant document"}
        ],
        relevant_document_ids=["doc-1"],
        k=5,
    )

    assert report["precision_at_k"] == 1.0
    assert report["recall_at_k"] == 1.0
    assert report["mean_pairwise_similarity"] is None
