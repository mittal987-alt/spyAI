
import numpy as np
import pytest

import app.diagnostics.rag_answer_evaluation as evaluator


def fake_embeddings(texts, model_name=None):
    vectors = []

    for text in texts:
        normalized = text.lower()

        if "billing" in normalized or "invoice" in normalized:
            vector = [1.0, 0.0, 0.0]
        elif "account" in normalized or "password" in normalized:
            vector = [0.0, 1.0, 0.0]
        else:
            vector = [0.0, 0.0, 1.0]

        vectors.append(vector)

    return np.asarray(vectors, dtype=np.float32)


def test_evaluates_answer_and_context(monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "embed_texts",
        fake_embeddings,
    )

    report = evaluator.evaluate_rag_answer(
        query="How does billing work?",
        answer="Billing information is available in your invoice.",
        retrieved_contexts=[
            "Billing details are provided in the invoice."
        ],
    )

    assert report["answer_relevance_score"] == pytest.approx(1.0)
    assert report["context_support_coverage"] == 1.0
    assert report["potentially_unsupported_sentence_count"] == 0


def test_flags_low_similarity_sentence(monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "embed_texts",
        fake_embeddings,
    )

    report = evaluator.evaluate_rag_answer(
        query="How does billing work?",
        answer="The account password expires tomorrow.",
        retrieved_contexts=[
            "Billing details are provided in the invoice."
        ],
        support_threshold=0.5,
    )

    assert report["potentially_unsupported_sentence_count"] == 1
    assert report["context_support_coverage"] == 0.0


def test_rejects_empty_query():
    with pytest.raises(ValueError):
        evaluator.evaluate_rag_answer(
            query=" ",
            answer="An answer.",
            retrieved_contexts=["A context passage."],
        )


def test_rejects_empty_context():
    with pytest.raises(ValueError):
        evaluator.evaluate_rag_answer(
            query="A question?",
            answer="An answer.",
            retrieved_contexts=[],
        )


def test_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        evaluator.evaluate_rag_answer(
            query="A question?",
            answer="An answer.",
            retrieved_contexts=["A context passage."],
            support_threshold=1.5,
        )
