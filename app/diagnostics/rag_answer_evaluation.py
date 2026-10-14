
from __future__ import annotations

import re

import numpy as np

from app.diagnostics.embeddings import embed_texts


def _split_sentences(text: str) -> list[str]:
    """Split an answer into basic sentence-level units."""

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def _validate_text_list(
    values: list[str],
    field_name: str,
) -> None:
    if not values:
        raise ValueError(f"{field_name} must not be empty")

    if any(
        not isinstance(value, str) or not value.strip()
        for value in values
    ):
        raise ValueError(
            f"Every item in {field_name} must be non-empty text"
        )


def evaluate_rag_answer(
    query: str,
    answer: str,
    retrieved_contexts: list[str],
    relevance_threshold: float = 0.45,
    support_threshold: float = 0.50,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    """
    Evaluate answer relevance and potential context support.

    Embedding similarity is a heuristic. It does not establish that a
    statement is factually correct or logically entailed by the context.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be non-empty text")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be non-empty text")

    _validate_text_list(
        retrieved_contexts,
        "retrieved_contexts",
    )

    if not 0 <= relevance_threshold <= 1:
        raise ValueError(
            "relevance_threshold must be between 0 and 1"
        )

    if not 0 <= support_threshold <= 1:
        raise ValueError(
            "support_threshold must be between 0 and 1"
        )

    answer_sentences = _split_sentences(answer)

    if not answer_sentences:
        raise ValueError("answer must contain at least one sentence")

    # Encode the query, complete answer, answer sentences, and contexts
    # in a single call for consistent embeddings.
    texts = (
        [query, answer]
        + answer_sentences
        + retrieved_contexts
    )

    embeddings = embed_texts(
        texts,
        model_name=model_name,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("Unexpected embedding output shape")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")

    # embed_texts returns normalized vectors, so dot products are cosine
    # similarities. Clip negative values to zero for heuristic scores.
    query_vector = embeddings[0]
    answer_vector = embeddings[1]

    answer_relevance = float(
        np.clip(np.dot(query_vector, answer_vector), 0.0, 1.0)
    )

    sentence_start = 2
    sentence_end = sentence_start + len(answer_sentences)

    sentence_vectors = embeddings[sentence_start:sentence_end]
    context_vectors = embeddings[sentence_end:]

    similarity_matrix = sentence_vectors @ context_vectors.T

    sentence_support = []

    for index, sentence in enumerate(answer_sentences):
        similarities = similarity_matrix[index]
        best_index = int(np.argmax(similarities))
        best_similarity = float(
            np.clip(similarities[best_index], 0.0, 1.0)
        )

        sentence_support.append(
            {
                "sentence": sentence,
                "support_score": best_similarity,
                "best_context_index": best_index,
                "supported_by_similarity_threshold": (
                    best_similarity >= support_threshold
                ),
            }
        )

    supported_count = sum(
        item["supported_by_similarity_threshold"]
        for item in sentence_support
    )

    total_sentences = len(sentence_support)

    support_coverage = supported_count / total_sentences

    potentially_unsupported = [
        item
        for item in sentence_support
        if not item["supported_by_similarity_threshold"]
    ]

    return {
        "query": query,
        "answer": answer,
        "retrieved_context_count": len(retrieved_contexts),
        "answer_sentence_count": total_sentences,
        "answer_relevance_score": answer_relevance,
        "answer_relevant_by_threshold": (
            answer_relevance >= relevance_threshold
        ),
        "context_support_coverage": float(support_coverage),
        "supported_sentence_count": int(supported_count),
        "potentially_unsupported_sentence_count": len(
            potentially_unsupported
        ),
        "sentence_analysis": sentence_support,
        "potentially_unsupported_sentences": (
            potentially_unsupported
        ),
        "thresholds": {
            "relevance": relevance_threshold,
            "support": support_threshold,
        },
        "limitations": [
            (
                "Embedding similarity is not proof of factual support "
                "or logical entailment."
            ),
            (
                "A supported sentence can still be false if the "
                "retrieved context is false."
            ),
            (
                "A low similarity score can occur for valid paraphrases "
                "or statements requiring multiple context passages."
            ),
            (
                "Thresholds require calibration on human-labelled "
                "examples from the target application."
            ),
        ],
    }
