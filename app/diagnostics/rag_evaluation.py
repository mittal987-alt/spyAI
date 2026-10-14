
from __future__ import annotations

import numpy as np

from app.diagnostics.embeddings import embed_texts


def evaluate_retrieval(
    query: str,
    retrieved_documents: list[dict],
    relevant_document_ids: list[str],
    k: int = 5,
) -> dict:
    """
    Evaluate a ranked list of retrieved documents.

    Each retrieved document must contain:
        {"id": "doc-1", "text": "Document contents..."}

    relevant_document_ids represents known relevant documents for
    this query. These labels must come from a trusted evaluation set
    or human annotation.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be non-empty text")

    if k < 1:
        raise ValueError("k must be positive")

    if not retrieved_documents:
        raise ValueError("retrieved_documents must not be empty")

    if not relevant_document_ids:
        raise ValueError(
            "relevant_document_ids must not be empty"
        )

    if any(
        not isinstance(document, dict)
        or not isinstance(document.get("id"), str)
        or not document["id"].strip()
        or not isinstance(document.get("text"), str)
        or not document["text"].strip()
        for document in retrieved_documents
    ):
        raise ValueError(
            "Each retrieved document must have a non-empty string id and text"
        )

    if any(
        not isinstance(doc_id, str) or not doc_id.strip()
        for doc_id in relevant_document_ids
    ):
        raise ValueError(
            "Relevant document IDs must be non-empty strings"
        )

    retrieved_ids = [
        document["id"]
        for document in retrieved_documents
    ]

    if len(set(retrieved_ids)) != len(retrieved_ids):
        raise ValueError(
            "Retrieved document IDs must be unique"
        )

    relevant_ids = set(relevant_document_ids)
    if not relevant_ids:
        raise ValueError("Relevant document IDs must not be empty")

    top_documents = retrieved_documents[:k]
    top_ids = [document["id"] for document in top_documents]

    relevant_in_top_k = [
        doc_id for doc_id in top_ids
        if doc_id in relevant_ids
    ]

    true_positives = len(relevant_in_top_k)

    precision_at_k = true_positives / len(top_documents)
    recall_at_k = true_positives / len(relevant_ids)

    reciprocal_rank = 0.0

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    # Binary relevance nDCG@k.
    relevance = [
        1.0 if doc_id in relevant_ids else 0.0
        for doc_id in top_ids
    ]

    discounts = np.log2(
        np.arange(2, len(relevance) + 2, dtype=float)
    )

    dcg = float(
        np.sum(np.asarray(relevance, dtype=float) / discounts)
    )

    ideal_relevance = np.ones(
        min(len(relevant_ids), len(top_documents)),
        dtype=float,
    )

    ideal_discounts = np.log2(
        np.arange(2, len(ideal_relevance) + 2, dtype=float)
    )

    ideal_dcg = float(
        np.sum(ideal_relevance / ideal_discounts)
    )

    ndcg_at_k = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    # Measure semantic redundancy among retrieved documents.
    if len(top_documents) >= 2:
        texts = [document["text"] for document in top_documents]

        embeddings = embed_texts(texts)

        similarities = embeddings @ embeddings.T
        upper_triangle = similarities[
            np.triu_indices(len(top_documents), k=1)
        ]

        mean_pairwise_similarity = float(
            np.mean(upper_triangle)
        )
    else:
        mean_pairwise_similarity = None

    return {
        "query": query,
        "retrieved_count": len(retrieved_documents),
        "evaluation_k": k,
        "relevant_document_count": len(relevant_ids),
        "relevant_retrieved_count": true_positives,
        "precision_at_k": float(precision_at_k),
        "recall_at_k": float(recall_at_k),
        "reciprocal_rank": float(reciprocal_rank),
        "ndcg_at_k": float(ndcg_at_k),
        "mean_pairwise_similarity": mean_pairwise_similarity,
        "redundancy_note": (
            "Higher pairwise similarity can indicate redundancy, "
            "but similar documents may also provide useful corroboration."
        ),
        "limitations": [
            "Metrics depend on the completeness of relevance labels.",
            "Recall is measured against the supplied relevant document IDs.",
            "Similarity is a proxy for redundancy, not a definitive judgment.",
            "These metrics evaluate retrieval, not the factual correctness of an LLM answer.",
        ],
    }
