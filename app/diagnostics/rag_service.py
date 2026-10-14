
from __future__ import annotations

from datetime import datetime, timezone

from app.diagnostics.rag_evaluation import evaluate_retrieval
from app.diagnostics.rag_answer_evaluation import evaluate_rag_answer


MAX_QUERY_LENGTH = 2000
MAX_ANSWER_LENGTH = 10000
MAX_DOCUMENTS = 100
MAX_DOCUMENT_LENGTH = 10000


def run_rag_evaluation(
    query: str,
    answer: str,
    retrieved_documents: list[dict],
    relevant_document_ids: list[str],
    k: int = 5,
    relevance_threshold: float = 0.45,
    support_threshold: float = 0.50,
) -> dict:
    """Combine retrieval and generated-answer diagnostics."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be non-empty text")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"query cannot exceed {MAX_QUERY_LENGTH} characters"
        )

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be non-empty text")

    if len(answer) > MAX_ANSWER_LENGTH:
        raise ValueError(
            f"answer cannot exceed {MAX_ANSWER_LENGTH} characters"
        )

    if not retrieved_documents:
        raise ValueError("retrieved_documents must not be empty")

    if len(retrieved_documents) > MAX_DOCUMENTS:
        raise ValueError(
            f"At most {MAX_DOCUMENTS} retrieved documents are allowed"
        )

    if not relevant_document_ids:
        raise ValueError(
            "relevant_document_ids must not be empty"
        )

    if k < 1:
        raise ValueError("k must be positive")

    for index, document in enumerate(retrieved_documents):
        if not isinstance(document, dict):
            raise ValueError(
                f"Retrieved document at index {index} must be an object"
            )

        document_id = document.get("id")
        document_text = document.get("text")

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                f"Document at index {index} needs a non-empty string id"
            )

        if not isinstance(document_text, str) or not document_text.strip():
            raise ValueError(
                f"Document at index {index} needs non-empty text"
            )

        if len(document_text) > MAX_DOCUMENT_LENGTH:
            raise ValueError(
                f"Document at index {index} exceeds "
                f"{MAX_DOCUMENT_LENGTH} characters"
            )

    if any(
        not isinstance(doc_id, str) or not doc_id.strip()
        for doc_id in relevant_document_ids
    ):
        raise ValueError(
            "Relevant document IDs must be non-empty strings"
        )

    if len(set(relevant_document_ids)) != len(relevant_document_ids):
        raise ValueError(
            "relevant_document_ids must not contain duplicates"
        )

    if not 0 <= relevance_threshold <= 1:
        raise ValueError(
            "relevance_threshold must be between 0 and 1"
        )

    if not 0 <= support_threshold <= 1:
        raise ValueError(
            "support_threshold must be between 0 and 1"
        )

    # Retrieval evaluation uses supplied relevance annotations.
    retrieval_report = evaluate_retrieval(
        query=query,
        retrieved_documents=retrieved_documents,
        relevant_document_ids=relevant_document_ids,
        k=k,
    )

    # Answer evaluation uses the text of the retrieved documents.
    answer_report = evaluate_rag_answer(
        query=query,
        answer=answer,
        retrieved_contexts=[
            document["text"]
            for document in retrieved_documents
        ],
        relevance_threshold=relevance_threshold,
        support_threshold=support_threshold,
    )

    retrieval_precision = retrieval_report["precision_at_k"]
    answer_relevance = answer_report["answer_relevance_score"]
    support_coverage = answer_report["context_support_coverage"]

    # These are diagnostic flags, not a calibrated production risk score.
    findings = []

    if retrieval_precision < 0.5:
        findings.append({
            "type": "low_retrieval_precision",
            "severity": "medium",
            "message": (
                "Fewer than half of the documents in the evaluated "
                "top-k results were labelled relevant."
            ),
        })

    if answer_relevance < relevance_threshold:
        findings.append({
            "type": "low_answer_relevance",
            "severity": "medium",
            "message": (
                "The answer has low semantic similarity to the query "
                "under the configured threshold."
            ),
        })

    if answer_report["potentially_unsupported_sentence_count"] > 0:
        findings.append({
            "type": "potentially_unsupported_claims",
            "severity": "review",
            "message": (
                "Some answer sentences did not meet the context "
                "similarity threshold and should be reviewed."
            ),
        })

    return {
        "service": "ModelAutopsy AI RAG Evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieved_document_count": len(retrieved_documents),
        "relevant_document_count": len(set(relevant_document_ids)),
        "evaluation": {
            "retrieval": retrieval_report,
            "answer": answer_report,
        },
        "summary": {
            "precision_at_k": retrieval_precision,
            "recall_at_k": retrieval_report["recall_at_k"],
            "reciprocal_rank": retrieval_report["reciprocal_rank"],
            "ndcg_at_k": retrieval_report["ndcg_at_k"],
            "answer_relevance_score": answer_relevance,
            "context_support_coverage": support_coverage,
            "potentially_unsupported_sentence_count": (
                answer_report[
                    "potentially_unsupported_sentence_count"
                ]
            ),
            "finding_count": len(findings),
        },
        "findings": findings,
        "limitations": [
            (
                "Retrieval metrics depend on the quality and completeness "
                "of supplied relevance labels."
            ),
            (
                "Embedding similarity does not prove factual correctness "
                "or logical entailment."
            ),
            (
                "The finding thresholds are initial heuristics and need "
                "calibration against human-reviewed examples."
            ),
        ],
    }
