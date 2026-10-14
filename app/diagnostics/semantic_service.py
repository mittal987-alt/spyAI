
from __future__ import annotations

from datetime import datetime, timezone

from app.diagnostics.semantic_drift import detect_semantic_drift
from app.diagnostics.topic_shift import analyze_topic_shift


MAX_DOCUMENTS_PER_COLLECTION = 500
MAX_DOCUMENT_LENGTH = 2000


def run_semantic_monitoring(
    reference_texts: list[str],
    current_texts: list[str],
    n_clusters: int = 5,
    n_permutations: int = 199,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """Run semantic-drift detection and topic-shift analysis."""

    for name, documents in (
        ("reference_texts", reference_texts),
        ("current_texts", current_texts),
    ):
        if not documents:
            raise ValueError(f"{name} must not be empty")

        if len(documents) > MAX_DOCUMENTS_PER_COLLECTION:
            raise ValueError(
                f"{name} cannot contain more than "
                f"{MAX_DOCUMENTS_PER_COLLECTION} documents"
            )

        if any(
            not isinstance(document, str) or not document.strip()
            for document in documents
        ):
            raise ValueError(
                f"Every document in {name} must be non-empty text"
            )

        if any(
            len(document) > MAX_DOCUMENT_LENGTH
            for document in documents
        ):
            raise ValueError(
                f"Documents in {name} cannot exceed "
                f"{MAX_DOCUMENT_LENGTH} characters"
            )

    if len(reference_texts) < 5 or len(current_texts) < 5:
        raise ValueError(
            "At least 5 documents are required in each collection"
        )

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")

    if n_clusters > len(reference_texts) + len(current_texts):
        raise ValueError(
            "n_clusters cannot exceed the total document count"
        )

    if not 19 <= n_permutations <= 999:
        raise ValueError(
            "n_permutations must be between 19 and 999"
        )

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    semantic_report = detect_semantic_drift(
        reference_texts=reference_texts,
        current_texts=current_texts,
        n_permutations=n_permutations,
        alpha=alpha,
        random_state=random_state,
    )

    topic_report = analyze_topic_shift(
        reference_texts=reference_texts,
        current_texts=current_texts,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    return {
        "service": "ModelAutopsy AI Semantic Monitoring",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_document_count": len(reference_texts),
        "current_document_count": len(current_texts),
        "status": (
            "drift_detected"
            if semantic_report["drift_detected"]
            else "no_significant_drift_detected"
        ),
        "semantic_drift": semantic_report,
        "topic_shift": topic_report,
        "limitations": [
            "Semantic drift is not proof of model-performance degradation.",
            "Topic clusters are not automatically named or validated topics.",
            "The analysis uses embeddings and synthetic statistical thresholds "
            "that should be calibrated for the target application.",
        ],
    }
