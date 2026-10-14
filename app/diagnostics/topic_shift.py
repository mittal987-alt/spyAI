
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

from app.diagnostics.embeddings import embed_texts


def analyze_topic_shift(
    reference_texts: list[str],
    current_texts: list[str],
    n_clusters: int = 5,
    random_state: int = 42,
    examples_per_cluster: int = 3,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    """
    Cluster reference and current text using a shared embedding space.

    K-means is fitted on the combined collections so each document is
    assigned to the same cluster system. Cluster proportions are compared
    to identify groups that have grown or shrunk.

    Cluster IDs are numerical group labels, not human-readable topic names.
    """

    if not reference_texts or not current_texts:
        raise ValueError(
            "Reference and current text collections must not be empty"
        )

    all_texts = reference_texts + current_texts

    if any(
        not isinstance(text, str) or not text.strip()
        for text in all_texts
    ):
        raise ValueError(
            "All documents must be non-empty strings"
        )

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")

    if n_clusters > len(all_texts):
        raise ValueError(
            "n_clusters cannot exceed the total document count"
        )

    if examples_per_cluster < 1:
        raise ValueError(
            "examples_per_cluster must be positive"
        )

    reference_count = len(reference_texts)

    embeddings = embed_texts(
        all_texts,
        model_name=model_name,
    )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")

    # Both collections use the same clustering model and labels.
    clusterer = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    labels = clusterer.fit_predict(embeddings)

    reference_labels = labels[:reference_count]
    current_labels = labels[reference_count:]

    reference_counts = Counter(reference_labels.tolist())
    current_counts = Counter(current_labels.tolist())

    reference_total = len(reference_labels)
    current_total = len(current_labels)

    reference_proportions = (
        np.bincount(reference_labels, minlength=n_clusters)
        / reference_total
    )
    current_proportions = (
        np.bincount(current_labels, minlength=n_clusters)
        / current_total
    )

    clusters = []

    for cluster_id in range(n_clusters):
        reference_proportion = float(
            reference_proportions[cluster_id]
        )
        current_proportion = float(
            current_proportions[cluster_id]
        )

        change = current_proportion - reference_proportion

        if change > 0:
            direction = "increased"
        elif change < 0:
            direction = "decreased"
        else:
            direction = "unchanged"

        # Representative examples are nearest to the cluster centroid.
        member_indices = np.flatnonzero(labels == cluster_id)
        centroid = clusterer.cluster_centers_[cluster_id]

        if len(member_indices):
            distances = np.linalg.norm(
                embeddings[member_indices] - centroid,
                axis=1,
            )

            ranked_indices = member_indices[
                np.argsort(distances)
            ][:examples_per_cluster]

            representative_examples = [
                all_texts[int(index)]
                for index in ranked_indices
            ]
        else:
            representative_examples = []

        clusters.append(
            {
                "cluster_id": cluster_id,
                "reference_document_count": int(
                    reference_counts.get(cluster_id, 0)
                ),
                "current_document_count": int(
                    current_counts.get(cluster_id, 0)
                ),
                "reference_proportion": reference_proportion,
                "current_proportion": current_proportion,
                "proportion_change": float(change),
                "direction": direction,
                "representative_examples": representative_examples,
            }
        )

    # Total variation distance ranges from 0 to 1.
    total_variation_distance = float(
        0.5
        * np.abs(
            reference_proportions - current_proportions
        ).sum()
    )

    growing_clusters = sorted(
        [
            cluster
            for cluster in clusters
            if cluster["proportion_change"] > 0
        ],
        key=lambda item: item["proportion_change"],
        reverse=True,
    )

    shrinking_clusters = sorted(
        [
            cluster
            for cluster in clusters
            if cluster["proportion_change"] < 0
        ],
        key=lambda item: item["proportion_change"],
    )

    return {
        "method": "Sentence embeddings + K-means clustering",
        "reference_document_count": reference_total,
        "current_document_count": current_total,
        "embedding_dimension": int(embeddings.shape[1]),
        "n_clusters": n_clusters,
        "total_variation_distance": total_variation_distance,
        "clusters": clusters,
        "largest_increases": growing_clusters[:3],
        "largest_decreases": shrinking_clusters[:3],
        "interpretation": (
            "Cluster proportions summarize how the text mix changed. "
            "Review representative examples to understand the groups."
        ),
        "limitations": [
            "Cluster IDs are not automatically meaningful topic names.",
            "The combined dataset determines cluster centroids.",
            "K-means requires a chosen cluster count.",
            "Proportion changes do not establish statistical significance.",
            "Text groups can mix several topics or split one topic into several clusters.",
        ],
    }
