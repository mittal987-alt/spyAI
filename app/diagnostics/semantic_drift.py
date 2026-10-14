
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

from app.diagnostics.embeddings import embed_texts


def _distribution_distance(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """Average normalized Wasserstein distance across components."""

    distances = []

    for component in range(reference.shape[1]):
        ref_values = reference[:, component]
        cur_values = current[:, component]

        scale = float(np.std(ref_values))

        # Avoid division by zero for constant reference components.
        if scale < 1e-8:
            scale = 1e-8

        distance = wasserstein_distance(
            ref_values,
            cur_values,
        )

        distances.append(float(distance / scale))

    return float(np.mean(distances))


def detect_semantic_drift(
    reference_texts: list[str],
    current_texts: list[str],
    n_components: int = 5,
    n_permutations: int = 199,
    alpha: float = 0.05,
    random_state: int = 42,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict:
    """
    Compare reference and current text distributions using embeddings.

    PCA is fitted on reference embeddings only. The permutation test
    compares the observed projected distribution distance with distances
    obtained after randomly shuffling reference/current membership.

    The reported effect size is a normalized projected Wasserstein
    distance. It is not a probability of model failure.
    """

    if not reference_texts or not current_texts:
        raise ValueError(
            "Reference and current text collections must not be empty"
        )

    if any(
        not isinstance(text, str) or not text.strip()
        for text in reference_texts + current_texts
    ):
        raise ValueError(
            "All documents must be non-empty strings"
        )

    if len(reference_texts) < 5 or len(current_texts) < 5:
        raise ValueError(
            "Each collection must contain at least 5 documents"
        )

    if n_components < 1:
        raise ValueError("n_components must be positive")

    if n_permutations < 19:
        raise ValueError("n_permutations must be at least 19")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    reference_embeddings = embed_texts(
        reference_texts,
        model_name=model_name,
    )
    current_embeddings = embed_texts(
        current_texts,
        model_name=model_name,
    )

    if reference_embeddings.shape[1] != current_embeddings.shape[1]:
        raise ValueError(
            "Reference and current embeddings have different dimensions"
        )

    # Fit PCA using only the reference collection.
    components = min(
        n_components,
        reference_embeddings.shape[1],
        len(reference_texts) - 1,
    )

    pca = PCA(n_components=components)
    reference_projected = pca.fit_transform(reference_embeddings)
    current_projected = pca.transform(current_embeddings)

    observed_distance = _distribution_distance(
        reference_projected,
        current_projected,
    )

    # Permute collection membership to estimate a null distribution.
    combined = np.vstack(
        [reference_projected, current_projected]
    )
    reference_count = len(reference_projected)
    rng = np.random.default_rng(random_state)

    permutation_distances = []

    for _ in range(n_permutations):
        indices = rng.permutation(len(combined))

        perm_reference = combined[
            indices[:reference_count]
        ]
        perm_current = combined[
            indices[reference_count:]
        ]

        distance = _distribution_distance(
            perm_reference,
            perm_current,
        )
        permutation_distances.append(distance)

    # Add-one correction avoids reporting a zero p-value.
    p_value = (
        1
        + sum(
            value >= observed_distance
            for value in permutation_distances
        )
    ) / (n_permutations + 1)

    drift_detected = p_value < alpha

    if not drift_detected:
        severity = "none"
    elif observed_distance >= 1.0:
        severity = "high"
    elif observed_distance >= 0.5:
        severity = "medium"
    else:
        severity = "low"

    return {
        "method": "PCA + normalized Wasserstein distance + permutation test",
        "reference_document_count": len(reference_texts),
        "current_document_count": len(current_texts),
        "embedding_dimension": int(reference_embeddings.shape[1]),
        "pca_components": int(components),
        "explained_variance_ratio": [
            float(value)
            for value in pca.explained_variance_ratio_
        ],
        "observed_distance": float(observed_distance),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "drift_detected": bool(drift_detected),
        "severity": severity,
        "n_permutations": int(n_permutations),
        "interpretation": (
            "A semantic distribution change was detected in the "
            "projected embeddings."
            if drift_detected
            else (
                "The test did not detect a statistically significant "
                "change in the projected embedding distributions."
            )
        ),
        "limitations": [
            "The test compares distributions in a reduced PCA space.",
            "Results depend on the embedding model and reference dataset.",
            "A non-significant result does not prove the distributions are identical.",
            "Severity thresholds are initial heuristics, not calibrated risk levels.",
            "The permutation test assumes exchangeable observations.",
        ],
    }
 