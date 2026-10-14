
import numpy as np
from scipy.stats import ks_2samp


def benjamini_hochberg(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg correction.

    Returns adjusted p-values and rejection decisions in the
    original input order.
    """
    p_values = np.asarray(p_values, dtype=float)

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")

    if not np.isfinite(p_values).all():
        raise ValueError("p_values must be finite.")

    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must be between 0 and 1.")

    count = len(p_values)

    if count == 0:
        return [], []

    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.empty(count, dtype=float)
    running_min = 1.0

    for index in range(count - 1, -1, -1):
        rank = index + 1
        adjusted = sorted_p[index] * count / rank
        running_min = min(running_min, adjusted)
        adjusted_sorted[index] = min(running_min, 1.0)

    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted

    rejected = adjusted < alpha

    return adjusted.tolist(), rejected.tolist()


def detect_feature_drift(
    reference,
    current,
    feature_names,
    alpha=0.05,
    effect_threshold=0.10,
    medium_threshold=0.10,
    high_threshold=0.30,
):
    """
    Detect feature drift using KS tests with multiple-testing correction.

    The KS statistic measures distributional difference. Adjusted
    p-values control the false discovery rate across tested features.

    A feature is flagged only when both statistical significance
    and the configured minimum effect-size threshold are satisfied.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    feature_names = list(feature_names)

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    if not 0 <= effect_threshold <= 1:
        raise ValueError("effect_threshold must be between 0 and 1.")

    if not 0 <= medium_threshold <= 1:
        raise ValueError("medium_threshold must be between 0 and 1.")

    if not 0 <= high_threshold <= 1:
        raise ValueError("high_threshold must be between 0 and 1.")

    if medium_threshold > high_threshold:
        raise ValueError(
            "medium_threshold cannot exceed high_threshold."
        )

    if reference.ndim != 2 or current.ndim != 2:
        raise ValueError("Inputs must be two-dimensional arrays.")

    if reference.shape[1] != current.shape[1]:
        raise ValueError("Feature dimensions do not match.")

    if reference.shape[1] != len(feature_names):
        raise ValueError(
            "Feature names do not match feature dimensions."
        )

    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature names must be unique.")

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Input datasets must not be empty.")

    if not np.isfinite(reference).all():
        raise ValueError("Reference data must contain finite values.")

    if not np.isfinite(current).all():
        raise ValueError("Current data must contain finite values.")

    raw_results = []

    for index, feature in enumerate(feature_names):
        statistic, p_value = ks_2samp(
            reference[:, index],
            current[:, index],
            alternative="two-sided",
            method="auto",
        )

        raw_results.append({
            "feature": str(feature),
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
        })

    adjusted_p_values, _ = benjamini_hochberg(
        [item["p_value"] for item in raw_results],
        alpha=alpha,
    )

    results = []

    for item, adjusted_p in zip(raw_results, adjusted_p_values):
        effect_size = item["ks_statistic"]
        statistically_significant = adjusted_p < alpha

        drift_detected = (
            statistically_significant
            and effect_size >= effect_threshold
        )

        if not drift_detected:
            severity = "none"
        elif effect_size >= high_threshold:
            severity = "high"
        elif effect_size >= medium_threshold:
            severity = "medium"
        else:
            severity = "low"

        results.append({
            **item,
            "adjusted_p_value": float(adjusted_p),
            "effect_size": float(effect_size),
            "statistically_significant": bool(
                statistically_significant
            ),
            "drift_detected": bool(drift_detected),
            "severity": severity,
        })

    return results
