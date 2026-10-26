
from typing import Any
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


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


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> dict[str, Any]:
    """
    Calculate Population Stability Index (PSI) and quantile bucket statistics.

    PSI benchmarks:
        < 0.10: Stable / No significant distribution change
        0.10 - 0.25: Moderate drift / Monitor closely
        >= 0.25: Significant drift / Model retraining or intervention required
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()

    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("Inputs for PSI calculation must not be empty.")

    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, percentiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return {
            "psi_score": 0.0,
            "risk_level": "stable",
            "bins": [],
        }

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_len = len(ref)
    cur_len = len(cur)

    ref_pct = (ref_counts + epsilon) / (ref_len + epsilon * len(ref_counts))
    cur_pct = (cur_counts + epsilon) / (cur_len + epsilon * len(cur_counts))

    sub_psi = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    total_psi = float(np.sum(sub_psi))

    if total_psi < 0.10:
        risk_level = "stable"
    elif total_psi < 0.25:
        risk_level = "moderate"
    else:
        risk_level = "significant"

    bins_data = []
    actual_edges = np.percentile(ref, percentiles)
    for i in range(len(ref_counts)):
        low = float(actual_edges[i])
        high = float(actual_edges[min(i + 1, len(actual_edges) - 1)])
        bins_data.append({
            "bin_index": i + 1,
            "label": f"[{low:.2f}, {high:.2f}]",
            "expected_pct": float(round(ref_counts[i] / ref_len * 100, 2)),
            "actual_pct": float(round(cur_counts[i] / cur_len * 100, 2)),
            "sub_psi": float(round(sub_psi[i], 4)),
        })

    return {
        "psi_score": float(round(total_psi, 4)),
        "risk_level": risk_level,
        "bins": bins_data,
    }


def calculate_wasserstein(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """Compute Wasserstein-1 distance (Earth Mover's Distance)."""
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    return float(round(float(wasserstein_distance(ref, cur)), 4))


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
    Detect feature drift using KS tests with multiple-testing correction,
    enriched with Population Stability Index (PSI) and Wasserstein Distance.
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
        ref_col = reference[:, index]
        cur_col = current[:, index]

        statistic, p_value = ks_2samp(
            ref_col,
            cur_col,
            alternative="two-sided",
            method="auto",
        )

        psi_metrics = calculate_psi(ref_col, cur_col)
        w_dist = calculate_wasserstein(ref_col, cur_col)

        raw_results.append({
            "feature": str(feature),
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "psi": psi_metrics,
            "wasserstein_distance": w_dist,
            "summary_stats": {
                "reference_mean": float(round(float(np.mean(ref_col)), 4)),
                "current_mean": float(round(float(np.mean(cur_col)), 4)),
                "reference_std": float(round(float(np.std(ref_col)), 4)),
                "current_std": float(round(float(np.std(cur_col)), 4)),
            },
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
