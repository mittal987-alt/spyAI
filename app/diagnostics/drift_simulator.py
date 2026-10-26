from __future__ import annotations

from typing import Any
import numpy as np
from scipy.stats import ks_2samp
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.diagnostics.drift import benjamini_hochberg, calculate_psi, calculate_wasserstein


def get_drift_simulation_presets() -> list[dict[str, Any]]:
    """Return ready-to-test perturbation scenario presets."""
    return [
        {
            "id": "sensor_drift",
            "name": "Subtle Sensor Degradation",
            "description": "Gradual calibration drift with mild Gaussian noise and positive mean offset on features 0 & 1.",
            "config": {
                "mean_shift": 1.5,
                "variance_scale": 1.2,
                "gaussian_noise": 0.4,
                "outlier_rate": 0.0,
                "label_flip_rate": 0.0,
                "target_features": [0, 1],
            },
        },
        {
            "id": "outlier_shock",
            "name": "Outlier Spikes & Contamination",
            "description": "Sensor glitches and corrupted payloads injecting 8% extreme outlier spikes into feature 0 & 2.",
            "config": {
                "mean_shift": 0.0,
                "variance_scale": 1.0,
                "gaussian_noise": 0.2,
                "outlier_rate": 0.08,
                "outlier_scale": 6.0,
                "label_flip_rate": 0.0,
                "target_features": [0, 2],
            },
        },
        {
            "id": "severe_covariate_shift",
            "name": "Severe Covariate Shift",
            "description": "Major demographic or operational environment shift with large offset (+3.5) and 2.0x variance expansion.",
            "config": {
                "mean_shift": 3.5,
                "variance_scale": 2.0,
                "gaussian_noise": 0.8,
                "outlier_rate": 0.02,
                "outlier_scale": 4.0,
                "label_flip_rate": 0.0,
                "target_features": [0, 1, 2],
            },
        },
        {
            "id": "concept_drift",
            "name": "Concept Shift (Ground Truth Flip)",
            "description": "User behavior or fraud patterns mutate, inverting 20% of class labels alongside feature variance shift.",
            "config": {
                "mean_shift": 0.5,
                "variance_scale": 1.4,
                "gaussian_noise": 0.5,
                "outlier_rate": 0.0,
                "label_flip_rate": 0.20,
                "target_features": [0, 1],
            },
        },
        {
            "id": "clean_baseline",
            "name": "Clean Stationary Control",
            "description": "Zero perturbations applied to verify detector specificity and test false positive rates.",
            "config": {
                "mean_shift": 0.0,
                "variance_scale": 1.0,
                "gaussian_noise": 0.0,
                "outlier_rate": 0.0,
                "label_flip_rate": 0.0,
                "target_features": [],
            },
        },
    ]


def simulate_drift_and_perturbations(
    n_samples: int = 1500,
    n_features: int = 10,
    test_size: float = 0.3,
    random_state: int = 42,
    mean_shift: float = 0.0,
    variance_scale: float = 1.0,
    gaussian_noise: float = 0.0,
    outlier_rate: float = 0.0,
    outlier_scale: float = 5.0,
    label_flip_rate: float = 0.0,
    target_features: list[int] | None = None,
) -> dict[str, Any]:
    """
    Execute synthetic data generation, inject parameterized perturbations,
    benchmark model resilience before vs after, and compute multi-metric drift statistics.
    """
    if n_samples < 200:
        raise ValueError("n_samples must be at least 200")
    if n_features < 4:
        raise ValueError("n_features must be at least 4")
    if not (0.1 <= test_size <= 0.5):
        raise ValueError("test_size must be between 0.1 and 0.5")
    if variance_scale <= 0:
        raise ValueError("variance_scale must be strictly positive")
    if not (0.0 <= outlier_rate <= 0.40):
        raise ValueError("outlier_rate must be between 0.0 and 0.40")
    if not (0.0 <= label_flip_rate <= 0.50):
        raise ValueError("label_flip_rate must be between 0.0 and 0.50")

    rng = np.random.default_rng(random_state)

    # 1. Generate clean classification baseline
    informative_count = min(5, n_features)
    redundant_count = min(2, max(0, n_features - informative_count))

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=informative_count,
        n_redundant=redundant_count,
        n_classes=2,
        class_sep=1.0,
        random_state=random_state,
    )

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_ref, y_ref = X[test_idx].copy(), y[test_idx].copy()
    X_pert, y_pert = X[test_idx].copy(), y[test_idx].copy()

    # Determine targeted feature indices
    if target_features is None or len(target_features) == 0:
        # Default to first 3 features if none specified and any shift is active
        if mean_shift != 0.0 or variance_scale != 1.0 or gaussian_noise > 0.0 or outlier_rate > 0.0:
            target_indices = [0, 1, 2] if n_features >= 3 else list(range(n_features))
        else:
            target_indices = []
    else:
        target_indices = [idx for idx in target_features if 0 <= idx < n_features]

    # 2. Inject Perturbations into X_pert
    for feat_idx in target_indices:
        feat_col = X_pert[:, feat_idx]
        col_mean = float(np.mean(feat_col))
        col_std = float(max(np.std(feat_col), 1e-6))

        # Variance scaling
        if variance_scale != 1.0:
            feat_col = col_mean + (feat_col - col_mean) * variance_scale

        # Mean shift
        if mean_shift != 0.0:
            feat_col = feat_col + mean_shift

        # Gaussian noise
        if gaussian_noise > 0.0:
            feat_col = feat_col + rng.normal(0.0, gaussian_noise, size=len(feat_col))

        # Outlier spikes
        if outlier_rate > 0.0:
            n_outliers = int(outlier_rate * len(feat_col))
            if n_outliers > 0:
                outlier_indices = rng.choice(len(feat_col), size=n_outliers, replace=False)
                spikes = rng.choice([-1.0, 1.0], size=n_outliers) * outlier_scale * col_std
                feat_col[outlier_indices] += spikes

        X_pert[:, feat_idx] = feat_col

    # Label flips (Concept drift)
    if label_flip_rate > 0.0:
        n_flips = int(label_flip_rate * len(y_pert))
        if n_flips > 0:
            flip_indices = rng.choice(len(y_pert), size=n_flips, replace=False)
            y_pert[flip_indices] = 1 - y_pert[flip_indices]

    # 3. Model Benchmark & Stress Test
    model = RandomForestClassifier(n_estimators=30, random_state=random_state, max_depth=6)
    model.fit(X_train, y_train)

    # Baseline performance on clean test set
    pred_ref = model.predict(X_ref)
    base_acc = float(accuracy_score(y_ref, pred_ref))
    base_f1 = float(f1_score(y_ref, pred_ref, zero_division=0))
    base_prec = float(precision_score(y_ref, pred_ref, zero_division=0))
    base_rec = float(recall_score(y_ref, pred_ref, zero_division=0))

    # Perturbed performance on drifted test set
    pred_pert = model.predict(X_pert)
    pert_acc = float(accuracy_score(y_pert, pred_pert))
    pert_f1 = float(f1_score(y_pert, pred_pert, zero_division=0))
    pert_prec = float(precision_score(y_pert, pred_pert, zero_division=0))
    pert_rec = float(recall_score(y_pert, pred_pert, zero_division=0))

    acc_drop = float(base_acc - pert_acc)
    f1_drop = float(base_f1 - pert_f1)

    if acc_drop >= 0.20 or pert_acc < 0.60:
        resilience_rating = "critical_vulnerability"
    elif acc_drop >= 0.10:
        resilience_rating = "moderate_degradation"
    elif acc_drop > 0.03:
        resilience_rating = "minor_impact"
    else:
        resilience_rating = "highly_resilient"

    # 4. Multi-Metric Feature Drift Diagnostics
    raw_p_values = []
    ks_results = []
    psi_results = []
    wasserstein_results = []

    for f_idx in range(n_features):
        ref_series = X_ref[:, f_idx]
        pert_series = X_pert[:, f_idx]

        # KS-Test
        ks_stat, p_val = ks_2samp(ref_series, pert_series)
        raw_p_values.append(float(p_val))
        ks_results.append(float(ks_stat))

        # PSI
        psi_report = calculate_psi(ref_series, pert_series, num_bins=10)
        psi_results.append(psi_report)

        # Wasserstein
        w_dist = calculate_wasserstein(ref_series, pert_series)
        wasserstein_results.append(w_dist)

    # Benjamini-Hochberg FDR correction
    adj_p_values, rejected = benjamini_hochberg(raw_p_values, alpha=0.05)

    feature_drifts = []
    distribution_visuals = []

    for f_idx in range(n_features):
        name = f"feature_{f_idx}"
        ref_s = X_ref[:, f_idx]
        pert_s = X_pert[:, f_idx]

        is_perturbed = f_idx in target_indices
        is_drift_detected = bool(rejected[f_idx])
        psi_item = psi_results[f_idx]
        w_dist_val = float(wasserstein_results[f_idx])

        feature_drifts.append({
            "feature_index": f_idx,
            "feature_name": name,
            "is_targeted_in_simulation": is_perturbed,
            "ks_statistic": float(ks_results[f_idx]),
            "raw_p_value": float(raw_p_values[f_idx]),
            "adjusted_p_value": float(adj_p_values[f_idx]),
            "drift_detected": is_drift_detected,
            "psi_score": float(psi_item["psi_score"]),
            "psi_risk": psi_item["risk_level"],
            "wasserstein_distance": w_dist_val,
            "baseline_mean": float(np.mean(ref_s)),
            "baseline_std": float(np.std(ref_s)),
            "perturbed_mean": float(np.mean(pert_s)),
            "perturbed_std": float(np.std(pert_s)),
            "mean_shift_magnitude": float(abs(np.mean(pert_s) - np.mean(ref_s))),
        })

        # Generate comparative histogram/density data for top 4 features
        if f_idx < 4:
            combined_min = float(min(np.min(ref_s), np.min(pert_s)))
            combined_max = float(max(np.max(ref_s), np.max(pert_s)))
            bin_edges = np.linspace(combined_min, combined_max, 16)
            ref_counts, _ = np.histogram(ref_s, bins=bin_edges)
            pert_counts, _ = np.histogram(pert_s, bins=bin_edges)

            bins_data = []
            for b_idx in range(len(ref_counts)):
                midpoint = float((bin_edges[b_idx] + bin_edges[b_idx + 1]) / 2.0)
                bins_data.append({
                    "bin_center": round(midpoint, 2),
                    "baseline_density": round(float(ref_counts[b_idx] / len(ref_s)), 4),
                    "perturbed_density": round(float(pert_counts[b_idx] / len(pert_s)), 4),
                })

            distribution_visuals.append({
                "feature_name": name,
                "feature_index": f_idx,
                "is_drifted": is_drift_detected,
                "psi_score": float(psi_item["psi_score"]),
                "density_bins": bins_data,
            })

    # Sort feature drifts by PSI descending
    feature_drifts.sort(key=lambda x: x["psi_score"], reverse=True)

    drifted_count = sum(1 for f in feature_drifts if f["drift_detected"])
    critical_psi_count = sum(1 for f in feature_drifts if f["psi_risk"] == "critical")

    # Automated remediation recommendations
    recommendations = []
    if critical_psi_count > 0:
        recommendations.append(
            f"{critical_psi_count} feature(s) exhibit critical PSI (> 0.25). Immediate distribution alignment or feature normalization update recommended."
        )
    if label_flip_rate > 0.10:
        recommendations.append(
            f"Ground truth concept shift detected ({int(label_flip_rate * 100)}% label flips). Model predictions are decoupling from true targets; retraining with fresh labels is required."
        )
    if outlier_rate > 0.03:
        recommendations.append(
            f"Outlier contamination rate ({int(outlier_rate * 100)}%) is elevated. Implement robust input capping or quantile clipping before model inference."
        )
    if acc_drop > 0.10:
        recommendations.append(
            f"Model accuracy collapsed by {(acc_drop * 100):.1f}%. Consider automated fallback to rule-based safety heuristics or shadow model rollback."
        )
    if len(recommendations) == 0:
        recommendations.append(
            "System is operating within expected stability thresholds. No model intervention or feature recalibration needed."
        )

    return {
        "status": "success",
        "dataset_metadata": {
            "total_samples": n_samples,
            "train_size": len(X_train),
            "test_size": len(X_ref),
            "feature_count": n_features,
            "target_features_perturbed": target_indices,
        },
        "applied_perturbations": {
            "mean_shift": mean_shift,
            "variance_scale": variance_scale,
            "gaussian_noise": gaussian_noise,
            "outlier_rate": outlier_rate,
            "outlier_scale": outlier_scale,
            "label_flip_rate": label_flip_rate,
        },
        "model_stress_test": {
            "model_type": "RandomForestClassifier",
            "baseline": {
                "accuracy": base_acc,
                "f1_score": base_f1,
                "precision": base_prec,
                "recall": base_rec,
            },
            "perturbed": {
                "accuracy": pert_acc,
                "f1_score": pert_f1,
                "precision": pert_prec,
                "recall": pert_rec,
            },
            "accuracy_drop": acc_drop,
            "f1_drop": f1_drop,
            "resilience_rating": resilience_rating,
        },
        "drift_summary": {
            "total_features": n_features,
            "drifted_features_count": drifted_count,
            "critical_psi_features_count": critical_psi_count,
            "drift_percentage": round((drifted_count / n_features) * 100, 1),
            "max_psi_score": max((f["psi_score"] for f in feature_drifts), default=0.0),
            "max_wasserstein": max((f["wasserstein_distance"] for f in feature_drifts), default=0.0),
        },
        "feature_drifts": feature_drifts,
        "distribution_visuals": distribution_visuals,
        "recommendations": recommendations,
    }
