
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np

from app.diagnostics.drift import detect_feature_drift


def run_experiment():
    X, y = make_classification(
        n_samples=6000,
        n_features=8,
        n_informative=5,
        n_redundant=2,
        class_sep=1.0,
        random_state=42,
    )

    names = [f"feature_{i}" for i in range(X.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    baseline_predictions = model.predict(X_test)

    # Simulate a change in incoming feature distributions.
    rng = np.random.default_rng(42)
    X_changed = X_test.copy()
    X_changed[:, 0] += 3.0
    X_changed[:, 1] *= 2.5
    X_changed[:, 2] += rng.normal(
        loc=0.0, scale=1.5, size=len(X_changed)
    )

    changed_predictions = model.predict(X_changed)

    baseline_metrics = {
        "accuracy": round(
            float(accuracy_score(y_test, baseline_predictions)), 4
        ),
        "f1": round(
            float(f1_score(y_test, baseline_predictions)), 4
        ),
    }

    changed_metrics = {
        "accuracy": round(
            float(accuracy_score(y_test, changed_predictions)), 4
        ),
        "f1": round(
            float(f1_score(y_test, changed_predictions)), 4
        ),
    }

    drift = detect_feature_drift(
        X_train, X_changed, names
    )

    return {
        "experiment": "synthetic_feature_drift",
        "baseline_metrics": baseline_metrics,
        "changed_metrics": changed_metrics,
        "accuracy_change": round(
            changed_metrics["accuracy"]
            - baseline_metrics["accuracy"],
            4,
        ),
        "drifted_features": [
            item["feature"]
            for item in drift
            if item["drift_detected"]
        ],
        "feature_drift": drift,
        "note": (
            "This is a controlled synthetic experiment. "
            "Drift does not necessarily imply model degradation."
        ),
    }
