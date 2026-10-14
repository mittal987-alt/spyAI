
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from app.config import (
    N_FEATURES,
    N_SAMPLES,
    RANDOM_STATE,
    TEST_SIZE,
)


def load_datasets():
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.01,
        random_state=RANDOM_STATE,
    )

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rng = np.random.default_rng(RANDOM_STATE)
    X_drifted = X_test.copy()

    # Simulate a covariate-distribution shift.
    X_drifted[:, 0] += 3.0
    X_drifted[:, 1] *= 2.5
    X_drifted[:, 2] += rng.normal(
        loc=0.0,
        scale=1.5,
        size=len(X_drifted),
    )

    feature_names = [
        f"feature_{i}" for i in range(X.shape[1])
    ]

    return {
        "X_train": X_train,
        "X_reference": X_test,
        "X_drifted": X_drifted,
        "y_train": y_train,
        "y_reference": y_test,
        "y_drifted": y_test.copy(),
        "feature_names": feature_names,
    }
