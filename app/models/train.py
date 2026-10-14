
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import ARTIFACTS_DIR, RANDOM_STATE

MODEL_PATH = ARTIFACTS_DIR / "baseline_model.joblib"


def train_and_save_model(X_train, y_train):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run training first."
        )

    return joblib.load(MODEL_PATH)
