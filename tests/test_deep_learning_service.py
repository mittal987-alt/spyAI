
from app.deep_learning.service import run_deep_learning_diagnosis


def test_diagnosis_returns_expected_structure(tmp_path, monkeypatch):
    from app.config import MLFLOW_TRACKING_URI

    # Use an isolated tracking database for this test.
    monkeypatch.setattr(
        "app.deep_learning.service.MLFLOW_TRACKING_URI",
        f"sqlite:///{(tmp_path / 'test_mlflow.db').as_posix()}",
    )

    result = run_deep_learning_diagnosis(
        epochs=2,
        batch_size=32,
        n_samples=200,
        n_features=4,
        hidden_units=8,
        random_state=42,
    )

    assert result["experiment"] == "ModelAutopsy-DeepLearning-Diagnostics"
    assert result["run_id"]
    assert "diagnostics" in result
    assert "training_history" in result
    assert len(result["training_history"]["train_loss"]) == 2


def test_diagnosis_metrics_are_finite(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.deep_learning.service.MLFLOW_TRACKING_URI",
        f"sqlite:///{(tmp_path / 'metrics_mlflow.db').as_posix()}",
    )

    result = run_deep_learning_diagnosis(
        epochs=2,
        batch_size=32,
        n_samples=200,
        n_features=4,
        hidden_units=8,
    )

    diagnostics = result["diagnostics"]

    assert 0 <= diagnostics["final_train_accuracy"] <= 1
    assert 0 <= diagnostics["final_validation_accuracy"] <= 1
    assert diagnostics["best_validation_epoch"] >= 1
