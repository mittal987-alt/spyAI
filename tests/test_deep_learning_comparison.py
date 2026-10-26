from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.deep_learning.comparison import compare_models


client = TestClient(app)


def test_compare_models_input_validation():
    with pytest.raises(ValueError, match="n_samples must be at least 100"):
        compare_models(n_samples=50, n_features=10, epochs=5)

    with pytest.raises(ValueError, match="n_features must be at least 4"):
        compare_models(n_samples=150, n_features=2, epochs=5)

    with pytest.raises(ValueError, match="epochs must be at least 2"):
        compare_models(n_samples=150, n_features=6, epochs=1)


def test_compare_deep_learning_api_endpoint():
    # Use small dataset and 2 epochs for quick integration test
    payload = {
        "epochs": 2,
        "n_samples": 120,
        "n_features": 6,
        "random_state": 42,
    }

    response = client.post("/api/v1/deep-learning/compare", json=payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert "best_model" in data
    assert "leaderboard" in data
    assert len(data["leaderboard"]) == 4

    best = data["best_model"]
    assert best["rank"] == 1
    assert "best_validation_loss" in best
    assert "best_validation_accuracy" in best
    assert "accuracy_gap" in best
    assert best["best_validation_loss"] >= 0

    # Ensure leaderboard is sorted by best_validation_loss ascending
    losses = [item["best_validation_loss"] for item in data["leaderboard"]]
    assert losses == sorted(losses)

    # Ensure rank numbers match 1..4
    ranks = [item["rank"] for item in data["leaderboard"]]
    assert ranks == [1, 2, 3, 4]


def test_compare_deep_learning_api_validation_error():
    payload = {
        "epochs": 1,  # Invalid: requires >= 2
        "n_samples": 120,
        "n_features": 6,
        "random_state": 42,
    }

    response = client.post("/api/v1/deep-learning/compare", json=payload)
    assert response.status_code == 422
