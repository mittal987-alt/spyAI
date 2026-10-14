
import numpy as np
import pytest

import app.diagnostics.semantic_drift as semantic_drift


def fake_embeddings(texts, model_name=None):
    rng = np.random.default_rng(42)

    if texts[0].startswith("shifted"):
        return rng.normal(
            loc=2.0,
            scale=1.0,
            size=(len(texts), 8),
        ).astype(np.float32)

    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=(len(texts), 8),
    ).astype(np.float32)


def test_detect_semantic_drift_returns_expected_fields(monkeypatch):
    monkeypatch.setattr(
        semantic_drift,
        "embed_texts",
        fake_embeddings,
    )

    reference = [f"reference document {i}" for i in range(30)]
    current = [f"reference document {i}" for i in range(30, 60)]

    report = semantic_drift.detect_semantic_drift(
        reference,
        current,
        n_components=3,
        n_permutations=19,
    )

    assert "observed_distance" in report
    assert "p_value" in report
    assert "drift_detected" in report
    assert report["severity"] in {
        "none",
        "low",
        "medium",
        "high",
    }
    assert 0 <= report["p_value"] <= 1


def test_detect_semantic_drift_rejects_empty_collection():
    with pytest.raises(ValueError):
        semantic_drift.detect_semantic_drift(
            [],
            ["document"] * 5,
        )


def test_detect_semantic_drift_requires_enough_documents():
    with pytest.raises(ValueError):
        semantic_drift.detect_semantic_drift(
            ["document"] * 4,
            ["document"] * 5,
        )


def test_detect_semantic_drift_validates_permutations():
    with pytest.raises(ValueError):
        semantic_drift.detect_semantic_drift(
            ["reference"] * 5,
            ["current"] * 5,
            n_permutations=5,
        )
