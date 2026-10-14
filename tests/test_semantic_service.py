
import pytest

import app.diagnostics.semantic_service as service


def test_semantic_monitoring_combines_reports(monkeypatch):
    def fake_drift(**kwargs):
        return {
            "drift_detected": True,
            "severity": "medium",
            "p_value": 0.01,
        }

    def fake_topics(**kwargs):
        return {
            "n_clusters": kwargs["n_clusters"],
            "total_variation_distance": 0.3,
            "clusters": [],
        }

    monkeypatch.setattr(
        service,
        "detect_semantic_drift",
        fake_drift,
    )
    monkeypatch.setattr(
        service,
        "analyze_topic_shift",
        fake_topics,
    )

    report = service.run_semantic_monitoring(
        reference_texts=[f"reference {i}" for i in range(10)],
        current_texts=[f"current {i}" for i in range(10)],
        n_clusters=3,
    )

    assert report["status"] == "drift_detected"
    assert report["semantic_drift"]["p_value"] == 0.01
    assert report["topic_shift"]["n_clusters"] == 3
    assert report["reference_document_count"] == 10
    assert report["current_document_count"] == 10


def test_semantic_monitoring_rejects_too_few_documents():
    with pytest.raises(ValueError):
        service.run_semantic_monitoring(
            reference_texts=["one", "two", "three", "four"],
            current_texts=[f"current {i}" for i in range(5)],
        )


def test_semantic_monitoring_rejects_long_documents():
    with pytest.raises(ValueError):
        service.run_semantic_monitoring(
            reference_texts=["x" * 2001] + [
                f"reference {i}" for i in range(4)
            ],
            current_texts=[f"current {i}" for i in range(5)],
        )


def test_semantic_monitoring_rejects_invalid_permutations():
    with pytest.raises(ValueError):
        service.run_semantic_monitoring(
            reference_texts=[f"reference {i}" for i in range(5)],
            current_texts=[f"current {i}" for i in range(5)],
            n_permutations=5,
        )
