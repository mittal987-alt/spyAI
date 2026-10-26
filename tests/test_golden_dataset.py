"""
Unit tests for Synthetic Golden Dataset & Benchmark Generator for RAG.
"""

import pytest
from app.diagnostics.golden_dataset import (
    DOMAIN_PRESETS,
    generate_golden_dataset,
)


def test_domain_presets_exist():
    assert "cloud_infrastructure" in DOMAIN_PRESETS
    assert "mlops_governance" in DOMAIN_PRESETS
    assert len(DOMAIN_PRESETS["cloud_infrastructure"]["text"]) > 100


def test_generate_golden_dataset_from_text():
    sample_text = (
        "Vector databases index dense embeddings using hierarchical navigable small world (HNSW) graphs. "
        "Approximate nearest neighbor (ANN) search reduces search latency from linear O(N) to sub-linear time. "
        "Reciprocal Rank Fusion (RRF) combines sparse lexical BM25 ranks with dense semantic cosine ranks "
        "to deliver higher mean reciprocal rank (MRR) across heterogeneous enterprise technical queries."
    )

    cases = generate_golden_dataset(
        sample_text,
        num_pairs=3,
        include_hallucination_cases=True,
    )

    assert len(cases) >= 2
    for case in cases:
        assert "query" in case
        assert "answer" in case
        assert "retrieved_documents" in case
        assert "relevant_document_ids" in case
        assert len(case["retrieved_documents"]) >= 1
        assert case["test_type"] in {"factual_benchmark", "adversarial_hallucination"}


def test_empty_document_raises_error():
    with pytest.raises(ValueError):
        generate_golden_dataset("   ")
