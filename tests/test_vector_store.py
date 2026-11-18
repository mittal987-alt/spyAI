"""
Unit tests for Persistent Dense Vector Store & Neural Reranker.
"""
from pathlib import Path
import pytest

from app.diagnostics.vector_store import (
    _cosine_similarity,
    _simple_dense_embedding,
    add_documents,
    clear_vector_store,
    get_vector_store_status,
    initialize_vector_store_db,
    search_vector_store,
)


@pytest.fixture
def temp_vector_db(tmp_path: Path):
    db_file = tmp_path / "test_vector.db"
    initialize_vector_store_db(db_file)
    return db_file


def test_embedding_normalization_and_similarity():
    v1 = _simple_dense_embedding("Machine learning model drift")
    v2 = _simple_dense_embedding("Model drift in machine learning")
    v3 = _simple_dense_embedding("Cooking recipe for chocolate chip cookies")

    assert len(v1) == 64
    sim_close = _cosine_similarity(v1, v2)
    sim_far = _cosine_similarity(v1, v3)

    assert sim_close > sim_far
    assert sim_close > 0.60


def test_add_and_search_vector_documents(temp_vector_db: Path):
    docs = [
        "Population Stability Index (PSI) measures shifts in feature distributions.",
        "Precision, Recall, and F1-score evaluate classification model accuracy.",
        "Deep learning transformers use self-attention mechanisms for sequences.",
    ]
    metas = [{"tag": "drift"}, {"tag": "metrics"}, {"tag": "architecture"}]

    ids = add_documents(texts=docs, metadatas=metas, db_path=temp_vector_db)
    assert len(ids) == 3

    status = get_vector_store_status(db_path=temp_vector_db)
    assert status["document_count"] == 3

    # Query matching first doc
    results = search_vector_store(
        query="How does PSI detect feature drift?",
        top_k=2,
        rerank=True,
        db_path=temp_vector_db,
    )
    assert len(results) >= 1
    assert "PSI" in results[0]["text"] or "feature" in results[0]["text"]
    assert "rerank_score" in results[0]


def test_clear_vector_store(temp_vector_db: Path):
    add_documents(texts=["Doc A", "Doc B"], db_path=temp_vector_db)
    assert get_vector_store_status(db_path=temp_vector_db)["document_count"] == 2

    cleared = clear_vector_store(db_path=temp_vector_db)
    assert cleared is True
    assert get_vector_store_status(db_path=temp_vector_db)["document_count"] == 0
