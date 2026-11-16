import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.diagnostics.golden_dataset import (
    initialize_golden_db,
    save_golden_dataset,
    list_saved_golden_datasets,
    get_saved_golden_dataset,
    delete_saved_golden_dataset,
)

@pytest.fixture(autouse=True)
def setup_golden_db(tmp_path, monkeypatch):
    test_db = tmp_path / "test_golden.db"
    monkeypatch.setattr("app.diagnostics.golden_dataset.GOLDEN_DB_PATH", test_db)
    initialize_golden_db(test_db)

def test_golden_dataset_persistence():
    cases = [
        {
            "test_id": "case-1",
            "query": "What is Istio?",
            "answer": "Istio is an open-source service mesh.",
            "ground_truth_context": "Service meshes like Istio decouple operations.",
            "relevant_document_ids": ["doc-1"],
            "retrieved_documents": [{"id": "doc-1", "text": "Service meshes like Istio decouple operations."}],
        }
    ]

    saved = save_golden_dataset("Kubernetes Benchmark", cases, description="Cloud tests")
    assert saved["name"] == "Kubernetes Benchmark"
    assert saved["case_count"] == 1
    ds_id = saved["dataset_id"]

    all_ds = list_saved_golden_datasets()
    assert len(all_ds) == 1
    assert all_ds[0]["dataset_id"] == ds_id

    retrieved = get_saved_golden_dataset(ds_id)
    assert retrieved is not None
    assert len(retrieved["cases"]) == 1

    deleted = delete_saved_golden_dataset(ds_id)
    assert deleted is True

def test_golden_dataset_api():
    client = TestClient(app)

    # 1. Save dataset
    payload = {
        "name": "API Benchmark",
        "description": "Via API",
        "cases": [
            {
                "test_id": "test-1",
                "query": "What is Python?",
                "answer": "A programming language.",
                "relevant_document_ids": ["doc-1"],
                "retrieved_documents": [{"id": "doc-1", "text": "Python is a language."}],
            }
        ],
    }
    r = client.post("/api/v1/rag/datasets/save", json=payload)
    assert r.status_code == 201
    ds_id = r.json()["dataset_id"]

    # 2. List
    r_list = client.get("/api/v1/rag/datasets")
    assert r_list.status_code == 200
    assert r_list.json()["count"] == 1

    # 3. Get
    r_get = client.get(f"/api/v1/rag/datasets/{ds_id}")
    assert r_get.status_code == 200
    assert r_get.json()["name"] == "API Benchmark"

    # 4. Delete
    r_del = client.delete(f"/api/v1/rag/datasets/{ds_id}")
    assert r_del.status_code == 200
    assert r_del.json()["deleted"] is True
