"""
Synthetic Golden Dataset & Benchmark Generator for RAG Pipelines.

Extracts semantic chunks from raw knowledge documents and generates realistic
(query, answer, context, retrieved_documents) evaluation benchmarks with ground-truth
attribution and adversarial hallucination test cases.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import numpy as np

from app.diagnostics.embeddings import embed_texts


DOMAIN_PRESETS: dict[str, dict[str, Any]] = {
    "cloud_infrastructure": {
        "title": "Cloud Native Microservices Architecture",
        "text": (
            "Service meshes like Istio decouple operational management from application business code. "
            "They inject an Envoy sidecar proxy into each pod to handle mutual TLS (mTLS) encryption, "
            "circuit breaking, and distributed tracing via OpenTelemetry headers. "
            "Kubernetes horizontal pod autoscalers (HPA) scale replicas based on CPU utilization "
            "or custom Prometheus metrics such as incoming requests per second. "
            "Persistent data is decoupled from container runtimes using Container Storage Interface (CSI) "
            "drivers backed by cloud block storage volumes. "
            "Distributed cache layers using Redis Cluster reduce database read pressure by storing "
            "precomputed query results with explicit TTL expiration policies. "
            "Database connection pooling using PgBouncer prevents PostgreSQL backend process exhaustion "
            "during sudden traffic spikes."
        ),
    },
    "mlops_governance": {
        "title": "Machine Learning Operations & Observability",
        "text": (
            "Feature stores like Feast provide point-in-time correct historical feature retrieval "
            "for training data to eliminate data leakage. "
            "Model registries track lineage, artifacts, hyperparameters, and production deployment stages. "
            "Statistical drift detection compares production inference feature distributions against "
            "training baselines using two-sample Kolmogorov-Smirnov tests and Population Stability Index (PSI). "
            "Model quantization reduces 32-bit floating point weights to 8-bit integers (INT8) "
            "to halve memory bandwidth and accelerate CPU and GPU inference throughput. "
            "Canary deployments route a small fraction (e.g. 5%) of live production queries to the "
            "candidate model while monitoring error rates and latency percentiles before full cutover."
        ),
    },
    "health_informatics": {
        "title": "Clinical Information Systems & HL7 FHIR",
        "text": (
            "Fast Healthcare Interoperability Resources (FHIR) is a RESTful standard that represents "
            "clinical concepts as standardized JSON or XML resources like Patient, Observation, and DiagnosticReport. "
            "SNOMED CT provides a comprehensive clinical healthcare terminology covering diseases, "
            "findings, and medical procedures with formal description logics. "
            "LOINC codes standardize laboratory observations, vital signs, and diagnostic measurements. "
            "De-identification algorithms scrub HIPAA Safe Harbor identifiers including patient names, "
            "precise geographic subdivisions, phone numbers, and dates prior to secondary medical research. "
            "Clinical decision support systems integrate with EHR platforms using the SMART on FHIR "
            "OAuth 2.0 authorization framework to launch secure clinician-facing applications."
        ),
    },
}


def _split_into_chunks(text: str, min_chars: int = 40) -> list[str]:
    """Split raw document into coherent semantic paragraphs or sentence blocks."""
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []

    for para in raw_paragraphs:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", para)
            if len(s.strip()) >= min_chars
        ]
        if len(sentences) > 1:
            chunks.extend(sentences)
        elif para:
            chunks.append(para)

    if not chunks:
        chunks = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.strip()) > 20
        ]

    return chunks if chunks else [text.strip()]


def _synthesize_query_and_answer(chunk: str) -> tuple[str, str, str]:
    """Extract a salient query and factual answer from a text chunk."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
    if not sentences:
        sentences = [chunk]

    primary_sentence = sentences[0]

    # Heuristic question synthesis based on grammatical keywords
    if "like" in primary_sentence or "such as" in primary_sentence:
        match = re.search(r"(\b[\w\s]{3,30}\b)\s+(?:like|such as)\s+([^,.]+)", primary_sentence, re.IGNORECASE)
        if match:
            query = f"What examples of {match.group(1).strip()} are mentioned and what do they do?"
        else:
            query = f"What role does {primary_sentence[:40]} play?"
    elif "reduce" in primary_sentence or "accelerate" in primary_sentence or "prevent" in primary_sentence:
        query = f"How does the system handle {primary_sentence[:35]}?"
    elif "standard" in primary_sentence or "provides" in primary_sentence:
        query = f"What standard or capability is described regarding {primary_sentence[:30]}?"
    elif "because" in primary_sentence or "to " in primary_sentence:
        query = f"Why is {primary_sentence[:35]} used?"
    else:
        words = primary_sentence.split()[:6]
        query = f"How is {' '.join(words)} implemented?"

    answer = " ".join(sentences[:2])
    return query, answer, primary_sentence


def generate_golden_dataset(
    document_text: str,
    num_pairs: int = 4,
    include_hallucination_cases: bool = True,
    k_retrieved: int = 5,
) -> list[dict[str, Any]]:
    """
    Generate synthetic (query, answer, context, retrieved_docs) evaluation pairs.
    """
    if not document_text or not document_text.strip():
        raise ValueError("Document text must not be empty.")

    chunks = _split_into_chunks(document_text)
    if len(chunks) == 0:
        raise ValueError("Could not extract semantic chunks from document.")

    # Limit to requested number of test cases
    target_count = max(1, min(num_pairs, len(chunks), 15))
    selected_chunk_indices = list(range(min(len(chunks), target_count)))

    # Compute embeddings for all chunks to rank realistic retrieved candidates
    try:
        chunk_embeddings = embed_texts(chunks)
    except Exception:
        chunk_embeddings = None

    golden_cases: list[dict[str, Any]] = []

    for idx in selected_chunk_indices:
        target_chunk = chunks[idx]
        query, answer, _ = _synthesize_query_and_answer(target_chunk)
        case_id = f"gold-{idx + 1}-{uuid.uuid4().hex[:6]}"
        target_doc_id = f"doc-{idx + 1}"

        # Determine test type (alternate with adversarial hallucination case if requested)
        is_hallucination = include_hallucination_cases and (idx % 2 == 1)
        test_type = "adversarial_hallucination" if is_hallucination else "factual_benchmark"

        if is_hallucination:
            # Inject an unsupported claim into the answer to test RAG hallucination detector
            tampered_answer = (
                f"{answer} Furthermore, benchmark testing verified a 99.8% quantum efficiency improvement "
                f"under proprietary zero-gravity validation constraints."
            )
            final_answer = tampered_answer
            notes = "Adversarial test: Injected unsupported quantum claim not grounded in context."
        else:
            final_answer = answer
            notes = "Clean factual benchmark grounded directly in retrieved knowledge."

        # Assemble retrieved documents: target doc + nearest distractors
        retrieved_docs: list[dict[str, Any]] = [
            {"id": target_doc_id, "text": target_chunk}
        ]

        # Pick other chunks as distractors
        other_indices = [i for i in range(len(chunks)) if i != idx]
        if chunk_embeddings is not None and other_indices:
            target_vec = chunk_embeddings[idx]
            sims = [
                (other_idx, float(np.dot(target_vec, chunk_embeddings[other_idx])))
                for other_idx in other_indices
            ]
            sims.sort(key=lambda s: s[1], reverse=True)
            distractor_indices = [s[0] for s in sims[: k_retrieved - 1]]
        else:
            distractor_indices = other_indices[: k_retrieved - 1]

        for d_idx in distractor_indices:
            retrieved_docs.append({
                "id": f"doc-{d_idx + 1}",
                "text": chunks[d_idx],
            })

        golden_cases.append({
            "test_id": case_id,
            "test_type": test_type,
            "query": query,
            "answer": final_answer,
            "ground_truth_context": target_chunk,
            "relevant_document_ids": [target_doc_id],
            "retrieved_documents": retrieved_docs,
            "k": len(retrieved_docs),
            "relevance_threshold": 0.45,
            "support_threshold": 0.50,
            "notes": notes,
        })

    return golden_cases
