"""
Hallucination Benchmarking & Factual Consistency Detector
==========================================================
Evaluates LLM-generated answers against reference documents using:
  - ROUGE-1 / ROUGE-2 / ROUGE-L (token overlap)
  - Token-level precision & recall (macro)
  - N-gram novelty rate (fraction of generated n-grams absent from reference)
  - Longest Common Subsequence ratio
  - Claim-level entailment heuristic (keyword matching)
  - Aggregated hallucination risk score (0–100)
"""
from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return _normalise(text).split()


# ---------------------------------------------------------------------------
# ROUGE helpers
# ---------------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _rouge_n(hyp_tokens: list[str], ref_tokens: list[str], n: int) -> dict[str, float]:
    hyp_ng = _ngrams(hyp_tokens, n)
    ref_ng = _ngrams(ref_tokens, n)
    if not ref_ng:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = sum((hyp_ng & ref_ng).values())
    precision = overlap / max(sum(hyp_ng.values()), 1)
    recall = overlap / max(sum(ref_ng.values()), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute LCS length via dynamic programming (space-optimised)."""
    if not a or not b:
        return 0
    la, lb = len(a), len(b)
    prev = [0] * (lb + 1)
    for i in range(la):
        curr = [0] * (lb + 1)
        for j in range(lb):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(prev[j + 1], curr[j])
        prev = curr
    return prev[lb]


def _rouge_l(hyp_tokens: list[str], ref_tokens: list[str]) -> dict[str, float]:
    lcs = _lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / max(len(hyp_tokens), 1)
    recall = lcs / max(len(ref_tokens), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "lcs_length": lcs,
    }


# ---------------------------------------------------------------------------
# Novelty / hallucination rate
# ---------------------------------------------------------------------------

def _novelty_rate(hyp_tokens: list[str], ref_tokens: list[str], n: int = 2) -> float:
    """Fraction of hypothesis n-grams not present in the reference."""
    hyp_ng = _ngrams(hyp_tokens, n)
    ref_ng = _ngrams(ref_tokens, n)
    if not hyp_ng:
        return 0.0
    novel = sum(v for k, v in hyp_ng.items() if k not in ref_ng)
    return round(novel / sum(hyp_ng.values()), 4)


# ---------------------------------------------------------------------------
# Token-level precision / recall
# ---------------------------------------------------------------------------

def _token_precision_recall(hyp_tokens: list[str], ref_tokens: list[str]) -> dict[str, float]:
    hyp_set = set(hyp_tokens)
    ref_set = set(ref_tokens)
    if not hyp_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = len(hyp_set & ref_set) / len(hyp_set)
    recall = len(hyp_set & ref_set) / max(len(ref_set), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Claim extraction heuristic
# ---------------------------------------------------------------------------

def _extract_claims(text: str) -> list[str]:
    """Split text into simple claim-like sentences."""
    sentences = re.split(r"[.!?]\s+", _normalise(text))
    return [s.strip() for s in sentences if len(s.split()) >= 3]


def _claim_support_rate(hyp_text: str, ref_text: str) -> dict[str, Any]:
    """
    Heuristic: for each claim sentence in the hypothesis, check whether ≥50%
    of its content words appear in the reference (a bag-of-words entailment proxy).
    """
    claims = _extract_claims(hyp_text)
    ref_tokens = set(_tokens(ref_text))
    results = []
    for claim in claims:
        claim_tokens = [t for t in _tokens(claim) if len(t) > 3]  # ignore stop words
        if not claim_tokens:
            continue
        covered = sum(1 for t in claim_tokens if t in ref_tokens)
        support_ratio = covered / len(claim_tokens)
        results.append(
            {
                "claim": claim[:200],
                "supported": support_ratio >= 0.5,
                "support_ratio": round(support_ratio, 3),
                "covered_tokens": covered,
                "total_tokens": len(claim_tokens),
            }
        )
    supported = sum(1 for r in results if r["supported"])
    return {
        "total_claims": len(results),
        "supported_claims": supported,
        "unsupported_claims": len(results) - supported,
        "support_rate": round(supported / max(len(results), 1), 4),
        "claim_details": results,
    }


# ---------------------------------------------------------------------------
# Semantic Similarity Helper
# ---------------------------------------------------------------------------

def _semantic_similarity(generated: str, reference: str) -> float:
    """
    TF-IDF cosine similarity between generated and reference texts.
    Returns a value in [0, 1]. Falls back to 0.0 if embedding fails.
    """
    try:
        from app.diagnostics.embeddings import embed_texts
        import numpy as np
        vecs = embed_texts([generated, reference])
        gen_vec = vecs[0]
        ref_vec = vecs[1]
        norm_gen = np.linalg.norm(gen_vec)
        norm_ref = np.linalg.norm(ref_vec)
        if norm_gen == 0 or norm_ref == 0:
            return 0.0
        return float(round(float(np.dot(gen_vec, ref_vec) / (norm_gen * norm_ref)), 4))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Aggregated hallucination risk score
# ---------------------------------------------------------------------------

def _hallucination_risk_score(
    rouge1_f1: float,
    rouge2_f1: float,
    rougel_f1: float,
    novelty_rate: float,
    claim_support_rate: float,
    semantic_similarity: float = 0.0,
) -> dict[str, Any]:
    """
    Produces a 0–100 hallucination risk score.
    Higher score = higher hallucination risk.
    Components:
      - Low ROUGE overlap → high risk
      - High novelty rate → high risk
      - Low claim support → high risk
      - Low semantic similarity → high risk
    """
    faithfulness = (
        0.25 * rouge1_f1
        + 0.20 * rouge2_f1
        + 0.20 * rougel_f1
        + 0.15 * claim_support_rate
        + 0.20 * semantic_similarity
    )
    hallucination_from_faithfulness = 1.0 - faithfulness
    raw_risk = (0.60 * hallucination_from_faithfulness + 0.40 * novelty_rate)
    risk_score = round(min(max(raw_risk * 100, 0), 100), 1)

    if risk_score <= 20:
        level = "Low"
        verdict = "Highly faithful — output strongly grounded in the reference."
    elif risk_score <= 40:
        level = "Moderate"
        verdict = "Mostly faithful — minor unsupported claims detected."
    elif risk_score <= 65:
        level = "High"
        verdict = "Notable hallucination risk — significant portions ungrounded."
    else:
        level = "Critical"
        verdict = "Severe hallucination — output largely diverges from reference."

    return {
        "risk_score": risk_score,
        "risk_level": level,
        "verdict": verdict,
        "faithfulness_score": round(faithfulness * 100, 1),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_hallucination(
    generated: str,
    reference: str,
) -> dict[str, Any]:
    """
    Full hallucination evaluation pipeline.

    Args:
        generated: The LLM-generated answer / completion.
        reference: The ground-truth reference document or gold answer.

    Returns:
        Dict with rouge_scores, token_metrics, novelty, claim_analysis,
        hallucination_risk, and meta sections.
    """
    if not generated or not generated.strip():
        raise ValueError("generated text must be non-empty")
    if not reference or not reference.strip():
        raise ValueError("reference text must be non-empty")

    hyp = _tokens(generated)
    ref = _tokens(reference)

    r1 = _rouge_n(hyp, ref, 1)
    r2 = _rouge_n(hyp, ref, 2)
    rl = _rouge_l(hyp, ref)
    nov = _novelty_rate(hyp, ref, 2)
    tok = _token_precision_recall(hyp, ref)
    claims = _claim_support_rate(generated, reference)
    sem_sim = _semantic_similarity(generated, reference)
    risk = _hallucination_risk_score(
        rouge1_f1=r1["f1"],
        rouge2_f1=r2["f1"],
        rougel_f1=rl["f1"],
        novelty_rate=nov,
        claim_support_rate=claims["support_rate"],
        semantic_similarity=sem_sim,
    )

    return {
        "rouge_scores": {
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": rl,
        },
        "semantic_similarity": {
            "score": sem_sim,
            "interpretation": (
                "High semantic similarity — output closely mirrors reference meaning."
                if sem_sim >= 0.7
                else "Moderate semantic similarity."
                if sem_sim >= 0.4
                else "Low semantic similarity — output diverges from reference."
            ),
        },
        "token_metrics": {
            **tok,
            "hypothesis_length": len(hyp),
            "reference_length": len(ref),
            "length_ratio": round(len(hyp) / max(len(ref), 1), 3),
        },
        "novelty": {
            "bigram_novelty_rate": nov,
            "interpretation": (
                "Low novelty — output closely mirrors reference wording."
                if nov < 0.3
                else "Moderate novelty — some new content introduced."
                if nov < 0.6
                else "High novelty — substantial new content not in reference."
            ),
        },
        "claim_analysis": claims,
        "hallucination_risk": risk,
    }


def batch_evaluate_hallucination(
    pairs: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Evaluate multiple generated/reference pairs and return individual results
    plus aggregated statistics.

    Args:
        pairs: List of {"generated": str, "reference": str} dicts.

    Returns:
        Dict with results list and aggregate_stats.
    """
    if not pairs:
        raise ValueError("pairs list must be non-empty")
    if len(pairs) > 50:
        raise ValueError("Maximum 50 pairs per batch")

    results = []
    for i, pair in enumerate(pairs):
        try:
            result = evaluate_hallucination(
                generated=pair.get("generated", ""),
                reference=pair.get("reference", ""),
            )
            result["pair_index"] = i
            result["status"] = "ok"
        except ValueError as exc:
            result = {"pair_index": i, "status": "error", "error": str(exc)}
        results.append(result)

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        avg_risk = sum(r["hallucination_risk"]["risk_score"] for r in ok_results) / len(ok_results)
        avg_faithfulness = sum(r["hallucination_risk"]["faithfulness_score"] for r in ok_results) / len(ok_results)
        avg_r1 = sum(r["rouge_scores"]["rouge1"]["f1"] for r in ok_results) / len(ok_results)
        high_risk_count = sum(
            1 for r in ok_results if r["hallucination_risk"]["risk_score"] > 40
        )
    else:
        avg_risk = avg_faithfulness = avg_r1 = 0.0
        high_risk_count = 0

    return {
        "results": results,
        "aggregate_stats": {
            "total_pairs": len(pairs),
            "evaluated": len(ok_results),
            "errors": len(pairs) - len(ok_results),
            "avg_hallucination_risk": round(avg_risk, 1),
            "avg_faithfulness_score": round(avg_faithfulness, 1),
            "avg_rouge1_f1": round(avg_r1, 4),
            "high_risk_count": high_risk_count,
            "high_risk_rate": round(high_risk_count / max(len(ok_results), 1), 3),
        },
    }
