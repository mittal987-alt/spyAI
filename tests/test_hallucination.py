"""Tests for hallucination benchmarking engine and API endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.diagnostics.hallucination import (
    evaluate_hallucination,
    batch_evaluate_hallucination,
    _tokens, _ngrams, _rouge_n, _rouge_l,
    _novelty_rate, _token_precision_recall,
    _claim_support_rate, _hallucination_risk_score,
    _lcs_length,
)
from app.main import app

client = TestClient(app)

# ── Fixtures ──────────────────────────────────────────────────────────────────

REFERENCE = (
    "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars "
    "in Paris, France. It was constructed between 1887 and 1889 as the centerpiece "
    "of the 1889 World's Fair. The tower stands 330 metres tall."
)

FAITHFUL = (
    "The Eiffel Tower, a wrought-iron lattice tower in Paris, France, was built "
    "between 1887 and 1889 for the 1889 World's Fair. It is 330 metres tall."
)

HALLUCINATED = (
    "The Eiffel Tower was built in 1920 by Leonardo da Vinci in Rome, Italy. "
    "It is made of solid gold and stands 10 metres tall."
)


# ---------------------------------------------------------------------------
# Unit: _tokens
# ---------------------------------------------------------------------------

class TestTokens:
    def test_lowercases(self):
        assert "hello" in _tokens("Hello World")

    def test_strips_punctuation(self):
        toks = _tokens("Hello, world!")
        assert "world" in toks
        assert "," not in toks

    def test_empty_string(self):
        assert _tokens("") == []


# ---------------------------------------------------------------------------
# Unit: _ngrams
# ---------------------------------------------------------------------------

class TestNgrams:
    def test_unigrams(self):
        ng = _ngrams(["a", "b", "c"], 1)
        assert ng[("a",)] == 1

    def test_bigrams(self):
        ng = _ngrams(["a", "b", "c"], 2)
        assert ng[("a", "b")] == 1
        assert ng[("b", "c")] == 1

    def test_empty(self):
        assert _ngrams([], 2) == {}


# ---------------------------------------------------------------------------
# Unit: _lcs_length
# ---------------------------------------------------------------------------

class TestLcsLength:
    def test_identical(self):
        assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_disjoint(self):
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_partial(self):
        assert _lcs_length(["a", "b", "c"], ["a", "c"]) == 2

    def test_empty_a(self):
        assert _lcs_length([], ["a"]) == 0

    def test_empty_b(self):
        assert _lcs_length(["a"], []) == 0


# ---------------------------------------------------------------------------
# Unit: _rouge_n
# ---------------------------------------------------------------------------

class TestRougeN:
    def test_identical_text_r1(self):
        toks = ["the", "cat", "sat"]
        r = _rouge_n(toks, toks, 1)
        assert r["f1"] == pytest.approx(1.0)

    def test_disjoint_r1(self):
        r = _rouge_n(["a", "b"], ["c", "d"], 1)
        assert r["f1"] == 0.0

    def test_partial_overlap(self):
        r = _rouge_n(["a", "b", "c"], ["a", "b", "d"], 1)
        assert 0 < r["f1"] < 1

    def test_empty_ref_returns_zeros(self):
        r = _rouge_n(["a", "b"], [], 1)
        assert r["f1"] == 0.0

    def test_f1_between_precision_and_recall(self):
        r = _rouge_n(["a", "b", "c", "d"], ["a", "b"], 1)
        assert min(r["precision"], r["recall"]) <= r["f1"] <= max(r["precision"], r["recall"])


# ---------------------------------------------------------------------------
# Unit: _rouge_l
# ---------------------------------------------------------------------------

class TestRougeL:
    def test_identical(self):
        toks = ["a", "b", "c"]
        r = _rouge_l(toks, toks)
        assert r["f1"] == pytest.approx(1.0)

    def test_disjoint(self):
        r = _rouge_l(["x", "y"], ["a", "b"])
        assert r["f1"] == 0.0

    def test_lcs_length_key(self):
        r = _rouge_l(["a", "b", "c"], ["a", "b"])
        assert r["lcs_length"] == 2


# ---------------------------------------------------------------------------
# Unit: _novelty_rate
# ---------------------------------------------------------------------------

class TestNoveltyRate:
    def test_identical_zero_novelty(self):
        toks = _tokens("the cat sat on the mat")
        assert _novelty_rate(toks, toks, 2) == 0.0

    def test_disjoint_max_novelty(self):
        hyp = _tokens("alpha beta gamma")
        ref = _tokens("delta epsilon zeta theta")
        assert _novelty_rate(hyp, ref, 2) == pytest.approx(1.0)

    def test_rate_in_range(self):
        hyp = _tokens("the cat sat on the warm sunny mat")
        ref = _tokens("the cat sat on the mat")
        nov = _novelty_rate(hyp, ref, 2)
        assert 0.0 <= nov <= 1.0


# ---------------------------------------------------------------------------
# Unit: _token_precision_recall
# ---------------------------------------------------------------------------

class TestTokenPrecisionRecall:
    def test_identical(self):
        toks = ["a", "b", "c"]
        r = _token_precision_recall(toks, toks)
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)
        assert r["f1"] == pytest.approx(1.0)

    def test_empty_hyp(self):
        r = _token_precision_recall([], ["a", "b"])
        assert r["precision"] == 0.0

    def test_f1_between_p_and_r(self):
        r = _token_precision_recall(["a", "b", "c", "d"], ["a", "b"])
        assert min(r["precision"], r["recall"]) <= r["f1"] <= max(r["precision"], r["recall"])


# ---------------------------------------------------------------------------
# Unit: _claim_support_rate
# ---------------------------------------------------------------------------

class TestClaimSupportRate:
    def test_identical_text_fully_supported(self):
        text = "The tower was built in Paris in 1887."
        result = _claim_support_rate(text, text)
        assert result["support_rate"] == pytest.approx(1.0)

    def test_hallucinated_text_lower_support(self):
        faithful_result = _claim_support_rate(FAITHFUL, REFERENCE)
        hallucinated_result = _claim_support_rate(HALLUCINATED, REFERENCE)
        assert faithful_result["support_rate"] >= hallucinated_result["support_rate"]

    def test_returns_claim_details_list(self):
        result = _claim_support_rate(FAITHFUL, REFERENCE)
        assert isinstance(result["claim_details"], list)

    def test_total_claims_nonnegative(self):
        result = _claim_support_rate(FAITHFUL, REFERENCE)
        assert result["total_claims"] >= 0


# ---------------------------------------------------------------------------
# Unit: _hallucination_risk_score
# ---------------------------------------------------------------------------

class TestHallucinationRiskScore:
    def test_perfect_faithfulness_low_risk(self):
        result = _hallucination_risk_score(1.0, 1.0, 1.0, 0.0, 1.0)
        assert result["risk_score"] <= 20
        assert result["risk_level"] == "Low"

    def test_zero_faithfulness_high_risk(self):
        result = _hallucination_risk_score(0.0, 0.0, 0.0, 1.0, 0.0)
        assert result["risk_score"] >= 60

    def test_risk_score_in_range(self):
        result = _hallucination_risk_score(0.5, 0.3, 0.4, 0.3, 0.6)
        assert 0 <= result["risk_score"] <= 100

    def test_faithfulness_score_in_range(self):
        result = _hallucination_risk_score(0.7, 0.5, 0.6, 0.2, 0.8)
        assert 0 <= result["faithfulness_score"] <= 100

    def test_verdict_is_string(self):
        result = _hallucination_risk_score(0.5, 0.4, 0.5, 0.3, 0.5)
        assert isinstance(result["verdict"], str)


# ---------------------------------------------------------------------------
# Integration: evaluate_hallucination
# ---------------------------------------------------------------------------

class TestEvaluateHallucination:
    def test_returns_all_sections(self):
        result = evaluate_hallucination(FAITHFUL, REFERENCE)
        for key in ("rouge_scores", "token_metrics", "novelty", "claim_analysis", "hallucination_risk"):
            assert key in result

    def test_faithful_lower_risk_than_hallucinated(self):
        faithful_result = evaluate_hallucination(FAITHFUL, REFERENCE)
        hallucinated_result = evaluate_hallucination(HALLUCINATED, REFERENCE)
        assert (
            faithful_result["hallucination_risk"]["risk_score"]
            < hallucinated_result["hallucination_risk"]["risk_score"]
        )

    def test_identical_text_near_zero_risk(self):
        result = evaluate_hallucination(REFERENCE, REFERENCE)
        assert result["hallucination_risk"]["risk_score"] <= 25

    def test_rouge_scores_in_range(self):
        result = evaluate_hallucination(FAITHFUL, REFERENCE)
        for key in ("rouge1", "rouge2", "rougeL"):
            for sub in ("precision", "recall", "f1"):
                val = result["rouge_scores"][key][sub]
                assert 0.0 <= val <= 1.0, f"{key}.{sub} = {val}"

    def test_novelty_rate_in_range(self):
        result = evaluate_hallucination(FAITHFUL, REFERENCE)
        assert 0.0 <= result["novelty"]["bigram_novelty_rate"] <= 1.0

    def test_empty_generated_raises(self):
        with pytest.raises(ValueError, match="generated"):
            evaluate_hallucination("", REFERENCE)

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError, match="reference"):
            evaluate_hallucination(FAITHFUL, "")

    def test_hypothesis_length_correct(self):
        result = evaluate_hallucination(FAITHFUL, REFERENCE)
        assert result["token_metrics"]["hypothesis_length"] == len(_tokens(FAITHFUL))

    def test_claim_analysis_fields_present(self):
        result = evaluate_hallucination(FAITHFUL, REFERENCE)
        ca = result["claim_analysis"]
        for key in ("total_claims", "supported_claims", "unsupported_claims", "support_rate"):
            assert key in ca


# ---------------------------------------------------------------------------
# Integration: batch_evaluate_hallucination
# ---------------------------------------------------------------------------

class TestBatchEvaluateHallucination:
    def _pairs(self, n: int = 3) -> list[dict]:
        return [{"generated": FAITHFUL, "reference": REFERENCE}] * n

    def test_returns_results_and_stats(self):
        result = batch_evaluate_hallucination(self._pairs())
        assert "results" in result
        assert "aggregate_stats" in result

    def test_correct_pair_count(self):
        result = batch_evaluate_hallucination(self._pairs(3))
        assert result["aggregate_stats"]["total_pairs"] == 3
        assert result["aggregate_stats"]["evaluated"] == 3

    def test_error_handled_gracefully(self):
        pairs = [{"generated": "", "reference": REFERENCE}, {"generated": FAITHFUL, "reference": REFERENCE}]
        result = batch_evaluate_hallucination(pairs)
        assert result["aggregate_stats"]["errors"] == 1
        assert result["aggregate_stats"]["evaluated"] == 1

    def test_avg_risk_in_range(self):
        result = batch_evaluate_hallucination(self._pairs(2))
        assert 0 <= result["aggregate_stats"]["avg_hallucination_risk"] <= 100

    def test_empty_pairs_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            batch_evaluate_hallucination([])

    def test_too_many_pairs_raises(self):
        with pytest.raises(ValueError, match="50"):
            batch_evaluate_hallucination([{"generated": "x", "reference": "y"}] * 51)

    def test_high_risk_count_nonnegative(self):
        result = batch_evaluate_hallucination(self._pairs())
        assert result["aggregate_stats"]["high_risk_count"] >= 0


# ---------------------------------------------------------------------------
# API: POST /api/v1/hallucination/evaluate
# ---------------------------------------------------------------------------

class TestHallucinationEvaluateEndpoint:
    _payload = {"generated": FAITHFUL, "reference": REFERENCE}

    def test_200(self):
        res = client.post("/api/v1/hallucination/evaluate", json=self._payload)
        assert res.status_code == 200

    def test_response_has_risk_section(self):
        data = client.post("/api/v1/hallucination/evaluate", json=self._payload).json()
        assert "hallucination_risk" in data

    def test_response_has_rouge_scores(self):
        data = client.post("/api/v1/hallucination/evaluate", json=self._payload).json()
        assert "rouge_scores" in data

    def test_empty_generated_returns_422(self):
        res = client.post("/api/v1/hallucination/evaluate", json={"generated": "", "reference": REFERENCE})
        assert res.status_code == 422

    def test_missing_fields_returns_422(self):
        res = client.post("/api/v1/hallucination/evaluate", json={})
        assert res.status_code == 422

    def test_hallucinated_higher_risk_than_faithful(self):
        r_faithful = client.post("/api/v1/hallucination/evaluate", json={"generated": FAITHFUL, "reference": REFERENCE}).json()
        r_hallucinated = client.post("/api/v1/hallucination/evaluate", json={"generated": HALLUCINATED, "reference": REFERENCE}).json()
        assert (
            r_faithful["hallucination_risk"]["risk_score"]
            < r_hallucinated["hallucination_risk"]["risk_score"]
        )

    def test_risk_level_valid(self):
        data = client.post("/api/v1/hallucination/evaluate", json=self._payload).json()
        assert data["hallucination_risk"]["risk_level"] in {"Low", "Moderate", "High", "Critical"}


# ---------------------------------------------------------------------------
# API: POST /api/v1/hallucination/batch
# ---------------------------------------------------------------------------

class TestHallucinationBatchEndpoint:
    _payload = {
        "pairs": [
            {"generated": FAITHFUL, "reference": REFERENCE},
            {"generated": HALLUCINATED, "reference": REFERENCE},
        ]
    }

    def test_200(self):
        res = client.post("/api/v1/hallucination/batch", json=self._payload)
        assert res.status_code == 200

    def test_response_has_results_and_stats(self):
        data = client.post("/api/v1/hallucination/batch", json=self._payload).json()
        assert "results" in data
        assert "aggregate_stats" in data

    def test_correct_total_pairs(self):
        data = client.post("/api/v1/hallucination/batch", json=self._payload).json()
        assert data["aggregate_stats"]["total_pairs"] == 2

    def test_avg_risk_in_range(self):
        data = client.post("/api/v1/hallucination/batch", json=self._payload).json()
        assert 0 <= data["aggregate_stats"]["avg_hallucination_risk"] <= 100

    def test_empty_pairs_returns_422(self):
        res = client.post("/api/v1/hallucination/batch", json={"pairs": []})
        assert res.status_code == 422

    def test_too_many_pairs_returns_422(self):
        pairs = [{"generated": "x", "reference": "y"}] * 51
        res = client.post("/api/v1/hallucination/batch", json={"pairs": pairs})
        assert res.status_code == 422
