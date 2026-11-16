"""Unit and integration tests for the Model Card & Regulatory Compliance generator.

These tests are written against the ACTUAL output schema of generate_model_card():

  card = {
    "metadata": {
      "report_id": str,
      "generated_at": str,
      "overall_compliance_score": float,
      "overall_compliance_status": str,
      "compliance_rating": str,
    },
    "model_overview": { name, version, type, developer, intended_use,
                         out_of_scope_use, custom_notes },
    "regulatory_compliance": {
      "eu_ai_act": { framework, assigned_risk_tier, score, status, article_audits },
      "nist_ai_rmf": { framework, score, core_functions },
      "iso_42001": { framework, score, controls },
    },
    "performance_and_robustness": { rag_evaluation, distribution_drift, security_guardrails },
    "mitigations_and_safeguards": list[str],
  }
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.diagnostics.model_card import (
    format_model_card_markdown,
    generate_model_card,
    get_compliance_frameworks_info,
)
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _card(**kwargs) -> dict:
    defaults = dict(
        model_name="TestModel",
        model_version="2.1.0",
        model_type="Binary Classifier",
        intended_use="Fraud detection",
        out_of_scope_use="Medical diagnosis",
        developer_or_team="ACME ML Team",
        custom_notes="",
        rag_metrics=None,
        drift_metrics=None,
        security_metrics=None,
    )
    defaults.update(kwargs)
    return generate_model_card(**defaults)


# ---------------------------------------------------------------------------
# get_compliance_frameworks_info
# ---------------------------------------------------------------------------

class TestGetComplianceFrameworksInfo:
    def test_returns_list(self):
        info = get_compliance_frameworks_info()
        assert isinstance(info, list)

    def test_at_least_three_frameworks(self):
        assert len(get_compliance_frameworks_info()) >= 3

    def test_each_entry_has_id_and_name(self):
        for fw in get_compliance_frameworks_info():
            assert "id" in fw
            assert "name" in fw

    def test_eu_ai_act_present(self):
        ids = {fw["id"] for fw in get_compliance_frameworks_info()}
        assert "eu_ai_act" in ids

    def test_nist_ai_rmf_present(self):
        ids = {fw["id"] for fw in get_compliance_frameworks_info()}
        assert "nist_ai_rmf" in ids

    def test_iso_42001_present(self):
        ids = {fw["id"] for fw in get_compliance_frameworks_info()}
        assert "iso_42001" in ids


# ---------------------------------------------------------------------------
# generate_model_card — metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_keys(self):
        meta = _card()["metadata"]
        for key in ("report_id", "generated_at", "overall_compliance_score",
                    "overall_compliance_status", "compliance_rating"):
            assert key in meta, f"Missing metadata key: {key}"

    def test_report_id_is_string(self):
        rid = _card()["metadata"]["report_id"]
        assert isinstance(rid, str) and rid.startswith("MC-")

    def test_overall_score_in_range(self):
        score = _card()["metadata"]["overall_compliance_score"]
        assert 0 <= score <= 100

    def test_compliance_status_valid(self):
        valid = {"FULLY_COMPLIANT", "CONDITIONAL_APPROVAL", "ACTION_REQUIRED"}
        assert _card()["metadata"]["overall_compliance_status"] in valid

    def test_compliance_rating_valid(self):
        assert _card()["metadata"]["compliance_rating"] in {"A", "B", "C"}


# ---------------------------------------------------------------------------
# generate_model_card — model_overview
# ---------------------------------------------------------------------------

class TestModelOverview:
    def test_fields_propagated(self):
        c = _card(
            model_name="Acme Classifier",
            model_version="3.0.1",
            model_type="Gradient Boosting",
            developer_or_team="Core ML",
            intended_use="Churn prediction",
            out_of_scope_use="Real-time surgery",
        )
        ov = c["model_overview"]
        assert ov["name"] == "Acme Classifier"
        assert ov["version"] == "3.0.1"
        assert ov["type"] == "Gradient Boosting"
        assert ov["developer"] == "Core ML"
        assert ov["intended_use"] == "Churn prediction"
        assert ov["out_of_scope_use"] == "Real-time surgery"

    def test_custom_notes_stored(self):
        c = _card(custom_notes="Demographic parity constraint applied.")
        assert "Demographic parity" in c["model_overview"]["custom_notes"]


# ---------------------------------------------------------------------------
# generate_model_card — regulatory_compliance
# ---------------------------------------------------------------------------

class TestRegulatoryCompliance:
    def test_top_level_keys_present(self):
        rc = _card()["regulatory_compliance"]
        assert "eu_ai_act" in rc
        assert "nist_ai_rmf" in rc
        assert "iso_42001" in rc

    def test_eu_ai_act_has_score(self):
        eu = _card()["regulatory_compliance"]["eu_ai_act"]
        assert 0 <= eu["score"] <= 100

    def test_eu_ai_act_has_article_audits(self):
        eu = _card()["regulatory_compliance"]["eu_ai_act"]
        assert isinstance(eu["article_audits"], list)
        assert len(eu["article_audits"]) > 0

    def test_eu_audit_items_have_status(self):
        for item in _card()["regulatory_compliance"]["eu_ai_act"]["article_audits"]:
            assert "status" in item
            assert "score" in item
            assert 0 <= item["score"] <= 100

    def test_nist_has_four_core_functions(self):
        funcs = _card()["regulatory_compliance"]["nist_ai_rmf"]["core_functions"]
        for fn in ("GOVERN", "MAP", "MEASURE", "MANAGE"):
            assert fn in funcs

    def test_nist_function_scores_in_range(self):
        funcs = _card()["regulatory_compliance"]["nist_ai_rmf"]["core_functions"]
        for fn in funcs.values():
            assert 0 <= fn["score"] <= 100

    def test_iso_42001_has_controls(self):
        controls = _card()["regulatory_compliance"]["iso_42001"]["controls"]
        assert isinstance(controls, list)
        assert len(controls) >= 4

    def test_iso_score_in_range(self):
        score = _card()["regulatory_compliance"]["iso_42001"]["score"]
        assert 0 <= score <= 100


# ---------------------------------------------------------------------------
# generate_model_card — performance_and_robustness
# ---------------------------------------------------------------------------

class TestPerformanceAndRobustness:
    def test_keys_present(self):
        perf = _card()["performance_and_robustness"]
        assert "rag_evaluation" in perf
        assert "distribution_drift" in perf
        assert "security_guardrails" in perf

    def test_custom_rag_metrics_passed_through(self):
        c = _card(rag_metrics={"faithfulness": 0.77})
        assert c["performance_and_robustness"]["rag_evaluation"]["faithfulness"] == pytest.approx(0.77)

    def test_custom_drift_metrics_passed_through(self):
        c = _card(drift_metrics={"max_psi_score": 0.55})
        assert c["performance_and_robustness"]["distribution_drift"]["max_psi_score"] == pytest.approx(0.55)

    def test_custom_security_metrics_passed_through(self):
        c = _card(security_metrics={"prompt_injection_guardrail": False})
        assert c["performance_and_robustness"]["security_guardrails"]["prompt_injection_guardrail"] is False

    def test_default_rag_metrics_populated(self):
        perf = _card()["performance_and_robustness"]["rag_evaluation"]
        assert "answer_relevance" in perf


# ---------------------------------------------------------------------------
# generate_model_card — mitigations_and_safeguards
# ---------------------------------------------------------------------------

class TestMitigations:
    def test_is_nonempty_list(self):
        mits = _card()["mitigations_and_safeguards"]
        assert isinstance(mits, list)
        assert len(mits) > 0

    def test_items_are_strings(self):
        for m in _card()["mitigations_and_safeguards"]:
            assert isinstance(m, str)


# ---------------------------------------------------------------------------
# format_model_card_markdown
# ---------------------------------------------------------------------------

class TestFormatModelCardMarkdown:
    def test_returns_string(self):
        assert isinstance(format_model_card_markdown(_card()), str)

    def test_starts_with_heading(self):
        md = format_model_card_markdown(_card())
        assert md.startswith("#")

    def test_contains_model_name(self):
        c = _card(model_name="Orion Classifier")
        md = format_model_card_markdown(c)
        assert "Orion Classifier" in md

    def test_contains_report_id(self):
        c = _card()
        md = format_model_card_markdown(c)
        assert c["metadata"]["report_id"] in md

    def test_contains_eu_ai_act_section(self):
        md = format_model_card_markdown(_card())
        assert "EU AI Act" in md

    def test_contains_nist_section(self):
        md = format_model_card_markdown(_card())
        assert "NIST" in md

    def test_contains_second_level_headings(self):
        md = format_model_card_markdown(_card())
        assert "##" in md

    def test_compliance_score_in_markdown(self):
        c = _card()
        md = format_model_card_markdown(c)
        assert str(c["metadata"]["overall_compliance_score"]) in md


# ---------------------------------------------------------------------------
# API: GET /api/v1/compliance/frameworks
# ---------------------------------------------------------------------------

class TestFrameworksEndpoint:
    def test_200(self):
        assert client.get("/api/v1/compliance/frameworks").status_code == 200

    def test_frameworks_key_present(self):
        data = client.get("/api/v1/compliance/frameworks").json()
        assert "frameworks" in data

    def test_frameworks_is_list(self):
        data = client.get("/api/v1/compliance/frameworks").json()
        assert isinstance(data["frameworks"], list)

    def test_three_or_more_frameworks(self):
        data = client.get("/api/v1/compliance/frameworks").json()
        assert len(data["frameworks"]) >= 3


# ---------------------------------------------------------------------------
# API: POST /api/v1/compliance/model-card
# ---------------------------------------------------------------------------

_PAYLOAD = {
    "model_name": "Sentinel v2",
    "model_version": "2.0.0",
    "model_type": "Neural Network",
    "intended_use": "Intrusion detection",
    "out_of_scope_use": "Medical triage",
    "developer_or_team": "Security AI Lab",
    "custom_notes": "",
}


class TestModelCardEndpoint:
    def test_200(self):
        assert client.post("/api/v1/compliance/model-card", json=_PAYLOAD).status_code == 200

    def test_metadata_in_response(self):
        data = client.post("/api/v1/compliance/model-card", json=_PAYLOAD).json()
        assert "metadata" in data

    def test_model_overview_in_response(self):
        data = client.post("/api/v1/compliance/model-card", json=_PAYLOAD).json()
        assert "model_overview" in data

    def test_regulatory_compliance_in_response(self):
        data = client.post("/api/v1/compliance/model-card", json=_PAYLOAD).json()
        assert "regulatory_compliance" in data

    def test_model_name_echoed(self):
        data = client.post("/api/v1/compliance/model-card", json=_PAYLOAD).json()
        assert data["model_overview"]["name"] == "Sentinel v2"

    def test_overall_score_numeric(self):
        data = client.post("/api/v1/compliance/model-card", json=_PAYLOAD).json()
        assert isinstance(data["metadata"]["overall_compliance_score"], (int, float))

    def test_eu_ai_act_in_regulatory(self):
        data = client.post("/api/v1/compliance/model-card", json=_PAYLOAD).json()
        assert "eu_ai_act" in data["regulatory_compliance"]

    def test_nist_in_regulatory(self):
        data = client.post("/api/v1/compliance/model-card", json=_PAYLOAD).json()
        assert "nist_ai_rmf" in data["regulatory_compliance"]

    def test_iso_in_regulatory(self):
        data = client.post("/api/v1/compliance/model-card", json=_PAYLOAD).json()
        assert "iso_42001" in data["regulatory_compliance"]

    def test_empty_payload_defaults_work(self):
        res = client.post("/api/v1/compliance/model-card", json={})
        assert res.status_code == 200

    def test_custom_rag_metrics_accepted(self):
        payload = {**_PAYLOAD, "rag_metrics": {"faithfulness": 0.91}}
        data = client.post("/api/v1/compliance/model-card", json=payload).json()
        assert data["performance_and_robustness"]["rag_evaluation"]["faithfulness"] == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# API: POST /api/v1/compliance/model-card/markdown
# ---------------------------------------------------------------------------

class TestModelCardMarkdownEndpoint:
    def test_200(self):
        assert client.post("/api/v1/compliance/model-card/markdown", json=_PAYLOAD).status_code == 200

    def test_content_type_markdown(self):
        res = client.post("/api/v1/compliance/model-card/markdown", json=_PAYLOAD)
        assert "text/markdown" in res.headers["content-type"]

    def test_content_disposition_attachment(self):
        res = client.post("/api/v1/compliance/model-card/markdown", json=_PAYLOAD)
        cd = res.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert ".md" in cd

    def test_body_is_non_empty_string(self):
        res = client.post("/api/v1/compliance/model-card/markdown", json=_PAYLOAD)
        assert len(res.text) > 100

    def test_body_contains_model_name(self):
        res = client.post("/api/v1/compliance/model-card/markdown", json=_PAYLOAD)
        assert "Sentinel v2" in res.text

    def test_body_contains_eu_ai_act(self):
        res = client.post("/api/v1/compliance/model-card/markdown", json=_PAYLOAD)
        assert "EU AI Act" in res.text

    def test_body_contains_nist(self):
        res = client.post("/api/v1/compliance/model-card/markdown", json=_PAYLOAD)
        assert "NIST" in res.text

    def test_empty_payload_succeeds(self):
        res = client.post("/api/v1/compliance/model-card/markdown", json={})
        assert res.status_code == 200
