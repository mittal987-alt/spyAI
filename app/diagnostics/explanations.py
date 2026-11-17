"""
Executive Diagnostic & Drift Explanation Engine for ModelAutopsy AI.

Translates complex statistical metrics (KS-stat, PSI, Wasserstein, P-values, RAG NDCG)
into concise, human-readable executive summaries and remediation actions.
Works via OpenAI LLM if configured, or falls back to deterministic rule-based synthesis.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import LLM_API_KEY, LLM_MODEL_NAME

logger = logging.getLogger(__name__)


def generate_heuristic_explanation(
    drift_summary: dict[str, Any],
    feature_drifts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Generate deterministic, executive-level diagnostic insights without calling an external LLM."""
    total_features = drift_summary.get("total_features", len(feature_drifts))
    drifted_count = drift_summary.get("drifted_features_count", sum(1 for f in feature_drifts if f.get("drift_detected")))
    drift_pct = round((drifted_count / max(total_features, 1)) * 100, 1)

    # Classify overall health
    if drift_pct >= 40:
        health_status = "CRITICAL"
        headline = f"Severe distribution collapse detected: {drifted_count}/{total_features} features ({drift_pct}%) have drifted."
    elif drift_pct >= 15:
        health_status = "WARNING"
        headline = f"Moderate covariate shift detected: {drifted_count}/{total_features} features ({drift_pct}%) exhibit statistically significant drift."
    elif drift_pct > 0:
        health_status = "MONITOR"
        headline = f"Mild localized drift detected: {drifted_count}/{total_features} features affected."
    else:
        health_status = "HEALTHY"
        headline = "All monitored features remain statistically stable compared to baseline."

    # Identify top affected features
    sorted_features = sorted(
        feature_drifts,
        key=lambda f: (f.get("psi_score", 0.0), f.get("ks_statistic", 0.0)),
        reverse=True,
    )
    top_affected = sorted_features[:3]

    remediation_steps = []
    if health_status == "CRITICAL":
        remediation_steps.append("Trigger automated model retraining using the most recent 14-day production window.")
        remediation_steps.append("Deploy shadow-model inference to benchmark degradation against business KPIs.")
    elif health_status == "WARNING":
        remediation_steps.append("Inspect data ingestion upstream for recent schema or provider formatting changes.")
        remediation_steps.append("Recalibrate feature normalization statistics (mean/variance scaling).")
    elif health_status == "MONITOR":
        remediation_steps.append("Continue standard telemetry collection and re-evaluate at next scheduled cadence.")
    else:
        remediation_steps.append("No immediate engineering intervention required.")

    findings = []
    for feat in top_affected:
        name = feat.get("feature_name", feat.get("feature", "unknown"))
        ks = feat.get("ks_statistic", 0.0)
        psi = feat.get("psi_score", 0.0)
        findings.append(
            f"Feature '{name}': KS divergence={ks:.3f}, PSI score={psi:.3f}. Severity tier: {feat.get('severity', 'none').upper()}."
        )

    return {
        "engine": "deterministic_rule_synthesis",
        "health_status": health_status,
        "headline": headline,
        "summary": (
            f"ModelAutopsy examined {total_features} operational features against reference distributions. "
            f"{headline} Top divergence concentrated in: {', '.join(f.get('feature_name', f.get('feature', '')) for f in top_affected)}."
        ),
        "key_findings": findings,
        "recommended_actions": remediation_steps,
    }


def explain_drift(
    drift_summary: dict[str, Any],
    feature_drifts: list[dict[str, Any]],
    api_key: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Generate executive explanation using OpenAI LLM if available, otherwise heuristic engine."""
    effective_key = api_key or LLM_API_KEY
    effective_model = model_name or LLM_MODEL_NAME

    # If no key available, use deterministic heuristic synthesis
    if not effective_key:
        return generate_heuristic_explanation(drift_summary, feature_drifts)

    try:
        import openai
        client = openai.OpenAI(api_key=effective_key)

        prompt = (
            f"You are an expert MLOps AI reliability engineer. Analyze this data drift telemetry and output a concise "
            f"executive summary with health status, root-cause assessment, and actionable next steps.\n\n"
            f"Summary: {drift_summary}\n"
            f"Top Features: {feature_drifts[:5]}"
        )

        response = client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": "You are ModelAutopsy AI diagnostic advisor. Provide crisp, professional ML observability summaries."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.2,
        )

        llm_text = response.choices[0].message.content or ""
        return {
            "engine": f"llm_assisted ({effective_model})",
            "health_status": "ANALYZED",
            "headline": "LLM-synthesized operational assessment",
            "summary": llm_text,
            "recommended_actions": ["Review generated diagnostic narrative and schedule feature recalibration."],
        }

    except Exception as exc:
        logger.warning("LLM explanation request failed; falling back to heuristic engine: %s", exc)
        fallback = generate_heuristic_explanation(drift_summary, feature_drifts)
        fallback["engine"] = f"fallback (LLM error: {str(exc)[:60]})"
        return fallback
