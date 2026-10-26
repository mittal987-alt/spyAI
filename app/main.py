
import asyncio
import io
import logging
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd

from app.events import publish, subscribe, unsubscribe

from app.experiments.run import run_experiment
from app.monitoring.alerts import get_alert, list_alerts
from app.deep_learning.service import run_deep_learning_diagnosis
from app.deep_learning.comparison import compare_models
from app.diagnostics.semantic_service import run_semantic_monitoring
from app.diagnostics.rag_service import run_rag_evaluation
from app.diagnostics.rag_history import (
    save_rag_evaluation,
    get_rag_evaluation,
    list_rag_evaluations,
    compare_rag_evaluations,
)
from app.diagnostics.rag_dashboard import get_rag_dashboard
from app.diagnostics.drift import detect_feature_drift
from app.diagnostics.drift_simulator import (
    get_drift_simulation_presets,
    simulate_drift_and_perturbations,
)
from app.diagnostics.guardrails import scan_prompt_guardrails
from app.diagnostics.golden_dataset import DOMAIN_PRESETS, generate_golden_dataset
from app.diagnostics.model_card import (
    get_compliance_frameworks_info,
    generate_model_card,
    format_model_card_markdown,
)
from app.monitoring.webhooks import (
    create_webhook,
    delete_webhook,
    list_webhook_logs,
    list_webhooks,
    test_webhook_connection,
    toggle_webhook,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ModelAutopsy AI",
    description=(
        "ML diagnostics, drift monitoring, deep-learning analysis, "
        "semantic drift detection, and RAG evaluation."
    ),
    version="0.6.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Request models
# -------------------------------------------------------------------

class DeepLearningRequest(BaseModel):
    epochs: int = Field(default=20, ge=2, le=100)
    batch_size: int = Field(default=64, ge=1, le=512)
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    n_samples: int = Field(default=3000, ge=100, le=20000)
    n_features: int = Field(default=20, ge=4, le=200)
    hidden_units: int = Field(default=64, ge=2, le=512)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    random_state: int = Field(default=42, ge=0)
    explain: bool = True


class ModelComparisonRequest(BaseModel):
    epochs: int = Field(default=15, ge=2, le=100)
    n_samples: int = Field(default=1500, ge=100, le=20000)
    n_features: int = Field(default=10, ge=4, le=200)
    random_state: int = Field(default=42, ge=0)


class SemanticMonitoringRequest(BaseModel):
    reference_texts: list[str] = Field(
        min_length=5,
        max_length=500,
    )
    current_texts: list[str] = Field(
        min_length=5,
        max_length=500,
    )
    n_clusters: int = Field(default=5, ge=2, le=20)
    n_permutations: int = Field(default=199, ge=19, le=999)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    random_state: int = Field(default=42, ge=0)


class RAGEvaluationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=10000)
    retrieved_documents: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=100,
    )
    relevant_document_ids: list[str] = Field(
        min_length=1,
        max_length=100,
    )
    k: int = Field(default=5, ge=1, le=100)
    relevance_threshold: float = Field(default=0.45, ge=0, le=1)
    support_threshold: float = Field(default=0.50, ge=0, le=1)


class RAGComparisonRequest(BaseModel):
    baseline_run_id: str = Field(min_length=1, max_length=100)
    candidate_run_id: str = Field(min_length=1, max_length=100)
    regression_threshold: float = Field(default=0.10, gt=0, le=1)


class GuardrailScanRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    check_injection: bool = Field(default=True)
    check_pii: bool = Field(default=True)
    check_anomaly: bool = Field(default=True)
    redact_pii: bool = Field(default=True)


class CreateWebhookRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    url: str = Field(min_length=8, max_length=1000)
    service_type: str = Field(default="generic")
    min_severity: str = Field(default="medium")


class TestWebhookRequest(BaseModel):
    url: str = Field(min_length=8, max_length=1000)
    service_type: str = Field(default="generic")
    name: str = Field(default="Test Webhook")


class ToggleWebhookRequest(BaseModel):
    is_enabled: bool


class GenerateGoldenDatasetRequest(BaseModel):
    document_text: str = Field(min_length=10, max_length=100000)
    num_pairs: int = Field(default=4, ge=1, le=15)
    include_hallucination_cases: bool = Field(default=True)
    k_retrieved: int = Field(default=5, ge=1, le=10)


class SimulateDriftRequest(BaseModel):
    n_samples: int = Field(default=1500, ge=200, le=20000)
    n_features: int = Field(default=10, ge=4, le=100)
    test_size: float = Field(default=0.3, ge=0.1, le=0.5)
    random_state: int = Field(default=42, ge=0)
    mean_shift: float = Field(default=0.0, ge=-10.0, le=10.0)
    variance_scale: float = Field(default=1.0, gt=0.0, le=10.0)
    gaussian_noise: float = Field(default=0.0, ge=0.0, le=10.0)
    outlier_rate: float = Field(default=0.0, ge=0.0, le=0.40)
    outlier_scale: float = Field(default=5.0, ge=1.0, le=20.0)
    label_flip_rate: float = Field(default=0.0, ge=0.0, le=0.50)
    target_features: list[int] | None = Field(default=None)


class ModelCardRequest(BaseModel):
    model_name: str = Field(default="ModelAutopsy Diagnostic Pipeline", max_length=200)
    model_version: str = Field(default="1.0.0", max_length=50)
    model_type: str = Field(default="Hybrid Deep Learning & RAG Evaluation System", max_length=300)
    intended_use: str = Field(default="Production AI model diagnostics, retrieval evaluation, covariate drift tracking, and adversarial security auditing.", max_length=2000)
    out_of_scope_use: str = Field(default="Autonomous safety-critical decisions without human verification or medical diagnostic prescription.", max_length=2000)
    developer_or_team: str = Field(default="AI Reliability & MLOps Engineering Team", max_length=200)
    custom_notes: str = Field(default="", max_length=5000)
    rag_metrics: dict | None = Field(default=None)
    drift_metrics: dict | None = Field(default=None)
    security_metrics: dict | None = Field(default=None)


# -------------------------------------------------------------------
# General endpoints
# -------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "ModelAutopsy AI",
        "version": app.version,
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ModelAutopsy AI",
        "version": app.version,
    }


# -------------------------------------------------------------------
# Experiment and drift endpoints
# -------------------------------------------------------------------

@app.post("/api/v1/experiments/drift")
def run_drift_experiment():
    try:
        result = run_experiment()
        publish("drift_completed", {"alert_count": len(result.get("alerts", []))})
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Drift experiment failed")
        raise HTTPException(
            status_code=500,
            detail="Drift experiment failed",
        ) from exc


@app.get("/api/v1/drift/presets")
def get_simulation_presets():
    return {"presets": get_drift_simulation_presets()}


@app.post("/api/v1/drift/simulate")
def simulate_drift(request: SimulateDriftRequest):
    try:
        report = simulate_drift_and_perturbations(
            n_samples=request.n_samples,
            n_features=request.n_features,
            test_size=request.test_size,
            random_state=request.random_state,
            mean_shift=request.mean_shift,
            variance_scale=request.variance_scale,
            gaussian_noise=request.gaussian_noise,
            outlier_rate=request.outlier_rate,
            outlier_scale=request.outlier_scale,
            label_flip_rate=request.label_flip_rate,
            target_features=request.target_features,
        )
        return report
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Drift simulation failed")
        raise HTTPException(
            status_code=500, detail="Drift simulation failed"
        ) from exc


# -------------------------------------------------------------------
# Alert endpoints
# -------------------------------------------------------------------

@app.get("/api/v1/alerts/summary")
def get_alert_summary():
    try:
        alerts = list_alerts(limit=500)

        by_severity = {
            "high": 0,
            "medium": 0,
            "low": 0,
            "none": 0,
        }

        for alert in alerts:
            severity = str(alert.get("severity", "none")).lower()
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            "total": len(alerts),
            "high": by_severity["high"],
            "medium": by_severity["medium"],
            "low": by_severity["low"],
            "by_severity": by_severity,
        }

    except Exception as exc:
        logger.exception("Could not retrieve alert summary")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve alert summary",
        ) from exc


@app.get("/api/v1/alerts")
def get_alerts(
    limit: int = Query(default=50, ge=1, le=500),
    severity: str | None = Query(default=None),
):
    try:
        alerts = list_alerts(
            limit=limit,
            severity=severity,
        )
        return {
            "alerts": alerts,
            "count": len(alerts),
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Could not retrieve alerts")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve alerts",
        ) from exc


@app.get("/api/v1/alerts/{alert_id}")
def get_alert_by_id(alert_id: str):
    try:
        alert = get_alert(alert_id)

        if alert is None:
            raise HTTPException(
                status_code=404,
                detail="Alert not found",
            )

        return alert

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Could not retrieve alert")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve alert",
        ) from exc


# -------------------------------------------------------------------
# Webhook notification endpoints
# -------------------------------------------------------------------

@app.get("/api/v1/webhooks")
def get_webhooks():
    try:
        return {"webhooks": list_webhooks()}
    except Exception as exc:
        logger.exception("Could not list webhooks")
        raise HTTPException(status_code=500, detail="Could not list webhooks") from exc


@app.post("/api/v1/webhooks")
def add_webhook(request: CreateWebhookRequest):
    try:
        return create_webhook(
            name=request.name,
            url=request.url,
            service_type=request.service_type,
            min_severity=request.min_severity,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Could not create webhook")
        raise HTTPException(status_code=500, detail="Could not create webhook") from exc


@app.patch("/api/v1/webhooks/{webhook_id}")
def update_webhook(webhook_id: str, request: ToggleWebhookRequest):
    try:
        updated = toggle_webhook(webhook_id, request.is_enabled)
        if not updated:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return {"webhook_id": webhook_id, "is_enabled": request.is_enabled}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not toggle webhook")
        raise HTTPException(status_code=500, detail="Could not toggle webhook") from exc


@app.delete("/api/v1/webhooks/{webhook_id}")
def remove_webhook(webhook_id: str):
    try:
        deleted = delete_webhook(webhook_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return {"deleted": True, "webhook_id": webhook_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not delete webhook")
        raise HTTPException(status_code=500, detail="Could not delete webhook") from exc


@app.post("/api/v1/webhooks/test")
def test_webhook(request: TestWebhookRequest):
    try:
        return test_webhook_connection(
            url=request.url,
            service_type=request.service_type,
            name=request.name,
        )
    except Exception as exc:
        logger.exception("Could not test webhook")
        raise HTTPException(status_code=500, detail="Could not test webhook") from exc


@app.get("/api/v1/webhooks/logs")
def get_webhook_logs(limit: int = Query(default=30, ge=1, le=200)):
    try:
        return {"logs": list_webhook_logs(limit=limit)}
    except Exception as exc:
        logger.exception("Could not retrieve webhook logs")
        raise HTTPException(status_code=500, detail="Could not retrieve webhook logs") from exc


# -------------------------------------------------------------------
# Deep-learning diagnostics
# -------------------------------------------------------------------

@app.post("/api/v1/deep-learning/diagnose")
def diagnose_deep_learning(request: DeepLearningRequest):
    try:
        return run_deep_learning_diagnosis(
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            n_samples=request.n_samples,
            n_features=request.n_features,
            hidden_units=request.hidden_units,
            dropout=request.dropout,
            random_state=request.random_state,
            explain=request.explain,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Deep-learning diagnosis failed")
        raise HTTPException(
            status_code=500,
            detail="Deep-learning diagnosis failed",
        ) from exc


@app.post("/api/v1/deep-learning/compare")
def compare_deep_learning_models(
    request: ModelComparisonRequest,
):
    try:
        return compare_models(
            n_samples=request.n_samples,
            n_features=request.n_features,
            epochs=request.epochs,
            random_state=request.random_state,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Model comparison failed")
        raise HTTPException(
            status_code=500,
            detail="Model comparison failed",
        ) from exc


# -------------------------------------------------------------------
# Semantic monitoring
# -------------------------------------------------------------------

@app.post("/api/v1/semantic/monitor")
def monitor_semantic_drift(
    request: SemanticMonitoringRequest,
):
    try:
        return run_semantic_monitoring(
            reference_texts=request.reference_texts,
            current_texts=request.current_texts,
            n_clusters=request.n_clusters,
            n_permutations=request.n_permutations,
            alpha=request.alpha,
            random_state=request.random_state,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Semantic monitoring failed")
        raise HTTPException(
            status_code=500,
            detail="Semantic monitoring failed",
        ) from exc


# -------------------------------------------------------------------
# Semantic embedding projector
# -------------------------------------------------------------------

@app.post("/api/v1/semantic/project")
def project_semantic_embeddings(request: SemanticMonitoringRequest):
    """
    Return 2-D PCA projections + KMeans cluster IDs for reference and
    current text collections. Designed to power the frontend scatter plot.
    """
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from app.diagnostics.embeddings import embed_texts

        ref_texts = request.reference_texts
        cur_texts = request.current_texts

        if len(ref_texts) < 5 or len(cur_texts) < 5:
            raise ValueError(
                "At least 5 documents are required in each collection"
            )

        all_texts = ref_texts + cur_texts
        embeddings = embed_texts(all_texts)

        pca = PCA(n_components=2, random_state=request.random_state)
        coords = pca.fit_transform(embeddings)

        n_clusters = max(2, min(request.n_clusters, len(all_texts)))
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=request.random_state,
            n_init="auto",
        )
        labels = kmeans.fit_predict(embeddings)

        ref_count = len(ref_texts)
        points = []
        for i, (x, y) in enumerate(coords):
            points.append({
                "x": float(x),
                "y": float(y),
                "cluster": int(labels[i]),
                "source": "reference" if i < ref_count else "current",
                "text": all_texts[i][:120],
            })

        variance = pca.explained_variance_ratio_.tolist()
        return {
            "points": points,
            "n_clusters": n_clusters,
            "pca_variance_explained": variance,
            "reference_count": ref_count,
            "current_count": len(cur_texts),
        }

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Semantic projection failed")
        raise HTTPException(
            status_code=500,
            detail="Semantic projection failed",
        ) from exc


# -------------------------------------------------------------------
# RAG golden benchmark generator
# -------------------------------------------------------------------

@app.get("/api/v1/rag/dataset-presets")
def get_golden_dataset_presets():
    return {"presets": DOMAIN_PRESETS}


@app.post("/api/v1/rag/generate-dataset")
def generate_dataset(request: GenerateGoldenDatasetRequest):
    try:
        cases = generate_golden_dataset(
            document_text=request.document_text,
            num_pairs=request.num_pairs,
            include_hallucination_cases=request.include_hallucination_cases,
            k_retrieved=request.k_retrieved,
        )
        return {"cases": cases, "total_generated": len(cases)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Golden dataset generation failed")
        raise HTTPException(
            status_code=500, detail="Golden dataset generation failed"
        ) from exc


# -------------------------------------------------------------------
# RAG evaluation
# -------------------------------------------------------------------

@app.post("/api/v1/rag/evaluate")
def evaluate_rag(request: RAGEvaluationRequest):
    try:
        report = run_rag_evaluation(
            query=request.query,
            answer=request.answer,
            retrieved_documents=request.retrieved_documents,
            relevant_document_ids=request.relevant_document_ids,
            k=request.k,
            relevance_threshold=request.relevance_threshold,
            support_threshold=request.support_threshold,
        )

        history = save_rag_evaluation(report)
        report["history"] = history

        publish("rag_evaluated", {
            "run_id": history.get("run_id", ""),
            "query_preview": request.query[:80],
        })

        return report

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("RAG evaluation failed")
        raise HTTPException(
            status_code=500,
            detail="RAG evaluation failed",
        ) from exc


# -------------------------------------------------------------------
# RAG evaluation history
# -------------------------------------------------------------------

@app.get("/api/v1/rag/history")
def get_rag_history(
    limit: int = Query(default=20, ge=1, le=100),
):
    try:
        evaluations = list_rag_evaluations(limit=limit)

        return {
            "count": len(evaluations),
            "evaluations": evaluations,
        }

    except Exception as exc:
        logger.exception("Could not retrieve RAG history")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve RAG history",
        ) from exc


@app.get("/api/v1/rag/history/{run_id}")
def get_rag_history_item(run_id: str):
    try:
        result = get_rag_evaluation(run_id)

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="RAG evaluation run not found",
            )

        return result

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Could not retrieve RAG evaluation")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve RAG evaluation",
        ) from exc


@app.post("/api/v1/rag/compare")
def compare_rag_runs(request: RAGComparisonRequest):
    try:
        return compare_rag_evaluations(
            baseline_run_id=request.baseline_run_id,
            candidate_run_id=request.candidate_run_id,
            regression_threshold=request.regression_threshold,
        )

    except ValueError as exc:
        message = str(exc)
        status_code = (
            404 if "was not found" in message else 422
        )

        raise HTTPException(
            status_code=status_code,
            detail=message,
        ) from exc

    except Exception as exc:
        logger.exception("RAG comparison failed")
        raise HTTPException(
            status_code=500,
            detail="RAG comparison failed",
        ) from exc


# -------------------------------------------------------------------
# RAG dashboard
# -------------------------------------------------------------------

@app.get("/api/v1/rag/dashboard")
def rag_dashboard(
    limit: int = Query(default=20, ge=1, le=100),
):
    try:
        return get_rag_dashboard(limit=limit)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Could not build RAG dashboard")
        raise HTTPException(
            status_code=500,
            detail="Could not build RAG dashboard",
        ) from exc


# -------------------------------------------------------------------
# CSV Dataset Upload — Drift Detection
# -------------------------------------------------------------------

_MAX_CSV_BYTES = 10 * 1024 * 1024  # 10 MB per file
_MAX_FEATURES = 200
_MAX_ROWS = 50_000


def _read_csv_upload(upload: UploadFile, label: str) -> pd.DataFrame:
    """Read an uploaded CSV file into a DataFrame with basic guards."""
    raw = upload.file.read()
    if len(raw) > _MAX_CSV_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{label} file exceeds the 10 MB limit.",
        )
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{label} could not be parsed as CSV: {exc}",
        ) from exc
    if df.empty:
        raise HTTPException(
            status_code=422,
            detail=f"{label} CSV is empty.",
        )
    if len(df) > _MAX_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"{label} exceeds {_MAX_ROWS} rows.",
        )
    return df


@app.post("/api/v1/drift/upload")
def upload_drift_csv(
    baseline: UploadFile = File(..., description="Baseline dataset CSV"),
    current: UploadFile = File(..., description="Current dataset CSV"),
    alpha: float = Query(default=0.05, gt=0, lt=1),
    effect_threshold: float = Query(default=0.10, ge=0, le=1),
):
    """
    Detect feature drift between two uploaded CSV datasets.

    Both files must share the same numeric column names.
    Returns per-feature KS statistics, adjusted p-values,
    drift flags, and severity tiers.
    """
    try:
        df_ref = _read_csv_upload(baseline, "baseline")
        df_cur = _read_csv_upload(current, "current")

        # Keep only numeric columns shared by both datasets
        ref_numeric = set(df_ref.select_dtypes(include="number").columns)
        cur_numeric = set(df_cur.select_dtypes(include="number").columns)
        shared = sorted(ref_numeric & cur_numeric)

        if not shared:
            raise HTTPException(
                status_code=422,
                detail="No numeric columns are shared between baseline and current.",
            )

        if len(shared) > _MAX_FEATURES:
            shared = shared[:_MAX_FEATURES]
            logger.warning(
                "CSV drift upload: truncated to %d features.",
                _MAX_FEATURES,
            )

        import numpy as np
        ref_arr = df_ref[shared].to_numpy(dtype=float)
        cur_arr = df_cur[shared].to_numpy(dtype=float)

        # Drop rows with NaN/inf
        ref_arr = ref_arr[np.isfinite(ref_arr).all(axis=1)]
        cur_arr = cur_arr[np.isfinite(cur_arr).all(axis=1)]

        if len(ref_arr) == 0 or len(cur_arr) == 0:
            raise HTTPException(
                status_code=422,
                detail="After removing non-finite rows, one or both datasets are empty.",
            )

        results = detect_feature_drift(
            reference=ref_arr,
            current=cur_arr,
            feature_names=shared,
            alpha=alpha,
            effect_threshold=effect_threshold,
        )

        drifted = [r for r in results if r["drift_detected"]]

        return {
            "baseline_rows": len(ref_arr),
            "current_rows": len(cur_arr),
            "features_tested": len(shared),
            "features_drifted": len(drifted),
            "alpha": alpha,
            "effect_threshold": effect_threshold,
            "results": results,
        }


    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("CSV drift upload failed")
        raise HTTPException(
            status_code=500,
            detail="CSV drift analysis failed.",
        ) from exc


# -------------------------------------------------------------------
# Diagnostic report generation
# -------------------------------------------------------------------

@app.get("/api/v1/reports/generate")
def generate_report(
    rag_limit: int = Query(default=20, ge=1, le=100),
    alert_limit: int = Query(default=50, ge=1, le=500),
):
    """
    Aggregate a full diagnostic report payload containing:
    - RAG evaluation dashboard metrics and recent run history
    - Drift alert summary and recent alert records
    - Service metadata and generation timestamp
    """
    from datetime import datetime, timezone

    try:
        dashboard = get_rag_dashboard(limit=rag_limit)
        recent_alerts = list_alerts(limit=alert_limit)

        severity_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "none": 0}
        for alert in recent_alerts:
            sev = str(alert.get("severity", "none")).lower()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "report_title": "ModelAutopsy AI — Diagnostic Audit Report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "service_version": app.version,
            "rag": {
                "dashboard": dashboard,
                "runs_included": dashboard.get("runs_included", 0),
                "finding_summary": dashboard.get("finding_summary", {}),
                "metrics": dashboard.get("metrics", {}),
                "recent_runs": dashboard.get("recent_runs", [])[:rag_limit],
            },
            "alerts": {
                "total": len(recent_alerts),
                "by_severity": severity_counts,
                "recent": recent_alerts[:20],
            },
        }

    except Exception as exc:
        logger.exception("Report generation failed")
        raise HTTPException(
            status_code=500,
            detail="Report generation failed.",
        ) from exc


# -------------------------------------------------------------------
# Prompt Guardrails & Security Scanner
# -------------------------------------------------------------------

@app.post("/api/v1/guardrails/scan")
def scan_guardrails(request: GuardrailScanRequest):
    try:
        report = scan_prompt_guardrails(
            text=request.text,
            check_injection=request.check_injection,
            check_pii=request.check_pii,
            check_anomaly=request.check_anomaly,
            redact_pii=request.redact_pii,
        )
        return report

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Guardrail scan failed")
        raise HTTPException(
            status_code=500,
            detail="Guardrail scan failed",
        ) from exc



# -------------------------------------------------------------------
# Server-Sent Events (SSE) stream
# -------------------------------------------------------------------

@app.get("/api/v1/events/stream")
async def events_stream():
    """
    Server-Sent Events endpoint pushing real-time diagnostic notifications.
    Sends keepalive pings every 20s if no events are published.
    """
    queue = await subscribe()

    async def event_generator():
        try:
            # Yield initial connection confirmation
            yield "data: {\"type\": \"connected\"}\n\n"
            while True:
                try:
                    # Wait for next event or timeout for keepalive
                    msg = await asyncio.wait_for(queue.get(), timeout=20.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            await unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



# -------------------------------------------------------------------
# Model Card & Regulatory Compliance
# -------------------------------------------------------------------

@app.get("/api/v1/compliance/frameworks")
def list_compliance_frameworks():
    return {"frameworks": get_compliance_frameworks_info()}


@app.post("/api/v1/compliance/model-card")
def create_model_card(request: ModelCardRequest):
    try:
        card = generate_model_card(
            model_name=request.model_name,
            model_version=request.model_version,
            model_type=request.model_type,
            intended_use=request.intended_use,
            out_of_scope_use=request.out_of_scope_use,
            developer_or_team=request.developer_or_team,
            custom_notes=request.custom_notes,
            rag_metrics=request.rag_metrics,
            drift_metrics=request.drift_metrics,
            security_metrics=request.security_metrics,
        )
        return card
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Model card generation failed")
        raise HTTPException(status_code=500, detail="Model card generation failed") from exc


@app.post("/api/v1/compliance/model-card/markdown")
def export_model_card_markdown(request: ModelCardRequest):
    try:
        card = generate_model_card(
            model_name=request.model_name,
            model_version=request.model_version,
            model_type=request.model_type,
            intended_use=request.intended_use,
            out_of_scope_use=request.out_of_scope_use,
            developer_or_team=request.developer_or_team,
            custom_notes=request.custom_notes,
            rag_metrics=request.rag_metrics,
            drift_metrics=request.drift_metrics,
            security_metrics=request.security_metrics,
        )
        md = format_model_card_markdown(card)
        rid = card["metadata"]["report_id"]
        return StreamingResponse(
            iter([md]),
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=model-card-" + rid + ".md"},
        )
    except Exception as exc:
        logger.exception("Model card markdown export failed")
        raise HTTPException(status_code=500, detail="Model card export failed") from exc
