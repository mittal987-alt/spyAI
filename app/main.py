
import asyncio
from contextlib import asynccontextmanager
import csv
import io
import logging
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import pandas as pd
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import (
    ALLOWED_HOSTS,
    CORS_ALLOWED_ORIGINS,
    IS_PRODUCTION,
    MODELAUTOPSY_API_KEY,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    validate_production_config,
)

from app.events import publish, subscribe, unsubscribe

from app.data.explorer import (
    DatasetExplorerError,
    _json_value,
    profile_dataset,
    read_dataset_csv,
)
from app.data.remediation import (
    apply_dataset_remediation,
    detect_dataset_outliers,
    export_dataframe_to_csv_bytes,
    generate_remediation_plan,
)
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
from app.diagnostics.hallucination import (
    evaluate_hallucination,
    batch_evaluate_hallucination,
)
from app.diagnostics.red_teaming import (
    generate_adversarial_probes,
    list_attack_archetypes,
    run_red_team_campaign,
)
from app.monitoring.webhooks import (
    create_webhook,
    delete_webhook,
    list_webhook_logs,
    list_webhooks,
    test_webhook_connection,
    toggle_webhook,
)
from app.monitoring.alert_rules import (
    create_rule,
    delete_rule,
    evaluate_rules,
    get_rule,
    get_supported_fields,
    list_rule_events,
    list_rules,
    toggle_rule,
    update_rule,
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    warnings = validate_production_config()
    for warning in warnings:
        logger.warning("Security configuration warning: %s", warning)
    yield

app = FastAPI(
    title="ModelAutopsy AI",
    description=(
        "ML diagnostics, drift monitoring, deep-learning analysis, "
        "semantic drift detection, and RAG evaluation."
    ),
    version="0.6.1",
    lifespan=lifespan,
)

import time
import threading
from collections import defaultdict, deque

class RateLimitMiddleware:
    """Sliding-window in-memory rate limiting middleware."""
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        self.app = app
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clients: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in ("/", "/health", "/docs", "/openapi.json"):
            await self.app(scope, receive, send)
            return

        client_ip = (scope.get("client") or ("unknown", 0))[0]
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            queue = self._clients[client_ip]
            while queue and queue[0] < window_start:
                queue.popleft()

            if len(queue) >= self.max_requests:
                retry_after = int(queue[0] + self.window_seconds - now) + 1
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "Too Many Requests", "retry_after": retry_after},
                    headers={"Retry-After": str(retry_after)},
                )
                await response(scope, receive, send)
                return

            queue.append(now)

        await self.app(scope, receive, send)

app.add_middleware(
    RateLimitMiddleware,
    max_requests=RATE_LIMIT_REQUESTS,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
)

# -------------------------------------------------------------------
# Security Middleware & Error Handling
# -------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global safe exception handler preventing internal traceback leaks."""
    logger.exception(
        "Unhandled exception processing %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )

# Host validation
if ALLOWED_HOSTS and "*" not in ALLOWED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=ALLOWED_HOSTS,
    )

# CORS configuration
cors_origins = [o for o in CORS_ALLOWED_ORIGINS if o]
if "*" in cors_origins and IS_PRODUCTION:
    cors_origins = [o for o in cors_origins if o != "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# Optional API Key Authentication dependency
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key_if_configured(key: str | None = Security(api_key_header)) -> None:
    """Enforce API key authentication on sensitive operations when configured."""
    if MODELAUTOPSY_API_KEY:
        if not key or key != MODELAUTOPSY_API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key (X-API-Key header required).",
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


class SaveGoldenDatasetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    cases: list[dict[str, Any]] = Field(min_length=1)


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
    model_config = {"protected_namespaces": ()}
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


class ScorecardEntryRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str = Field(min_length=1, max_length=200)
    model_version: str = Field(default="1.0.0", max_length=50)
    source: str = Field(default="manual")
    notes: str = Field(default="", max_length=1000)
    metrics: dict = Field(default_factory=dict)


class RegisterModelRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    name: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=50)
    stage: str = Field(default="development")
    framework: str = Field(default="sklearn")
    description: str = Field(default="", max_length=1000)
    artifact_uri: str = Field(default="", max_length=1000)
    metrics: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class UpdateStageRequest(BaseModel):
    stage: str


class CompareModelsRequest(BaseModel):
    baseline_id: str
    candidate_id: str


class ExplainDriftRequest(BaseModel):
    drift_summary: dict[str, Any] = Field(default_factory=dict)
    feature_drifts: list[dict[str, Any]] = Field(default_factory=list)
    api_key: str | None = None
    provider: str = Field(default="auto")
    model: str | None = None


class LLMGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)
    system_prompt: str = Field(default="", max_length=2000)
    provider: str = Field(default="auto")
    model: str | None = None
    api_key: str | None = None


class HardenPromptRequest(BaseModel):
    original_prompt: str = Field(min_length=1, max_length=10000)
    attack_vector: str = Field(default="prompt_injection")
    exploit_sample: str = Field(default="", max_length=5000)
    provider: str = Field(default="auto")
    model: str | None = None


class CheckRetrainingRequest(BaseModel):
    psi: float = Field(default=0.0)
    ks_p_value: float = Field(default=1.0)
    accuracy_drop_pct: float = Field(default=0.0)


class TriggerRetrainingRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)
    metrics: dict[str, Any] | None = None


class JudgePanelRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    answer: str = Field(min_length=1, max_length=10000)
    context: str = Field(default="", max_length=20000)
    provider: str = Field(default="auto")
    model: str | None = None


class VectorIndexRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    metadatas: list[dict[str, Any]] | None = None


class VectorSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = Field(default=True)


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
@app.post("/api/v1/datasets/profile")
async def profile_uploaded_dataset(
    file: UploadFile = File(...),
    preview_rows: int = Query(default=10, ge=1, le=100),
):
    """Upload a CSV and return a dataset profile."""

    try:
        # Read at most 10 MB plus one byte to enforce the size limit.
        content = await file.read(10 * 1024 * 1024 + 1)

        frame = read_dataset_csv(file.filename, content)

        return profile_dataset(
            frame,
            preview_rows=preview_rows,
        )

    except DatasetExplorerError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    finally:
        await file.close()


@app.post("/api/v1/datasets/report")
async def download_dataset_report(
    file: UploadFile = File(...),
):
    """Generate a downloadable JSON dataset report."""

    try:
        content = await file.read(10 * 1024 * 1024 + 1)

        frame = read_dataset_csv(file.filename, content)
        report = profile_dataset(frame, preview_rows=10)

        return JSONResponse(
            content=report,
            headers={
                "Content-Disposition": (
                    'attachment; filename="dataset-report.json"'
                )
            },
        )

    except DatasetExplorerError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    finally:
        await file.close()


@app.post("/api/v1/datasets/outliers")
async def get_dataset_outliers(
    file: UploadFile = File(...),
    z_threshold: float = Query(default=3.0, ge=1.0, le=10.0),
    iqr_multiplier: float = Query(default=1.5, ge=0.5, le=5.0),
):
    """Detect statistical and ensemble outliers in an uploaded dataset."""
    try:
        content = await file.read(10 * 1024 * 1024 + 1)
        frame = read_dataset_csv(file.filename, content)
        return detect_dataset_outliers(
            frame,
            z_threshold=z_threshold,
            iqr_multiplier=iqr_multiplier,
        )
    except DatasetExplorerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/v1/datasets/remediation-plan")
async def get_dataset_remediation_plan(
    file: UploadFile = File(...),
):
    """Diagnose data quality issues and return an automated remediation plan."""
    try:
        content = await file.read(10 * 1024 * 1024 + 1)
        frame = read_dataset_csv(file.filename, content)
        return generate_remediation_plan(frame)
    except DatasetExplorerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/v1/datasets/remediate")
async def remediate_dataset(
    file: UploadFile = File(...),
    drop_duplicates: bool = Form(default=True),
    drop_constant_columns: bool = Form(default=True),
    impute_numeric_strategy: str = Form(default="median"),
    impute_categorical_strategy: str = Form(default="mode"),
    clip_outliers: bool = Form(default=False),
    preview_rows: int = Form(default=10),
):
    """Apply cleaning, deduplication, and imputation transformations to dataset."""
    try:
        content = await file.read(10 * 1024 * 1024 + 1)
        frame = read_dataset_csv(file.filename, content)

        cleaned, report = apply_dataset_remediation(
            frame,
            drop_duplicates=drop_duplicates,
            drop_constant_columns=drop_constant_columns,
            impute_numeric_strategy=impute_numeric_strategy,
            impute_categorical_strategy=impute_categorical_strategy,
            clip_outliers=clip_outliers,
        )

        rows_to_preview = max(1, min(preview_rows, 50))
        preview = {
            "columns": [str(c) for c in cleaned.columns],
            "rows": [
                [_json_value(v) for v in row]
                for row in cleaned.head(rows_to_preview).itertuples(index=False, name=None)
            ],
        }

        return {
            "report": report,
            "preview": preview,
        }
    except DatasetExplorerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/v1/datasets/remediate/export")
async def export_remediated_csv(
    file: UploadFile = File(...),
    drop_duplicates: bool = Form(default=True),
    drop_constant_columns: bool = Form(default=True),
    impute_numeric_strategy: str = Form(default="median"),
    impute_categorical_strategy: str = Form(default="mode"),
    clip_outliers: bool = Form(default=False),
):
    """Apply remediation and stream cleaned CSV bytes as a downloadable attachment."""
    try:
        content = await file.read(10 * 1024 * 1024 + 1)
        frame = read_dataset_csv(file.filename, content)

        cleaned, _ = apply_dataset_remediation(
            frame,
            drop_duplicates=drop_duplicates,
            drop_constant_columns=drop_constant_columns,
            impute_numeric_strategy=impute_numeric_strategy,
            impute_categorical_strategy=impute_categorical_strategy,
            clip_outliers=clip_outliers,
        )

        csv_bytes = export_dataframe_to_csv_bytes(cleaned)
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="cleaned-dataset.csv"'},
        )
    except DatasetExplorerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()

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


@app.post("/api/v1/explain/drift")
def api_explain_drift(request: ExplainDriftRequest):
    """Generate plain-English executive analysis of drift telemetry (Ollama local, OpenAI, or heuristic)."""
    from app.diagnostics.explanations import explain_drift
    try:
        return explain_drift(
            drift_summary=request.drift_summary,
            feature_drifts=request.feature_drifts,
            api_key=request.api_key,
            model_name=request.model,
            provider=request.provider,
        )
    except Exception as exc:
        logger.exception("Failed to generate drift explanation")
        raise HTTPException(status_code=500, detail="Failed to generate drift explanation.") from exc


@app.get("/api/v1/llm/providers")
def api_get_llm_providers():
    """List available LLM inference backends (Ollama local, OpenAI cloud, heuristic)."""
    from app.models.llm_provider import get_available_providers
    return get_available_providers()


@app.post("/api/v1/llm/generate")
def api_llm_generate(request: LLMGenerateRequest):
    """Generate text via selected LLM backend (Ollama, OpenAI, or heuristic)."""
    from app.models.llm_provider import unified_generate
    try:
        return unified_generate(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            provider=request.provider,
            model=request.model,
            api_key=request.api_key,
        )
    except Exception as exc:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/llm/traces")
def api_get_llm_traces(limit: int = Query(default=50, ge=1, le=500)):
    """List recent LLM traces with latency, token counts, and cost telemetry."""
    from app.monitoring.llm_tracer import list_traces
    return {"traces": list_traces(limit=limit)}


@app.get("/api/v1/llm/traces/stats")
def api_get_llm_trace_stats():
    """Aggregate statistics across all recorded LLM calls."""
    from app.monitoring.llm_tracer import get_trace_stats
    return get_trace_stats()


@app.get("/api/v1/llm/traces/{trace_id}")
def api_get_llm_trace(trace_id: str):
    """Retrieve details for a single LLM trace including span waterfall."""
    from app.monitoring.llm_tracer import get_trace
    trace = get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace


# -------------------------------------------------------------------
# Autonomous Self-Healing & Remediation Agents
# -------------------------------------------------------------------

@app.post("/api/v1/agents/harden-prompt")
def api_harden_prompt(req: HardenPromptRequest):
    """Autonomous Prompt Hardener Agent: fortifies prompts against injections, jailbreaks, and leaks."""
    from app.agents.prompt_hardener import harden_prompt
    try:
        return harden_prompt(
            original_prompt=req.original_prompt,
            attack_vector=req.attack_vector,
            exploit_sample=req.exploit_sample,
            provider=req.provider,
            model=req.model,
        )
    except Exception as exc:
        logger.exception("Prompt hardening agent failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/agents/check-retraining")
def api_check_retraining(req: CheckRetrainingRequest):
    """Evaluates telemetry to decide if automated retraining should be triggered."""
    from app.agents.retraining_trigger import evaluate_retraining_need
    metrics = {
        "psi": req.psi,
        "ks_p_value": req.ks_p_value,
        "accuracy_drop_pct": req.accuracy_drop_pct,
    }
    return evaluate_retraining_need(metrics=metrics)


@app.post("/api/v1/agents/trigger-retraining")
def api_trigger_retraining(req: TriggerRetrainingRequest):
    """Autonomous Retraining Trigger: queues a self-healing retraining run."""
    from app.agents.retraining_trigger import trigger_autonomous_retraining
    return trigger_autonomous_retraining(reason=req.reason, metrics=req.metrics)


@app.get("/api/v1/agents/retraining-history")
def api_get_retraining_history(limit: int = Query(default=50, ge=1, le=500)):
    """Retrieve historical autonomous retraining runs."""
    from app.agents.retraining_trigger import get_retraining_history
    return {"runs": get_retraining_history(limit=limit)}


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
    offset: int = Query(default=0, ge=0),
):
    try:
        alerts = list_alerts(
            limit=limit,
            severity=severity,
            offset=offset,
        )
        return {
            "alerts": alerts,
            "count": len(alerts),
            "limit": limit,
            "offset": offset,
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


@app.delete("/api/v1/alerts/cleanup", dependencies=[Depends(require_api_key_if_configured)])
def cleanup_alerts(retention_days: int = Query(default=30, ge=1, le=365)):
    """Delete alerts older than retention_days days."""
    from app.monitoring.alerts import cleanup_old_alerts
    deleted = cleanup_old_alerts(retention_days=retention_days)
    return {"deleted": deleted, "retention_days": retention_days}


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


@app.post("/api/v1/webhooks", dependencies=[Depends(require_api_key_if_configured)])
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


@app.patch("/api/v1/webhooks/{webhook_id}", dependencies=[Depends(require_api_key_if_configured)])
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


@app.delete("/api/v1/webhooks/{webhook_id}", dependencies=[Depends(require_api_key_if_configured)])
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


@app.post("/api/v1/webhooks/test", dependencies=[Depends(require_api_key_if_configured)])
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


@app.post("/api/v1/rag/datasets/save", status_code=201, dependencies=[Depends(require_api_key_if_configured)])
def api_save_golden_dataset(request: SaveGoldenDatasetRequest):
    """Save generated golden cases as a named persistent benchmark dataset."""
    from app.diagnostics.golden_dataset import save_golden_dataset
    try:
        return save_golden_dataset(
            name=request.name,
            cases=request.cases,
            description=request.description,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to save golden dataset")
        raise HTTPException(status_code=500, detail="Failed to save golden dataset.") from exc


@app.get("/api/v1/rag/datasets")
def api_list_golden_datasets(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List saved golden datasets."""
    from app.diagnostics.golden_dataset import list_saved_golden_datasets
    datasets = list_saved_golden_datasets(limit=limit, offset=offset)
    return {"datasets": datasets, "count": len(datasets), "limit": limit, "offset": offset}


@app.get("/api/v1/rag/datasets/{dataset_id}")
def api_get_golden_dataset(dataset_id: str):
    """Get full golden dataset with all test cases."""
    from app.diagnostics.golden_dataset import get_saved_golden_dataset
    ds = get_saved_golden_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


@app.delete("/api/v1/rag/datasets/{dataset_id}", dependencies=[Depends(require_api_key_if_configured)])
def api_delete_golden_dataset(dataset_id: str):
    """Delete a saved golden dataset."""
    from app.diagnostics.golden_dataset import delete_saved_golden_dataset
    deleted = delete_saved_golden_dataset(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"deleted": True, "dataset_id": dataset_id}


@app.post("/api/v1/rag/judge-panel")
def api_run_judge_panel(request: JudgePanelRequest):
    """LLM-as-a-Judge 3-Panel Consensus Jury (Factual Auditor, Security Officer, Relevance Judge)."""
    from app.diagnostics.llm_judge import run_judge_panel
    try:
        return run_judge_panel(
            query=request.query,
            answer=request.answer,
            context=request.context,
            provider=request.provider,
            model=request.model,
        )
    except Exception as exc:
        logger.exception("Judge panel execution failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# -------------------------------------------------------------------
# Dense Vector Store & Neural Reranker Endpoints
# -------------------------------------------------------------------

@app.get("/api/v1/rag/vectorstore/status")
def api_vector_store_status():
    """Retrieve vector store index count and backend status."""
    from app.diagnostics.vector_store import get_vector_store_status
    return get_vector_store_status()


@app.post("/api/v1/rag/vectorstore/index")
def api_vector_store_index(req: VectorIndexRequest):
    """Index and embed documents into the persistent vector store."""
    from app.diagnostics.vector_store import add_documents
    try:
        doc_ids = add_documents(texts=req.texts, metadatas=req.metadatas)
        return {"indexed_count": len(doc_ids), "doc_ids": doc_ids}
    except Exception as exc:
        logger.exception("Vector store indexing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/rag/vectorstore/search")
def api_vector_store_search(req: VectorSearchRequest):
    """Semantic vector search with optional neural reranking."""
    from app.diagnostics.vector_store import search_vector_store
    try:
        results = search_vector_store(query=req.query, top_k=req.top_k, rerank=req.rerank)
        return {"results": results, "total_found": len(results)}
    except Exception as exc:
        logger.exception("Vector store search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/rag/vectorstore/clear")
def api_vector_store_clear():
    """Clear all indexed documents from vector store."""
    from app.diagnostics.vector_store import clear_vector_store
    clear_vector_store()
    return {"cleared": True}


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

        # Auto-populate scorecard
        try:
            from app.monitoring.scorecard import add_scorecard_entry
            scorecard_metrics = {
                "precision_at_k": report.get("precision_at_k", 0),
                "recall_at_k": report.get("recall_at_k", 0),
                "ndcg_at_k": report.get("ndcg_at_k", 0),
                "answer_relevance_score": report.get("answer_relevance_score", 0),
                "context_support_coverage": report.get("context_support_coverage", 0),
            }
            add_scorecard_entry(
                model_name="RAG Pipeline",
                metrics=scorecard_metrics,
                source="rag",
            )
        except Exception:
            pass  # Non-fatal

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
# CSV Dataset Upload â€” Drift Detection
# -------------------------------------------------------------------

_MAX_CSV_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_ROWS = 50_000
_MAX_COLUMNS = 200
_MAX_FEATURES = 200


def _read_csv_upload(
    upload: UploadFile,
    label: str,
) -> pd.DataFrame:
    """Read a CSV upload with bounded input, raw header duplicate checks, and safe error messages."""

    raw = upload.file.read(_MAX_CSV_BYTES + 1)

    if not raw:
        raise HTTPException(
            status_code=422,
            detail=f"{label} CSV is empty.",
        )

    if len(raw) > _MAX_CSV_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{label} file exceeds the 10 MB limit.",
        )

    if b"\x00" in raw[:65536]:
        raise HTTPException(
            status_code=422,
            detail=f"{label} CSV contains invalid null bytes.",
        )

    # Validate raw headers before Pandas renames duplicate columns
    try:
        header_text = raw[:131072].decode("utf-8-sig", errors="replace")
        first_line = next((line for line in header_text.splitlines() if line.strip()), None)
        if first_line:
            reader = csv.reader(io.StringIO(first_line))
            raw_headers = next(reader, [])
            if any(not str(h).strip() for h in raw_headers):
                raise HTTPException(
                    status_code=422,
                    detail=f"{label} contains an empty column name.",
                )
            trimmed = [str(h).strip() for h in raw_headers]
            if len(trimmed) != len(set(trimmed)):
                raise HTTPException(
                    status_code=422,
                    detail=f"{label} contains duplicate column names.",
                )
    except HTTPException:
        raise
    except Exception:
        pass

    try:
        df = pd.read_csv(
            io.BytesIO(raw),
            nrows=_MAX_ROWS + 1,
        )
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        logger.info(
            "Invalid CSV upload received for %s: %s",
            label,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=422,
            detail=f"{label} could not be parsed as a valid CSV.",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=422,
            detail=f"{label} CSV must contain at least one data row.",
        )

    if len(df) > _MAX_ROWS:
        raise HTTPException(
            status_code=422,
            detail=f"{label} exceeds the {_MAX_ROWS:,}-row limit.",
        )

    if df.shape[1] > _MAX_COLUMNS:
        raise HTTPException(
            status_code=422,
            detail=f"{label} exceeds the {_MAX_COLUMNS}-column limit.",
        )

    if df.columns.duplicated().any():
        raise HTTPException(
            status_code=422,
            detail=f"{label} contains duplicate column names.",
        )

    if any(not str(column).strip() for column in df.columns):
        raise HTTPException(
            status_code=422,
            detail=f"{label} contains an empty column name.",
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
            "report_title": "ModelAutopsy AI â€” Diagnostic Audit Report",
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

# -------------------------------------------------------------------
# Hallucination Benchmarking
# -------------------------------------------------------------------

class HallucinationEvalRequest(BaseModel):
    generated: str = Field(min_length=1, max_length=50000)
    reference: str = Field(min_length=1, max_length=100000)


class HallucinationBatchRequest(BaseModel):
    pairs: list[HallucinationEvalRequest] = Field(min_length=1, max_length=50)


@app.post("/api/v1/hallucination/evaluate")
def hallucination_evaluate(request: HallucinationEvalRequest):
    """Evaluate a single LLM output for hallucination against a reference document."""
    try:
        return evaluate_hallucination(
            generated=request.generated,
            reference=request.reference,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Hallucination evaluation failed")
        raise HTTPException(status_code=500, detail="Hallucination evaluation failed") from exc


@app.post("/api/v1/hallucination/batch")
def hallucination_batch(request: HallucinationBatchRequest):
    """Evaluate multiple generated/reference pairs and return aggregated hallucination stats."""
    try:
        pairs = [{"generated": p.generated, "reference": p.reference} for p in request.pairs]
        return batch_evaluate_hallucination(pairs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch hallucination evaluation failed")
        raise HTTPException(status_code=500, detail="Batch evaluation failed") from exc


# -------------------------------------------------------------------
# Automated LLM Red-Teaming & Adversarial Attack Simulation
# -------------------------------------------------------------------

class RedTeamGenerateRequest(BaseModel):
    topic: str = Field(default="bypassing safety guardrails", max_length=500)
    archetype_ids: list[str] | None = None


class RedTeamSimulateRequest(BaseModel):
    topic: str = Field(default="bypassing safety guardrails", max_length=500)
    archetype_ids: list[str] | None = None
    simulated_posture: str = Field(default="balanced")  # "hardened" | "vulnerable" | "balanced"
    custom_responses: dict[str, str] | None = None


@app.get("/api/v1/red-teaming/archetypes")
def get_red_teaming_archetypes():
    """List all supported adversarial jailbreak and exploit archetypes."""
    return {"archetypes": list_attack_archetypes()}


@app.post("/api/v1/red-teaming/generate-probes")
def generate_red_teaming_probes(request: RedTeamGenerateRequest):
    """Generate tailored adversarial probes for a specific topic or safety target."""
    try:
        probes = generate_adversarial_probes(
            topic=request.topic,
            archetype_ids=request.archetype_ids,
        )
        return {"probes": probes}
    except Exception as exc:
        logger.exception("Probe generation failed")
        raise HTTPException(status_code=500, detail="Probe generation failed") from exc


@app.post("/api/v1/red-teaming/simulate")
def simulate_red_teaming_campaign(request: RedTeamSimulateRequest):
    """Execute an automated adversarial red-teaming campaign and compute vulnerability scorecard."""
    try:
        return run_red_team_campaign(
            topic=request.topic,
            archetype_ids=request.archetype_ids,
            simulated_posture=request.simulated_posture,
            custom_responses=request.custom_responses,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Red-teaming campaign execution failed")
        raise HTTPException(status_code=500, detail="Red-teaming campaign failed") from exc


# -------------------------------------------------------------------
# Smart Alert Rules Engine
# -------------------------------------------------------------------


class CreateAlertRuleRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    logic: str = Field(default="AND")
    severity: str = Field(default="medium")
    conditions: list[dict] = Field(min_length=1, max_length=10)


class UpdateAlertRuleRequest(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    description: str | None = Field(default=None, max_length=500)
    logic: str | None = None
    severity: str | None = None
    conditions: list[dict] | None = Field(default=None, max_length=10)
    is_enabled: bool | None = None


class ToggleAlertRuleRequest(BaseModel):
    is_enabled: bool


class EvaluateRulesRequest(BaseModel):
    event_type: str = Field(default="manual", max_length=100)
    payload: dict = Field(default_factory=dict)


@app.get("/api/v1/alert-rules/fields")
def get_alert_rule_fields():
    """Return supported metric fields, operators, and descriptions for building alert rules."""
    return get_supported_fields()


@app.get("/api/v1/alert-rules")
def get_alert_rules(enabled_only: bool = Query(default=False)):
    """List all stored alert rules."""
    try:
        return {"rules": list_rules(enabled_only=enabled_only)}
    except Exception as exc:
        logger.exception("Failed to list alert rules")
        raise HTTPException(status_code=500, detail="Failed to list alert rules.") from exc


@app.post("/api/v1/alert-rules", status_code=201)
def create_alert_rule(request: CreateAlertRuleRequest):
    """Create a new smart alert rule with AND/OR condition logic."""
    try:
        rule = create_rule(
            name=request.name,
            conditions=request.conditions,
            logic=request.logic,
            description=request.description,
            severity=request.severity,
        )
        return rule
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create alert rule")
        raise HTTPException(status_code=500, detail="Failed to create alert rule.") from exc


@app.get("/api/v1/alert-rules/{rule_id}")
def get_alert_rule(rule_id: str):
    """Return a single alert rule by ID."""
    rule = get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"Alert rule '{rule_id}' not found.")
    return rule


@app.patch("/api/v1/alert-rules/{rule_id}")
def update_alert_rule(rule_id: str, request: UpdateAlertRuleRequest):
    """Update an existing alert rule's name, conditions, logic, or enabled status."""
    try:
        return update_rule(
            rule_id,
            name=request.name,
            description=request.description,
            logic=request.logic,
            conditions=request.conditions,
            severity=request.severity,
            is_enabled=request.is_enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to update alert rule %s", rule_id)
        raise HTTPException(status_code=500, detail="Failed to update alert rule.") from exc


@app.patch("/api/v1/alert-rules/{rule_id}/toggle")
def toggle_alert_rule(rule_id: str, request: ToggleAlertRuleRequest):
    """Enable or disable an alert rule."""
    try:
        return toggle_rule(rule_id, is_enabled=request.is_enabled)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/api/v1/alert-rules/{rule_id}")
def delete_alert_rule(rule_id: str):
    """Delete an alert rule permanently."""
    deleted = delete_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Alert rule '{rule_id}' not found.")
    return {"deleted": True, "rule_id": rule_id}


@app.post("/api/v1/alert-rules/evaluate")
def evaluate_alert_rules(request: EvaluateRulesRequest):
    """Evaluate all enabled rules against an event payload. Returns matched rules."""
    try:
        matched = evaluate_rules(
            event_type=request.event_type,
            payload=request.payload,
        )
        return {
            "matched_count": len(matched),
            "matched_rules": matched,
        }
    except Exception as exc:
        logger.exception("Failed to evaluate alert rules")
        raise HTTPException(status_code=500, detail="Rule evaluation failed.") from exc


@app.get("/api/v1/alert-rules/events/history")
def get_rule_events(
    rule_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Return recent rule-fire history, optionally filtered by rule_id."""
    try:
        events = list_rule_events(rule_id=rule_id, limit=limit)
        return {"events": events, "total": len(events)}
    except Exception as exc:
        logger.exception("Failed to fetch rule events")
        raise HTTPException(status_code=500, detail="Failed to fetch rule events.") from exc


# -------------------------------------------------------------------
# Model Scorecard & Leaderboard
# -------------------------------------------------------------------
from app.monitoring.scorecard import (
    add_scorecard_entry,
    list_scorecard_entries,
    get_scorecard_entry,
    delete_scorecard_entry,
    build_leaderboard,
    get_score_metadata,
)

@app.post("/api/v1/scorecard/entries", status_code=201, dependencies=[Depends(require_api_key_if_configured)])
def create_scorecard_entry(request: ScorecardEntryRequest):
    """Register a new model scorecard entry with composite scoring."""
    try:
        return add_scorecard_entry(
            model_name=request.model_name,
            metrics=request.metrics,
            model_version=request.model_version,
            source=request.source,
            notes=request.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create scorecard entry")
        raise HTTPException(status_code=500, detail="Failed to create scorecard entry.") from exc


@app.get("/api/v1/scorecard/entries")
def get_scorecard_entries(
    model_name: str | None = Query(default=None),
    source: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """List scorecard entries, optionally filtered by model name or source."""
    try:
        return {"entries": list_scorecard_entries(model_name=model_name, source=source, limit=limit)}
    except Exception as exc:
        logger.exception("Failed to list scorecard entries")
        raise HTTPException(status_code=500, detail="Failed to list scorecard entries.") from exc


@app.get("/api/v1/scorecard/entries/{entry_id}")
def get_scorecard_entry_by_id(entry_id: str):
    """Get a single scorecard entry by ID."""
    entry = get_scorecard_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Scorecard entry '{entry_id}' not found.")
    return entry


@app.delete("/api/v1/scorecard/entries/{entry_id}", dependencies=[Depends(require_api_key_if_configured)])
def remove_scorecard_entry(entry_id: str):
    """Delete a scorecard entry."""
    deleted = delete_scorecard_entry(entry_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Scorecard entry '{entry_id}' not found.")
    return {"deleted": True, "entry_id": entry_id}


@app.get("/api/v1/scorecard/leaderboard")
def get_leaderboard():
    """Get the model quality leaderboard with composite scores, trends, and metric averages."""
    try:
        return build_leaderboard()
    except Exception as exc:
        logger.exception("Failed to build leaderboard")
        raise HTTPException(status_code=500, detail="Failed to build leaderboard.") from exc


@app.get("/api/v1/scorecard/metadata")
def get_scorecard_metadata():
    """Return scoring formula, dimension weights, and descriptions."""
    return get_score_metadata()


# -------------------------------------------------------------------
# Model Registry & Lifecycle Management
# -------------------------------------------------------------------
from app.models.registry import (
    register_model,
    list_registered_models,
    get_registered_model,
    update_model_stage,
    delete_registered_model,
    compare_registered_models,
)

@app.post("/api/v1/models/registry", status_code=201, dependencies=[Depends(require_api_key_if_configured)])
def api_register_model(request: RegisterModelRequest):
    """Register a new model version with metadata, parameters, and metrics."""
    try:
        return register_model(
            name=request.name,
            version=request.version,
            stage=request.stage,
            framework=request.framework,
            description=request.description,
            artifact_uri=request.artifact_uri,
            metrics=request.metrics,
            parameters=request.parameters,
            tags=request.tags,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to register model")
        raise HTTPException(status_code=500, detail="Failed to register model.") from exc


@app.get("/api/v1/models/registry")
def api_list_models(
    name: str | None = Query(default=None),
    stage: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List registered models, optionally filtered by model name or lifecycle stage."""
    try:
        models = list_registered_models(name=name, stage=stage, limit=limit, offset=offset)
        return {"models": models, "count": len(models), "limit": limit, "offset": offset}
    except Exception as exc:
        logger.exception("Failed to list models")
        raise HTTPException(status_code=500, detail="Failed to list models.") from exc


@app.get("/api/v1/models/registry/{model_id}")
def api_get_model(model_id: str):
    """Get details for a registered model by ID."""
    model = get_registered_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return model


@app.patch("/api/v1/models/registry/{model_id}/stage", dependencies=[Depends(require_api_key_if_configured)])
def api_update_stage(model_id: str, request: UpdateStageRequest):
    """Promote or transition a model's lifecycle stage (development, staging, production, archived)."""
    try:
        return update_model_stage(model_id, request.stage)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to update model stage")
        raise HTTPException(status_code=500, detail="Failed to update model stage.") from exc


@app.delete("/api/v1/models/registry/{model_id}", dependencies=[Depends(require_api_key_if_configured)])
def api_delete_model(model_id: str):
    """Delete a model from the registry."""
    deleted = delete_registered_model(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")
    return {"deleted": True, "model_id": model_id}


@app.post("/api/v1/models/registry/compare")
def api_compare_models(request: CompareModelsRequest):
    """Head-to-head comparison between two registered model versions."""
    try:
        return compare_registered_models(request.baseline_id, request.candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to compare models")
        raise HTTPException(status_code=500, detail="Failed to compare models.") from exc
