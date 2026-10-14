
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
        return run_experiment()
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


# -------------------------------------------------------------------
# Alert endpoints
# -------------------------------------------------------------------

@app.get("/api/v1/alerts/summary")
def get_alert_summary():
    try:
        alerts = list_alerts(limit=1000)

        summary = {
            "total": len(alerts),
            "by_severity": {
                "high": 0,
                "medium": 0,
                "low": 0,
                "none": 0,
            },
        }

        for alert in alerts:
            severity = str(alert.get("severity", "none")).lower()
            summary["by_severity"][severity] = (
                summary["by_severity"].get(severity, 0) + 1
            )

        return summary

    except Exception as exc:
        logger.exception("Could not retrieve alert summary")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve alert summary",
        ) from exc


@app.get("/api/v1/alerts")
def get_alerts(
    limit: int = Query(default=50, ge=1, le=1000),
    severity: str | None = Query(default=None),
):
    try:
        return {
            "alerts": list_alerts(
                limit=limit,
                severity=severity,
            )
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
def get_alert_by_id(alert_id: int):
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
