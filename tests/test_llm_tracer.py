"""
Unit tests for Real-Time LLM Tracing & Token Cost Telemetry.
"""
from pathlib import Path
import pytest
from app.monitoring.llm_tracer import (
    LLMTrace,
    calculate_cost,
    get_trace,
    get_trace_stats,
    initialize_traces_db,
    list_traces,
    save_trace,
    trace_llm_call,
)
from app.models.llm_provider import unified_generate


@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "test_traces.db"
    initialize_traces_db(db_file)
    return db_file


def test_calculate_cost():
    # Local & heuristic should always be free ($0.0)
    assert calculate_cost("ollama", "llama3.2", 1000, 500) == 0.0
    assert calculate_cost("heuristic", "deterministic-heuristic-v1", 200, 50) == 0.0

    # OpenAI GPT-4o-mini
    # input: 0.000150 per 1k, output: 0.000600 per 1k
    cost = calculate_cost("openai", "gpt-4o-mini", 1000, 1000)
    assert cost == pytest.approx(0.00075, rel=1e-3)


def test_save_and_retrieve_trace(temp_db: Path):
    trace = LLMTrace(
        trace_id="test-trace-1",
        request_id="req-123",
        provider="ollama",
        model="llama3.2",
        prompt="Explain model drift",
        prompt_tokens=100,
        completion_tokens=250,
        total_tokens=350,
        latency_ms=45.2,
        estimated_cost_usd=0.0,
        spans=[{"name": "ollama_generate", "duration_ms": 42.0}],
        success=True,
    )
    save_trace(trace, db_path=temp_db)

    traces = list_traces(limit=10, db_path=temp_db)
    assert len(traces) == 1
    assert traces[0]["trace_id"] == "test-trace-1"
    assert traces[0]["provider"] == "ollama"
    assert traces[0]["total_tokens"] == 350
    assert traces[0]["success"] is True

    single = get_trace("test-trace-1", db_path=temp_db)
    assert single is not None
    assert single["model"] == "llama3.2"


def test_trace_stats_aggregation(temp_db: Path):
    trace1 = LLMTrace(
        trace_id="t1",
        provider="openai",
        model="gpt-4o-mini",
        prompt="prompt 1",
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        latency_ms=120.0,
        estimated_cost_usd=0.00045,
        success=True,
    )
    trace2 = LLMTrace(
        trace_id="t2",
        provider="ollama",
        model="llama3.2",
        prompt="prompt 2",
        prompt_tokens=500,
        completion_tokens=500,
        total_tokens=1000,
        latency_ms=80.0,
        estimated_cost_usd=0.0,
        success=True,
    )
    save_trace(trace1, db_path=temp_db)
    save_trace(trace2, db_path=temp_db)

    stats = get_trace_stats(db_path=temp_db)
    assert stats["total_calls"] == 2
    assert stats["total_tokens"] == 2500
    assert stats["total_cost_usd"] == pytest.approx(0.00045, rel=1e-3)
    assert stats["avg_latency_ms"] == pytest.approx(100.0, rel=1e-2)
    assert stats["success_rate"] == 100.0
    assert len(stats["by_provider"]) == 2


def test_trace_context_manager(temp_db: Path):
    with trace_llm_call(
        provider="heuristic",
        model="rule-v1",
        prompt="test prompt",
        persist=True,
    ) as ctx:
        ctx.start_span("retrieval")
        ctx.end_span("retrieval")
        ctx.finish(prompt_tokens=10, completion_tokens=20)


def test_unified_generate_traces_automatically():
    # Calling heuristic generation should write a trace
    res = unified_generate(prompt="What is PSI metric?", provider="heuristic")
    assert "Deterministic" in res["text"]

    traces = list_traces(limit=5)
    assert len(traces) >= 1
    assert any(t["provider"] == "heuristic" for t in traces)
