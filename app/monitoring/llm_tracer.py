"""
LLM Tracing & Token Cost Telemetry
Tracks every LLM call: latency, tokens, estimated cost, span timeline.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

# ---------------------------------------------------------------------------
# Pricing table  (USD per 1 000 tokens)
# ---------------------------------------------------------------------------
PRICING: dict[str, dict[str, float]] = {
    # provider:model -> {input, output}  (cost per 1_000 tokens)
    "openai:gpt-4o": {"input": 0.005, "output": 0.015},
    "openai:gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
    "openai:gpt-4-turbo": {"input": 0.010, "output": 0.030},
    "openai:gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # local/open-source → always $0
    "ollama:*": {"input": 0.0, "output": 0.0},
    "heuristic:*": {"input": 0.0, "output": 0.0},
}


def calculate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a single LLM call."""
    key = f"{provider}:{model}"
    pricing = PRICING.get(key) or PRICING.get(f"{provider}:*")
    if pricing is None:
        # Unknown provider — assume $0 (safe default)
        return 0.0
    cost = (prompt_tokens / 1_000) * pricing["input"] + (completion_tokens / 1_000) * pricing["output"]
    return round(cost, 8)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Span:
    name: str
    start_ms: float
    end_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class LLMTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    provider: str = "unknown"
    model: str = "unknown"
    prompt: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    spans: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# In-process trace context
# ---------------------------------------------------------------------------
class TraceContext:
    """Mutable context accumulated during a single LLM call."""

    def __init__(self, provider: str, model: str, prompt: str, request_id: str = ""):
        self.trace = LLMTrace(
            provider=provider,
            model=model,
            prompt=prompt[:500],  # truncate for storage
            request_id=request_id or str(uuid.uuid4()),
        )
        self._start_wall = time.monotonic()
        self._active_spans: dict[str, float] = {}  # name -> start_ms

    def start_span(self, name: str, metadata: dict | None = None) -> None:
        self._active_spans[name] = time.monotonic() * 1000
        return None

    def end_span(self, name: str, metadata: dict | None = None) -> None:
        start = self._active_spans.pop(name, None)
        if start is None:
            return
        span = Span(
            name=name,
            start_ms=start,
            end_ms=time.monotonic() * 1000,
            metadata=metadata or {},
        )
        self.trace.spans.append(asdict(span))

    def finish(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        error: str = "",
    ) -> LLMTrace:
        self.trace.prompt_tokens = prompt_tokens
        self.trace.completion_tokens = completion_tokens
        self.trace.total_tokens = prompt_tokens + completion_tokens
        self.trace.latency_ms = round((time.monotonic() - self._start_wall) * 1000, 2)
        self.trace.estimated_cost_usd = calculate_cost(
            self.trace.provider, self.trace.model, prompt_tokens, completion_tokens
        )
        self.trace.success = success
        self.trace.error = error
        return self.trace


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------
DB_PATH = Path("artifacts") / "traces.db"


def initialize_traces_db(db_path: Path | None = None) -> None:
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_traces (
            trace_id      TEXT PRIMARY KEY,
            request_id    TEXT,
            provider      TEXT,
            model         TEXT,
            prompt        TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens  INTEGER,
            latency_ms    REAL,
            estimated_cost_usd REAL,
            spans         TEXT,
            created_at    TEXT,
            success       INTEGER,
            error         TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_trace(trace: LLMTrace, db_path: Path | None = None) -> None:
    path = db_path or DB_PATH
    initialize_traces_db(path)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        INSERT OR REPLACE INTO llm_traces VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            trace.trace_id,
            trace.request_id,
            trace.provider,
            trace.model,
            trace.prompt,
            trace.prompt_tokens,
            trace.completion_tokens,
            trace.total_tokens,
            trace.latency_ms,
            trace.estimated_cost_usd,
            json.dumps(trace.spans),
            trace.created_at,
            int(trace.success),
            trace.error,
        ),
    )
    conn.commit()
    conn.close()


def list_traces(limit: int = 50, db_path: Path | None = None) -> list[dict]:
    path = db_path or DB_PATH
    if not path.exists():
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM llm_traces ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["spans"] = json.loads(d.get("spans") or "[]")
        d["success"] = bool(d["success"])
        result.append(d)
    return result


def get_trace(trace_id: str, db_path: Path | None = None) -> dict | None:
    path = db_path or DB_PATH
    if not path.exists():
        return None
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM llm_traces WHERE trace_id=?", (trace_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    d = dict(row)
    d["spans"] = json.loads(d.get("spans") or "[]")
    d["success"] = bool(d["success"])
    return d


def get_trace_stats(db_path: Path | None = None) -> dict:
    """Aggregate statistics across all saved traces."""
    path = db_path or DB_PATH
    if not path.exists():
        return _empty_stats()
    conn = sqlite3.connect(path)
    row = conn.execute("""
        SELECT
            COUNT(*)                     AS total_calls,
            SUM(total_tokens)            AS total_tokens,
            SUM(estimated_cost_usd)      AS total_cost_usd,
            AVG(latency_ms)              AS avg_latency_ms,
            MIN(latency_ms)              AS min_latency_ms,
            MAX(latency_ms)              AS max_latency_ms,
            SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) AS success_count
        FROM llm_traces
    """).fetchone()
    by_provider = conn.execute("""
        SELECT provider, COUNT(*) as calls, SUM(estimated_cost_usd) as cost
        FROM llm_traces GROUP BY provider
    """).fetchall()
    conn.close()

    if row is None or row[0] == 0:
        return _empty_stats()

    provider_breakdown = [
        {"provider": r[0], "calls": r[1], "total_cost_usd": round(r[2] or 0.0, 6)}
        for r in by_provider
    ]
    return {
        "total_calls": row[0],
        "total_tokens": row[1] or 0,
        "total_cost_usd": round(row[2] or 0.0, 6),
        "avg_latency_ms": round(row[3] or 0.0, 2),
        "min_latency_ms": round(row[4] or 0.0, 2),
        "max_latency_ms": round(row[5] or 0.0, 2),
        "success_rate": round((row[6] / row[0]) * 100, 1) if row[0] else 100.0,
        "by_provider": provider_breakdown,
    }


def _empty_stats() -> dict:
    return {
        "total_calls": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "avg_latency_ms": 0.0,
        "min_latency_ms": 0.0,
        "max_latency_ms": 0.0,
        "success_rate": 100.0,
        "by_provider": [],
    }


# ---------------------------------------------------------------------------
# Context manager helper for instrumented blocks
# ---------------------------------------------------------------------------
@contextmanager
def trace_llm_call(
    provider: str, model: str, prompt: str, request_id: str = "", persist: bool = True
) -> Generator[TraceContext, None, None]:
    """
    Usage::

        with trace_llm_call("openai", "gpt-4o-mini", prompt) as ctx:
            ctx.start_span("generation")
            response = call_openai(...)
            ctx.end_span("generation")
        # trace auto-saved after block
    """
    ctx = TraceContext(provider=provider, model=model, prompt=prompt, request_id=request_id)
    try:
        yield ctx
    except Exception as exc:
        ctx.finish(success=False, error=str(exc))
        if persist:
            save_trace(ctx.trace)
        raise
    finally:
        if ctx.trace.latency_ms == 0.0:
            ctx.finish()
        if persist:
            save_trace(ctx.trace)
