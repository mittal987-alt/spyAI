"""
Smart Alert Rules Engine for ModelAutopsy AI.

Users define named rules with one or more metric conditions joined by AND/OR.
When an event (drift completed, RAG evaluated, etc.) is dispatched, all enabled
rules are evaluated against the event payload. If a rule matches, an alert is
generated and optionally forwarded to registered webhooks.

Supported metric fields (checked against event payload):
  - effect_size          (drift)
  - adjusted_p_value     (drift)
  - precision_at_k       (RAG)
  - recall_at_k          (RAG)
  - ndcg_at_k            (RAG)
  - answer_relevance_score (RAG)
  - context_support_coverage (RAG)
  - hallucination_risk_score (hallucination)
  - severity_level       (mapped: low=1, medium=2, high=3)

Supported operators: gt, gte, lt, lte, eq, neq
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ALERT_RULES_DB_PATH, ALERT_RULES_MAX_CONDITIONS, ALERT_RULES_MAX_RULES

logger = logging.getLogger(__name__)

VALID_OPERATORS = {"gt", "gte", "lt", "lte", "eq", "neq"}
VALID_LOGIC = {"AND", "OR"}
VALID_FIELDS = {
    "effect_size",
    "adjusted_p_value",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "answer_relevance_score",
    "context_support_coverage",
    "hallucination_risk_score",
    "severity_level",
}

SEVERITY_MAP = {"low": 1, "medium": 2, "high": 3}


# ---------------------------------------------------------------------------
# DB Initialisation
# ---------------------------------------------------------------------------


def initialize_rules_db(db_path: str | Path = ALERT_RULES_DB_PATH) -> None:
    """Create tables for rules and their evaluation history."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_rules (
                rule_id     TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                logic       TEXT NOT NULL DEFAULT 'AND',
                conditions  TEXT NOT NULL,
                severity    TEXT NOT NULL DEFAULT 'medium',
                is_enabled  INTEGER NOT NULL DEFAULT 1,
                fire_count  INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_rule_events (
                event_id    TEXT PRIMARY KEY,
                rule_id     TEXT NOT NULL,
                rule_name   TEXT NOT NULL,
                matched_at  TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rule_events_rule_id
            ON alert_rule_events(rule_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rule_events_matched_at
            ON alert_rule_events(matched_at)
            """
        )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def create_rule(
    name: str,
    conditions: list[dict[str, Any]],
    logic: str = "AND",
    description: str = "",
    severity: str = "medium",
    db_path: str | Path = ALERT_RULES_DB_PATH,
) -> dict[str, Any]:
    """
    Create and persist a new alert rule.

    Each condition is a dict with keys:
      - field    : one of VALID_FIELDS
      - operator : one of VALID_OPERATORS
      - value    : numeric threshold
    """
    initialize_rules_db(db_path)

    name = name.strip()
    logic = logic.strip().upper()
    severity = severity.strip().lower()

    if not name:
        raise ValueError("Rule name must not be empty.")
    if logic not in VALID_LOGIC:
        raise ValueError(f"logic must be one of: {VALID_LOGIC}")
    if severity not in {"low", "medium", "high"}:
        raise ValueError("severity must be low, medium, or high")
    if not conditions:
        raise ValueError("At least one condition is required.")
    if len(conditions) > ALERT_RULES_MAX_CONDITIONS:
        raise ValueError(
            f"A rule may have at most {ALERT_RULES_MAX_CONDITIONS} conditions."
        )

    # Validate each condition
    validated: list[dict[str, Any]] = []
    for idx, cond in enumerate(conditions):
        field = str(cond.get("field", "")).strip()
        operator = str(cond.get("operator", "")).strip().lower()
        value = cond.get("value")

        if field not in VALID_FIELDS:
            raise ValueError(
                f"Condition {idx}: field '{field}' is not supported. "
                f"Valid fields: {sorted(VALID_FIELDS)}"
            )
        if operator not in VALID_OPERATORS:
            raise ValueError(
                f"Condition {idx}: operator '{operator}' is not valid. "
                f"Valid operators: {sorted(VALID_OPERATORS)}"
            )
        if value is None or not isinstance(value, (int, float)):
            raise ValueError(
                f"Condition {idx}: value must be a numeric threshold."
            )

        validated.append({"field": field, "operator": operator, "value": float(value)})

    # Enforce max rule count
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM alert_rules").fetchone()[0]
    if count >= ALERT_RULES_MAX_RULES:
        raise ValueError(
            f"Cannot create more than {ALERT_RULES_MAX_RULES} rules. "
            "Delete an existing rule first."
        )

    now = datetime.now(timezone.utc).isoformat()
    rule_id = str(uuid.uuid4())

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO alert_rules
                (rule_id, name, description, logic, conditions, severity, is_enabled, fire_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
            """,
            (rule_id, name, description, logic, json.dumps(validated), severity, now, now),
        )

    logger.info("Alert rule created: %s (%s)", name, rule_id)
    return get_rule(rule_id, db_path=db_path)  # type: ignore[return-value]


def list_rules(
    enabled_only: bool = False,
    db_path: str | Path = ALERT_RULES_DB_PATH,
) -> list[dict[str, Any]]:
    """Return all stored alert rules."""
    initialize_rules_db(db_path)

    query = "SELECT * FROM alert_rules"
    if enabled_only:
        query += " WHERE is_enabled = 1"
    query += " ORDER BY created_at DESC"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()

    return [_deserialize_rule(dict(row)) for row in rows]


def get_rule(
    rule_id: str,
    db_path: str | Path = ALERT_RULES_DB_PATH,
) -> dict[str, Any] | None:
    """Return a single rule by ID, or None."""
    initialize_rules_db(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM alert_rules WHERE rule_id = ?", (rule_id,)
        ).fetchone()

    if row is None:
        return None
    return _deserialize_rule(dict(row))


def update_rule(
    rule_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    logic: str | None = None,
    conditions: list[dict[str, Any]] | None = None,
    severity: str | None = None,
    is_enabled: bool | None = None,
    db_path: str | Path = ALERT_RULES_DB_PATH,
) -> dict[str, Any]:
    """Update mutable fields of an existing rule."""
    existing = get_rule(rule_id, db_path=db_path)
    if existing is None:
        raise ValueError(f"Rule '{rule_id}' not found.")

    updated_name = name.strip() if name is not None else existing["name"]
    updated_desc = description if description is not None else existing["description"]
    updated_logic = (logic.strip().upper() if logic is not None else existing["logic"])
    updated_severity = (severity.strip().lower() if severity is not None else existing["severity"])
    updated_enabled = int(is_enabled) if is_enabled is not None else existing["is_enabled"]

    if updated_logic not in VALID_LOGIC:
        raise ValueError(f"logic must be one of: {VALID_LOGIC}")
    if updated_severity not in {"low", "medium", "high"}:
        raise ValueError("severity must be low, medium, or high")

    if conditions is not None:
        if not conditions:
            raise ValueError("At least one condition is required.")
        if len(conditions) > ALERT_RULES_MAX_CONDITIONS:
            raise ValueError(
                f"A rule may have at most {ALERT_RULES_MAX_CONDITIONS} conditions."
            )
        validated_conds: list[dict] = []
        for idx, cond in enumerate(conditions):
            field = str(cond.get("field", "")).strip()
            operator = str(cond.get("operator", "")).strip().lower()
            value = cond.get("value")
            if field not in VALID_FIELDS:
                raise ValueError(f"Condition {idx}: field '{field}' is not supported.")
            if operator not in VALID_OPERATORS:
                raise ValueError(f"Condition {idx}: operator '{operator}' is not valid.")
            if value is None or not isinstance(value, (int, float)):
                raise ValueError(f"Condition {idx}: value must be numeric.")
            validated_conds.append({"field": field, "operator": operator, "value": float(value)})
        conditions_json = json.dumps(validated_conds)
    else:
        conditions_json = json.dumps(existing["conditions"])

    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE alert_rules
            SET name=?, description=?, logic=?, conditions=?, severity=?, is_enabled=?, updated_at=?
            WHERE rule_id=?
            """,
            (updated_name, updated_desc, updated_logic, conditions_json, updated_severity, updated_enabled, now, rule_id),
        )

    return get_rule(rule_id, db_path=db_path)  # type: ignore[return-value]


def delete_rule(
    rule_id: str,
    db_path: str | Path = ALERT_RULES_DB_PATH,
) -> bool:
    """Delete a rule by ID. Returns True if a row was deleted."""
    initialize_rules_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM alert_rules WHERE rule_id = ?", (rule_id,)
        )
    deleted = cursor.rowcount > 0
    if deleted:
        logger.info("Alert rule deleted: %s", rule_id)
    return deleted


def toggle_rule(
    rule_id: str,
    is_enabled: bool,
    db_path: str | Path = ALERT_RULES_DB_PATH,
) -> dict[str, Any]:
    """Enable or disable a rule."""
    return update_rule(rule_id, is_enabled=is_enabled, db_path=db_path)


# ---------------------------------------------------------------------------
# Rule Evaluation
# ---------------------------------------------------------------------------


def _extract_value(field: str, payload: dict[str, Any]) -> float | None:
    """Pull a numeric value from an event payload, supporting nested metrics."""
    if field == "severity_level":
        sev = payload.get("severity", payload.get("severity_level", ""))
        return float(SEVERITY_MAP.get(str(sev).lower(), 0))

    # Try top-level key first, then look inside common nested keys.
    for source in [payload, payload.get("metrics", {}), payload.get("summary", {}), payload.get("details", {})]:
        if isinstance(source, dict) and field in source:
            val = source[field]
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def _eval_condition(cond: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Evaluate a single condition against an event payload."""
    actual = _extract_value(cond["field"], payload)
    if actual is None:
        return False  # Field not present — condition not satisfied

    threshold = cond["value"]
    op = cond["operator"]

    if op == "gt":
        return actual > threshold
    if op == "gte":
        return actual >= threshold
    if op == "lt":
        return actual < threshold
    if op == "lte":
        return actual <= threshold
    if op == "eq":
        return actual == threshold
    if op == "neq":
        return actual != threshold
    return False


def evaluate_rules(
    event_type: str,
    payload: dict[str, Any],
    db_path: str | Path = ALERT_RULES_DB_PATH,
) -> list[dict[str, Any]]:
    """
    Evaluate all enabled rules against the given event payload.
    Returns a list of rules that matched, and persists a rule_event for each.
    """
    rules = list_rules(enabled_only=True, db_path=db_path)
    matched: list[dict[str, Any]] = []

    for rule in rules:
        conditions: list[dict] = rule["conditions"]
        logic: str = rule["logic"]

        results = [_eval_condition(c, payload) for c in conditions]

        if logic == "AND":
            fired = all(results)
        else:  # OR
            fired = any(results)

        if fired:
            matched.append(rule)
            _record_event(rule, event_type, payload, db_path=db_path)

    return matched


def _record_event(
    rule: dict[str, Any],
    event_type: str,
    payload: dict[str, Any],
    db_path: str | Path = ALERT_RULES_DB_PATH,
) -> None:
    """Persist a rule-fire event and increment fire count."""
    now = datetime.now(timezone.utc).isoformat()
    event_id = str(uuid.uuid4())

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO alert_rule_events
                    (event_id, rule_id, rule_name, matched_at, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    rule["rule_id"],
                    rule["name"],
                    now,
                    event_type,
                    json.dumps(payload, default=str),
                ),
            )
            conn.execute(
                "UPDATE alert_rules SET fire_count = fire_count + 1 WHERE rule_id = ?",
                (rule["rule_id"],),
            )
    except Exception as exc:
        logger.warning("Failed to record rule event for rule %s: %s", rule["rule_id"], exc)


def list_rule_events(
    rule_id: str | None = None,
    limit: int = 50,
    db_path: str | Path = ALERT_RULES_DB_PATH,
) -> list[dict[str, Any]]:
    """Return recent rule-fire events, optionally filtered by rule_id."""
    initialize_rules_db(db_path)

    if rule_id:
        query = (
            "SELECT * FROM alert_rule_events WHERE rule_id = ? "
            "ORDER BY matched_at DESC LIMIT ?"
        )
        params: tuple = (rule_id, limit)
    else:
        query = "SELECT * FROM alert_rule_events ORDER BY matched_at DESC LIMIT ?"
        params = (limit,)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    results = []
    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json", "{}"))
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deserialize_rule(row: dict[str, Any]) -> dict[str, Any]:
    row["conditions"] = json.loads(row.get("conditions", "[]"))
    row["is_enabled"] = bool(row.get("is_enabled", 1))
    row["fire_count"] = int(row.get("fire_count", 0))
    return row


def get_supported_fields() -> dict[str, Any]:
    """Return the list of supported fields, operators, and examples."""
    return {
        "fields": sorted(VALID_FIELDS),
        "operators": sorted(VALID_OPERATORS),
        "logic_modes": sorted(VALID_LOGIC),
        "field_descriptions": {
            "effect_size": "KS drift effect size (0–1)",
            "adjusted_p_value": "Bonferroni-adjusted p-value from drift detection",
            "precision_at_k": "RAG precision@K retrieval metric (0–1)",
            "recall_at_k": "RAG recall@K retrieval metric (0–1)",
            "ndcg_at_k": "Normalized discounted cumulative gain (0–1)",
            "answer_relevance_score": "Answer relevance to query (0–1)",
            "context_support_coverage": "Fraction of answer supported by context (0–1)",
            "hallucination_risk_score": "Composite hallucination risk (0–100)",
            "severity_level": "Alert severity mapped to numeric: low=1, medium=2, high=3",
        },
        "operator_descriptions": {
            "gt": "greater than",
            "gte": "greater than or equal to",
            "lt": "less than",
            "lte": "less than or equal to",
            "eq": "equal to",
            "neq": "not equal to",
        },
    }
