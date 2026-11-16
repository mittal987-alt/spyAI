import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import {
  AlertTriangle,
  Bell,
  BookOpen,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock,
  FileClock,
  Filter,
  FlaskConical,
  Loader2,
  Plus,
  RefreshCw,
  Shield,
  Sparkles,
  SwitchCamera,
  Trash2,
  Zap,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// ─── Types ───────────────────────────────────────────────────────────────────

type Operator = "gt" | "gte" | "lt" | "lte" | "eq" | "neq";
type LogicMode = "AND" | "OR";
type Severity = "low" | "medium" | "high";

interface Condition {
  field: string;
  operator: Operator;
  value: number;
}

interface AlertRule {
  rule_id: string;
  name: string;
  description: string;
  logic: LogicMode;
  conditions: Condition[];
  severity: Severity;
  is_enabled: boolean;
  fire_count: number;
  created_at: string;
  updated_at: string;
}

interface RuleEvent {
  event_id: string;
  rule_id: string;
  rule_name: string;
  matched_at: string;
  event_type: string;
  payload: Record<string, unknown>;
}

interface FieldMeta {
  fields: string[];
  operators: string[];
  logic_modes: string[];
  field_descriptions: Record<string, string>;
  operator_descriptions: Record<string, string>;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const SEVERITY_STYLES: Record<Severity, string> = {
  low: "border-sky-500/30 bg-sky-500/10 text-sky-300",
  medium: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  high: "border-rose-500/30 bg-rose-500/10 text-rose-300",
};

const OPERATOR_LABELS: Record<string, string> = {
  gt: ">",
  gte: "≥",
  lt: "<",
  lte: "≤",
  eq: "=",
  neq: "≠",
};

function fmtDate(iso: string) {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString();
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function SeverityBadge({ severity }: { severity: Severity }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${SEVERITY_STYLES[severity]}`}
    >
      {severity.toUpperCase()}
    </span>
  );
}

function ConditionPill({ cond }: { cond: Condition }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/60 px-2 py-1 text-xs font-mono text-cyan-300">
      <span className="text-slate-400">{cond.field}</span>
      <span className="text-violet-400">{OPERATOR_LABELS[cond.operator] ?? cond.operator}</span>
      <span>{cond.value}</span>
    </span>
  );
}

// ─── Rule Card ────────────────────────────────────────────────────────────────

function RuleCard({
  rule,
  onToggle,
  onDelete,
  onViewEvents,
}: {
  rule: AlertRule;
  onToggle: (rule: AlertRule) => void;
  onDelete: (ruleId: string) => void;
  onViewEvents: (ruleId: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`rounded-2xl border bg-[#0d1424] transition ${
        rule.is_enabled
          ? "border-slate-700 hover:border-cyan-400/30"
          : "border-slate-800 opacity-60"
      }`}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-3 p-4">
        <div className="flex min-w-0 flex-1 items-start gap-3">
          {/* Status indicator */}
          <span
            className={`mt-1 flex h-2.5 w-2.5 shrink-0 rounded-full ${
              rule.is_enabled ? "bg-emerald-400" : "bg-slate-600"
            }`}
          />

          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <p className="truncate font-semibold text-white">{rule.name}</p>
              <SeverityBadge severity={rule.severity} />
              <span
                className={`rounded-full border px-2 py-0.5 text-xs font-medium ${
                  rule.logic === "AND"
                    ? "border-cyan-500/30 bg-cyan-500/10 text-cyan-300"
                    : "border-violet-500/30 bg-violet-500/10 text-violet-300"
                }`}
              >
                {rule.logic}
              </span>
            </div>

            {rule.description && (
              <p className="mt-1 text-xs text-slate-500">{rule.description}</p>
            )}

            {/* Conditions preview */}
            <div className="mt-2 flex flex-wrap gap-1.5">
              {rule.conditions.map((cond, i) => (
                <ConditionPill key={i} cond={cond} />
              ))}
            </div>

            <p className="mt-2 text-xs text-slate-600">
              Fired {rule.fire_count} time{rule.fire_count !== 1 ? "s" : ""} · Created {fmtDate(rule.created_at)}
            </p>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            title={rule.is_enabled ? "Disable" : "Enable"}
            onClick={() => onToggle(rule)}
            className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium transition ${
              rule.is_enabled
                ? "border-slate-700 text-slate-400 hover:border-rose-500/40 hover:text-rose-300"
                : "border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20"
            }`}
          >
            <SwitchCamera size={13} />
          </button>

          <button
            type="button"
            title="View fire history"
            onClick={() => onViewEvents(rule.rule_id)}
            className="rounded-lg border border-slate-700 px-2.5 py-1.5 text-xs font-medium text-slate-400 transition hover:border-cyan-400/40 hover:text-cyan-300"
          >
            <FileClock size={13} />
          </button>

          <button
            type="button"
            title="Delete rule"
            onClick={() => onDelete(rule.rule_id)}
            className="rounded-lg border border-slate-700 px-2.5 py-1.5 text-xs font-medium text-slate-400 transition hover:border-rose-500/40 hover:text-rose-300"
          >
            <Trash2 size={13} />
          </button>

          <button
            type="button"
            onClick={() => setExpanded((x) => !x)}
            className="rounded-lg border border-slate-700 px-2.5 py-1.5 text-xs text-slate-400 transition hover:border-slate-600"
          >
            {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>
        </div>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="border-t border-slate-800 px-5 py-4">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-slate-500">
            Conditions ({rule.logic})
          </p>
          <div className="space-y-2">
            {rule.conditions.map((cond, i) => (
              <div key={i} className="flex items-center gap-2 text-sm">
                <span className="w-5 text-center text-xs text-slate-600">{i + 1}.</span>
                <span className="font-mono text-slate-300">{cond.field}</span>
                <span className="rounded border border-slate-700 px-1.5 py-0.5 text-xs text-violet-400">
                  {OPERATOR_LABELS[cond.operator] ?? cond.operator}
                </span>
                <span className="font-mono text-cyan-300">{cond.value}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Create Rule Form ─────────────────────────────────────────────────────────

const DEFAULT_CONDITION: Condition = { field: "effect_size", operator: "gt", value: 0.3 };

function CreateRuleForm({
  fieldMeta,
  onCreated,
}: {
  fieldMeta: FieldMeta | null;
  onCreated: () => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [logic, setLogic] = useState<LogicMode>("AND");
  const [severity, setSeverity] = useState<Severity>("medium");
  const [conditions, setConditions] = useState<Condition[]>([{ ...DEFAULT_CONDITION }]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  function addCondition() {
    setConditions((cs) => [...cs, { ...DEFAULT_CONDITION }]);
  }

  function removeCondition(i: number) {
    setConditions((cs) => cs.filter((_, idx) => idx !== i));
  }

  function updateCondition<K extends keyof Condition>(i: number, key: K, val: Condition[K]) {
    setConditions((cs) => cs.map((c, idx) => (idx === i ? { ...c, [key]: val } : c)));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await axios.post(`${API_BASE}/api/v1/alert-rules`, {
        name,
        description,
        logic,
        severity,
        conditions,
      });
      setSuccess(true);
      setName("");
      setDescription("");
      setConditions([{ ...DEFAULT_CONDITION }]);
      onCreated();
      setTimeout(() => setSuccess(false), 3000);
    } catch (err: unknown) {
      const msg =
        axios.isAxiosError(err)
          ? err.response?.data?.detail ?? err.message
          : "Failed to create rule.";
      setError(typeof msg === "string" ? msg : JSON.stringify(msg));
    } finally {
      setSubmitting(false);
    }
  }

  const fields = fieldMeta?.fields ?? [];
  const operators = fieldMeta?.operators ?? [];

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl border border-slate-700 bg-[#0b1020] p-5 space-y-5"
    >
      <p className="text-sm font-semibold text-white flex items-center gap-2">
        <Plus size={15} className="text-cyan-400" />
        New Alert Rule
      </p>

      {/* Name & description */}
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label className="mb-1.5 block text-xs text-slate-400">Rule Name *</label>
          <input
            id="alert-rule-name"
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="High Drift + Low Recall"
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white placeholder-slate-600 focus:border-cyan-400/60 focus:outline-none"
          />
        </div>
        <div>
          <label className="mb-1.5 block text-xs text-slate-400">Description</label>
          <input
            id="alert-rule-desc"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Optional description"
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white placeholder-slate-600 focus:border-cyan-400/60 focus:outline-none"
          />
        </div>
      </div>

      {/* Logic & Severity */}
      <div className="flex flex-wrap gap-4">
        <div>
          <label className="mb-1.5 block text-xs text-slate-400">Condition Logic</label>
          <div className="flex gap-2">
            {(["AND", "OR"] as LogicMode[]).map((l) => (
              <button
                type="button"
                key={l}
                onClick={() => setLogic(l)}
                className={`rounded-lg border px-3 py-1.5 text-xs font-semibold transition ${
                  logic === l
                    ? l === "AND"
                      ? "border-cyan-500/50 bg-cyan-500/10 text-cyan-300"
                      : "border-violet-500/50 bg-violet-500/10 text-violet-300"
                    : "border-slate-700 text-slate-500 hover:border-slate-600"
                }`}
              >
                {l}
              </button>
            ))}
          </div>
          <p className="mt-1 text-xs text-slate-600">
            {logic === "AND" ? "ALL conditions must match" : "ANY condition must match"}
          </p>
        </div>

        <div>
          <label className="mb-1.5 block text-xs text-slate-400">Severity</label>
          <div className="flex gap-2">
            {(["low", "medium", "high"] as Severity[]).map((s) => (
              <button
                type="button"
                key={s}
                onClick={() => setSeverity(s)}
                className={`rounded-lg border px-3 py-1.5 text-xs font-semibold capitalize transition ${
                  severity === s
                    ? SEVERITY_STYLES[s]
                    : "border-slate-700 text-slate-500 hover:border-slate-600"
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Conditions */}
      <div>
        <label className="mb-2 block text-xs text-slate-400">
          Conditions
          <span className="ml-2 text-slate-600">
            ({logic === "AND" ? "all must pass" : "any must pass"})
          </span>
        </label>

        <div className="space-y-2">
          {conditions.map((cond, i) => (
            <div key={i} className="flex items-center gap-2 flex-wrap">
              {/* Field */}
              <select
                id={`cond-field-${i}`}
                value={cond.field}
                onChange={(e) => updateCondition(i, "field", e.target.value)}
                className="flex-1 min-w-[160px] rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-2 text-xs text-white focus:border-cyan-400/60 focus:outline-none"
              >
                {fields.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>

              {/* Operator */}
              <select
                id={`cond-op-${i}`}
                value={cond.operator}
                onChange={(e) => updateCondition(i, "operator", e.target.value as Operator)}
                className="w-20 rounded-lg border border-slate-700 bg-slate-900 px-2 py-2 text-xs text-violet-300 focus:border-violet-400/60 focus:outline-none"
              >
                {operators.map((op) => (
                  <option key={op} value={op}>
                    {OPERATOR_LABELS[op] ?? op} ({op})
                  </option>
                ))}
              </select>

              {/* Value */}
              <input
                id={`cond-val-${i}`}
                type="number"
                step="any"
                value={cond.value}
                onChange={(e) =>
                  updateCondition(i, "value", parseFloat(e.target.value) || 0)
                }
                className="w-24 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-2 text-xs text-cyan-300 focus:border-cyan-400/60 focus:outline-none"
              />

              {/* Remove */}
              {conditions.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeCondition(i)}
                  className="rounded-lg border border-slate-700 p-2 text-slate-500 transition hover:border-rose-500/40 hover:text-rose-300"
                >
                  <Trash2 size={12} />
                </button>
              )}

              {/* Field description */}
              {fieldMeta?.field_descriptions[cond.field] && (
                <span className="hidden text-xs text-slate-600 xl:block">
                  {fieldMeta.field_descriptions[cond.field]}
                </span>
              )}
            </div>
          ))}
        </div>

        {conditions.length < 10 && (
          <button
            type="button"
            onClick={addCondition}
            className="mt-2 flex items-center gap-1.5 text-xs text-cyan-400 transition hover:text-cyan-300"
          >
            <Plus size={12} /> Add condition
          </button>
        )}
      </div>

      {/* Error / Success */}
      {error && (
        <div className="flex items-start gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          {error}
        </div>
      )}
      {success && (
        <div className="flex items-center gap-2 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs text-emerald-300">
          <CheckCircle2 size={14} />
          Rule created successfully!
        </div>
      )}

      <button
        type="submit"
        disabled={submitting}
        id="create-alert-rule-submit"
        className="flex items-center gap-2 rounded-xl border border-cyan-400/30 bg-cyan-400/10 px-4 py-2 text-sm font-medium text-cyan-300 transition hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {submitting ? (
          <Loader2 size={14} className="animate-spin" />
        ) : (
          <Zap size={14} />
        )}
        {submitting ? "Creating…" : "Create Rule"}
      </button>
    </form>
  );
}

// ─── Event History Panel ──────────────────────────────────────────────────────

function EventHistoryPanel({
  ruleId,
  ruleName,
  onClose,
}: {
  ruleId: string;
  ruleName: string;
  onClose: () => void;
}) {
  const [events, setEvents] = useState<RuleEvent[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios
      .get(`${API_BASE}/api/v1/alert-rules/events/history`, {
        params: { rule_id: ruleId, limit: 50 },
      })
      .then((r) => setEvents(r.data.events ?? []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [ruleId]);

  return (
    <div className="rounded-2xl border border-slate-700 bg-[#0b1020] p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileClock size={15} className="text-cyan-400" />
          <p className="text-sm font-semibold text-white">
            Fire History — <span className="text-cyan-300">{ruleName}</span>
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-xs text-slate-500 hover:text-slate-300"
        >
          ✕ Close
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 size={20} className="animate-spin text-cyan-400" />
        </div>
      ) : events.length === 0 ? (
        <div className="py-8 text-center">
          <Clock size={22} className="mx-auto text-slate-600" />
          <p className="mt-3 text-sm text-slate-400">No events recorded yet</p>
          <p className="mt-1 text-xs text-slate-600">This rule hasn't fired yet.</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
          {events.map((ev) => (
            <div
              key={ev.event_id}
              className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3"
            >
              <div className="flex items-center gap-3">
                <Zap size={13} className="text-amber-400" />
                <div>
                  <p className="text-xs font-medium text-white">{ev.event_type}</p>
                  <p className="text-xs text-slate-500">{fmtDate(ev.matched_at)}</p>
                </div>
              </div>
              <span className="text-xs text-slate-600 font-mono">
                {ev.event_id.slice(0, 8)}…
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Test Evaluator ───────────────────────────────────────────────────────────

function TestEvaluator() {
  const [payload, setPayload] = useState(
    JSON.stringify(
      {
        effect_size: 0.45,
        adjusted_p_value: 0.01,
        precision_at_k: 0.38,
        recall_at_k: 0.55,
        answer_relevance_score: 0.7,
        severity: "high",
      },
      null,
      2,
    ),
  );
  const [eventType, setEventType] = useState("drift_completed");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    matched_count: number;
    matched_rules: AlertRule[];
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleEvaluate() {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const parsed = JSON.parse(payload);
      const res = await axios.post(`${API_BASE}/api/v1/alert-rules/evaluate`, {
        event_type: eventType,
        payload: parsed,
      });
      setResult(res.data);
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail ?? err.message);
      } else {
        setError("Evaluation failed");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-2xl border border-slate-700 bg-[#0b1020] p-5 space-y-4">
      <p className="text-sm font-semibold text-white flex items-center gap-2">
        <FlaskConical size={15} className="text-violet-400" />
        Live Rule Evaluator
      </p>
      <p className="text-xs text-slate-500">
        Paste a metric payload below and test which enabled rules would fire against it.
      </p>

      <div className="grid gap-4 sm:grid-cols-3">
        <div className="sm:col-span-2">
          <label className="mb-1.5 block text-xs text-slate-400">Payload JSON</label>
          <textarea
            id="eval-payload"
            rows={8}
            value={payload}
            onChange={(e) => setPayload(e.target.value)}
            spellCheck={false}
            className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2.5 font-mono text-xs text-cyan-200 placeholder-slate-600 focus:border-violet-400/60 focus:outline-none resize-none"
          />
        </div>
        <div className="space-y-4">
          <div>
            <label className="mb-1.5 block text-xs text-slate-400">Event Type</label>
            <input
              id="eval-event-type"
              value={eventType}
              onChange={(e) => setEventType(e.target.value)}
              placeholder="e.g. drift_completed"
              className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs text-white focus:border-violet-400/60 focus:outline-none"
            />
          </div>
          <button
            type="button"
            id="eval-rules-btn"
            onClick={handleEvaluate}
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 rounded-xl border border-violet-400/30 bg-violet-400/10 px-4 py-2.5 text-sm font-medium text-violet-300 transition hover:bg-violet-400/20 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Sparkles size={14} />
            )}
            {loading ? "Evaluating…" : "Evaluate Rules"}
          </button>

          {result !== null && (
            <div
              className={`rounded-xl border p-3 text-center ${
                result.matched_count > 0
                  ? "border-amber-500/30 bg-amber-500/10 text-amber-200"
                  : "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"
              }`}
            >
              <p className="text-2xl font-bold">{result.matched_count}</p>
              <p className="text-xs mt-1 opacity-80">
                rule{result.matched_count !== 1 ? "s" : ""} matched
              </p>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {result && result.matched_rules.length > 0 && (
        <div>
          <p className="mb-2 text-xs font-medium text-slate-400">Matched Rules:</p>
          <div className="space-y-2">
            {result.matched_rules.map((r) => (
              <div
                key={r.rule_id}
                className="flex items-center gap-3 rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3"
              >
                <Zap size={14} className="text-amber-400 shrink-0" />
                <div>
                  <p className="text-sm font-medium text-white">{r.name}</p>
                  <p className="text-xs text-slate-500">
                    {r.logic} · {r.conditions.length} condition{r.conditions.length !== 1 ? "s" : ""} · severity: {r.severity}
                  </p>
                </div>
                <SeverityBadge severity={r.severity} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Field Reference Panel ────────────────────────────────────────────────────

function FieldReference({ meta }: { meta: FieldMeta | null }) {
  const [open, setOpen] = useState(false);

  if (!meta) return null;

  return (
    <div className="rounded-2xl border border-slate-800 bg-[#0d1424]">
      <button
        type="button"
        onClick={() => setOpen((x) => !x)}
        className="flex w-full items-center justify-between p-4 text-sm font-medium text-slate-300 hover:text-white transition"
      >
        <span className="flex items-center gap-2">
          <BookOpen size={14} className="text-cyan-400" />
          Supported Fields & Operators Reference
        </span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {open && (
        <div className="border-t border-slate-800 p-4">
          <div className="grid gap-6 sm:grid-cols-2">
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                Metric Fields
              </p>
              <div className="space-y-1.5">
                {meta.fields.map((f) => (
                  <div key={f} className="flex items-start gap-2">
                    <code className="shrink-0 rounded bg-slate-800 px-1.5 py-0.5 text-xs text-cyan-300">
                      {f}
                    </code>
                    <span className="text-xs text-slate-500">
                      {meta.field_descriptions[f]}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">
                Operators
              </p>
              <div className="space-y-1.5">
                {meta.operators.map((op) => (
                  <div key={op} className="flex items-center gap-2">
                    <code className="w-10 rounded bg-slate-800 px-1.5 py-0.5 text-center text-xs text-violet-300">
                      {OPERATOR_LABELS[op] ?? op}
                    </code>
                    <span className="text-xs text-slate-500">
                      {meta.operator_descriptions[op]} ({op})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function AlertRulesEngine() {
  const [rules, setRules] = useState<AlertRule[]>([]);
  const [fieldMeta, setFieldMeta] = useState<FieldMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState<"all" | "enabled" | "disabled">("all");
  const [viewEventsFor, setViewEventsFor] = useState<{ id: string; name: string } | null>(null);
  const [activeTab, setActiveTab] = useState<"rules" | "create" | "test">("rules");

  const loadRules = useCallback(async (showRefresh = false) => {
    if (showRefresh) setRefreshing(true);
    else setLoading(true);
    try {
      const [rulesRes, fieldsRes] = await Promise.all([
        axios.get(`${API_BASE}/api/v1/alert-rules`),
        axios.get(`${API_BASE}/api/v1/alert-rules/fields`),
      ]);
      setRules(rulesRes.data.rules ?? []);
      setFieldMeta(fieldsRes.data);
    } catch {
      /* swallow */
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadRules();
  }, [loadRules]);

  async function handleToggle(rule: AlertRule) {
    try {
      await axios.patch(`${API_BASE}/api/v1/alert-rules/${rule.rule_id}/toggle`, {
        is_enabled: !rule.is_enabled,
      });
      void loadRules(true);
    } catch {/* swallow */}
  }

  async function handleDelete(ruleId: string) {
    if (!confirm("Delete this alert rule permanently?")) return;
    try {
      await axios.delete(`${API_BASE}/api/v1/alert-rules/${ruleId}`);
      void loadRules(true);
    } catch {/* swallow */}
  }

  const filtered = rules.filter((r) => {
    if (filter === "enabled") return r.is_enabled;
    if (filter === "disabled") return !r.is_enabled;
    return true;
  });

  const enabledCount = rules.filter((r) => r.is_enabled).length;
  const totalFires = rules.reduce((acc, r) => acc + r.fire_count, 0);

  const tabs = [
    { id: "rules" as const, label: "Rules", icon: Shield },
    { id: "create" as const, label: "New Rule", icon: Plus },
    { id: "test" as const, label: "Test Evaluator", icon: FlaskConical },
  ];

  return (
    <div className="space-y-6">
      {/* Stats bar */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          { label: "Total Rules", value: rules.length, color: "text-white" },
          { label: "Enabled", value: enabledCount, color: "text-emerald-300" },
          { label: "Disabled", value: rules.length - enabledCount, color: "text-slate-400" },
          { label: "Total Fires", value: totalFires, color: "text-amber-300" },
        ].map((stat) => (
          <div
            key={stat.label}
            className="rounded-2xl border border-slate-800 bg-[#0d1424] p-4 text-center"
          >
            <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
            <p className="mt-1 text-xs text-slate-500">{stat.label}</p>
          </div>
        ))}
      </div>

      {/* Tab nav */}
      <div className="flex gap-1 rounded-xl border border-slate-800 bg-slate-900/50 p-1">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            id={`alert-rules-tab-${id}`}
            onClick={() => setActiveTab(id)}
            className={`flex flex-1 items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-medium transition ${
              activeTab === id
                ? "bg-slate-800 text-white shadow"
                : "text-slate-500 hover:text-slate-300"
            }`}
          >
            <Icon size={13} />
            {label}
          </button>
        ))}

        <button
          type="button"
          onClick={() => void loadRules(true)}
          className="ml-auto flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs text-slate-500 hover:text-slate-300 transition"
        >
          <RefreshCw size={13} className={refreshing ? "animate-spin" : ""} />
        </button>
      </div>

      {/* Tab: Rules list */}
      {activeTab === "rules" && (
        <div className="space-y-4">
          {/* Filter */}
          <div className="flex items-center gap-2">
            <Filter size={13} className="text-slate-500" />
            {(["all", "enabled", "disabled"] as const).map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setFilter(f)}
                className={`rounded-lg border px-2.5 py-1 text-xs capitalize transition ${
                  filter === f
                    ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-300"
                    : "border-slate-700 text-slate-500 hover:border-slate-600"
                }`}
              >
                {f}
              </button>
            ))}
          </div>

          {loading ? (
            <div className="flex flex-col items-center justify-center gap-3 py-16">
              <Loader2 size={28} className="animate-spin text-cyan-400" />
              <p className="text-sm text-slate-400">Loading alert rules…</p>
            </div>
          ) : filtered.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-slate-700 py-16 text-center">
              <Bell size={28} className="mx-auto text-slate-600" />
              <p className="mt-3 text-sm text-slate-300">
                {rules.length === 0 ? "No alert rules yet" : "No rules match this filter"}
              </p>
              <p className="mt-1 text-xs text-slate-500">
                {rules.length === 0
                  ? "Create your first rule in the \"New Rule\" tab."
                  : "Try changing the filter above."}
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {filtered.map((rule) => (
                <RuleCard
                  key={rule.rule_id}
                  rule={rule}
                  onToggle={handleToggle}
                  onDelete={handleDelete}
                  onViewEvents={(id) =>
                    setViewEventsFor({ id, name: rule.name })
                  }
                />
              ))}
            </div>
          )}

          {/* Event history inline panel */}
          {viewEventsFor && (
            <EventHistoryPanel
              ruleId={viewEventsFor.id}
              ruleName={viewEventsFor.name}
              onClose={() => setViewEventsFor(null)}
            />
          )}
        </div>
      )}

      {/* Tab: Create */}
      {activeTab === "create" && (
        <div className="space-y-4">
          <CreateRuleForm
            fieldMeta={fieldMeta}
            onCreated={() => {
              void loadRules(true);
              setActiveTab("rules");
            }}
          />
          <FieldReference meta={fieldMeta} />
        </div>
      )}

      {/* Tab: Test */}
      {activeTab === "test" && (
        <div className="space-y-4">
          <TestEvaluator />
          <FieldReference meta={fieldMeta} />
        </div>
      )}
    </div>
  );
}
