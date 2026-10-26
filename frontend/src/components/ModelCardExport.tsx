import { useState, useCallback } from "react";
import {
  Shield,
  FileText,
  Download,
  CheckCircle,
  AlertTriangle,
  XCircle,
  ChevronDown,
  ChevronRight,
  Loader2,
  ClipboardList,
  Info,
  Award,
} from "lucide-react";

// ── Types ──────────────────────────────────────────────────────────────────────

interface ComplianceRequirement {
  id: string;
  category: string;
  description: string;
  status: "pass" | "warning" | "fail" | "not_applicable";
  score: number;
  notes: string;
  remediation?: string;
}

interface FrameworkAudit {
  framework: string;
  version: string;
  overall_score: number;
  risk_level: string;
  requirements: ComplianceRequirement[];
  summary: string;
}

interface ModelCardData {
  metadata: {
    report_id: string;
    generated_at: string;
    schema_version: string;
  };
  model_overview: {
    name: string;
    version: string;
    type: string;
    developer: string;
    intended_use: string;
    out_of_scope_use: string;
  };
  compliance_audits: FrameworkAudit[];
  overall_compliance_score: number;
  highest_risk_level: string;
  executive_summary: string;
  recommendations: string[];
  metrics_summary: {
    rag_included: boolean;
    drift_included: boolean;
    security_included: boolean;
  };
}

// ── Sub-components ─────────────────────────────────────────────────────────────

const StatusIcon = ({ status }: { status: string }) => {
  if (status === "pass")
    return <CheckCircle className="mc-icon-pass" size={15} />;
  if (status === "warning")
    return <AlertTriangle className="mc-icon-warn" size={15} />;
  if (status === "fail") return <XCircle className="mc-icon-fail" size={15} />;
  return <Info className="mc-icon-na" size={15} />;
};

const ScoreRing = ({
  score,
  size = 90,
  risk,
}: {
  score: number;
  size?: number;
  risk: string;
}) => {
  const r = (size - 12) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (score / 100) * circ;
  const color =
    score >= 80 ? "#22c55e" : score >= 55 ? "#f59e0b" : "#ef4444";
  return (
    <div className="mc-score-ring" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke="#1e293b"
          strokeWidth={10}
          fill="none"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={color}
          strokeWidth={10}
          fill="none"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transform: "rotate(-90deg)", transformOrigin: "50% 50%" }}
        />
      </svg>
      <div className="mc-score-ring-label">
        <span className="mc-score-num" style={{ color }}>
          {score}
        </span>
        <span className="mc-score-risk">{risk}</span>
      </div>
    </div>
  );
};

const RequirementRow = ({ req }: { req: ComplianceRequirement }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className={`mc-req-row mc-req-${req.status}`}>
      <button className="mc-req-header" onClick={() => setOpen(!open)}>
        <div className="mc-req-left">
          <StatusIcon status={req.status} />
          <span className="mc-req-cat">{req.category}</span>
          <span className="mc-req-desc">{req.description}</span>
        </div>
        <div className="mc-req-right">
          <span className="mc-req-score">{req.score}%</span>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </div>
      </button>
      {open && (
        <div className="mc-req-body">
          <p className="mc-req-notes">{req.notes}</p>
          {req.remediation && (
            <p className="mc-req-remediation">
              <strong>Remediation:</strong> {req.remediation}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

const FrameworkPanel = ({ audit }: { audit: FrameworkAudit }) => {
  const [open, setOpen] = useState(true);
  const passCount = audit.requirements.filter((r) => r.status === "pass").length;
  const warnCount = audit.requirements.filter(
    (r) => r.status === "warning"
  ).length;
  const failCount = audit.requirements.filter((r) => r.status === "fail").length;

  return (
    <div className="mc-framework-panel">
      <button className="mc-framework-header" onClick={() => setOpen(!open)}>
        <div className="mc-fw-left">
          <Award size={18} />
          <div>
            <span className="mc-fw-name">{audit.framework}</span>
            <span className="mc-fw-version">v{audit.version}</span>
          </div>
        </div>
        <div className="mc-fw-right">
          <span className="mc-fw-tag mc-tag-pass">✓ {passCount}</span>
          <span className="mc-fw-tag mc-tag-warn">⚠ {warnCount}</span>
          <span className="mc-fw-tag mc-tag-fail">✗ {failCount}</span>
          <ScoreRing score={audit.overall_score} size={70} risk={audit.risk_level} />
          {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </div>
      </button>
      {open && (
        <div className="mc-framework-body">
          <p className="mc-fw-summary">{audit.summary}</p>
          <div className="mc-req-list">
            {audit.requirements.map((r) => (
              <RequirementRow key={r.id} req={r} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ── Main Component ─────────────────────────────────────────────────────────────

const API = "http://localhost:8000";

interface FormState {
  model_name: string;
  model_version: string;
  model_type: string;
  developer_or_team: string;
  intended_use: string;
  out_of_scope_use: string;
  custom_notes: string;
}

export default function ModelCardExport() {
  const [form, setForm] = useState<FormState>({
    model_name: "ModelAutopsy Diagnostic Pipeline",
    model_version: "1.0.0",
    model_type: "Hybrid Deep Learning & RAG Evaluation System",
    developer_or_team: "AI Reliability & MLOps Engineering Team",
    intended_use:
      "Production AI model diagnostics, drift tracking, retrieval evaluation, and adversarial security auditing.",
    out_of_scope_use:
      "Autonomous safety-critical decisions without human verification.",
    custom_notes: "",
  });
  const [card, setCard] = useState<ModelCardData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  const handleInput = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setForm((f) => ({ ...f, [e.target.name]: e.target.value }));
  };

  const generate = useCallback(async () => {
    setLoading(true);
    setError(null);
    setCard(null);
    try {
      const res = await fetch(`${API}/api/v1/compliance/model-card`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail ?? `HTTP ${res.status}`);
      }
      setCard(await res.json());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [form]);

  const downloadMarkdown = useCallback(async () => {
    setDownloading(true);
    try {
      const res = await fetch(
        `${API}/api/v1/compliance/model-card/markdown`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(form),
        }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const cd = res.headers.get("Content-Disposition") ?? "";
      const match = cd.match(/filename=([^\s;]+)/);
      a.download = match?.[1] ?? "model-card.md";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDownloading(false);
    }
  }, [form]);

  const riskColor: Record<string, string> = {
    Low: "#22c55e",
    Medium: "#f59e0b",
    High: "#f97316",
    Critical: "#ef4444",
  };

  return (
    <div className="mc-root">
      {/* ── Header ── */}
      <div className="mc-header">
        <div className="mc-header-icon">
          <Shield size={26} />
        </div>
        <div>
          <h1 className="mc-title">Model Card & Regulatory Compliance</h1>
          <p className="mc-subtitle">
            Auto-generates an audited model card aligned with EU AI Act, NIST AI
            RMF, and ISO/IEC 42001
          </p>
        </div>
      </div>

      <div className="mc-layout">
        {/* ── Left: Form ── */}
        <div className="mc-form-panel">
          <div className="mc-section-label">
            <ClipboardList size={15} />
            Model Metadata
          </div>

          {(
            [
              { name: "model_name", label: "Model Name" },
              { name: "model_version", label: "Version" },
              { name: "model_type", label: "Model Type" },
              { name: "developer_or_team", label: "Developer / Team" },
            ] as { name: keyof FormState; label: string }[]
          ).map(({ name, label }) => (
            <label key={name} className="mc-label">
              {label}
              <input
                className="mc-input"
                name={name}
                value={form[name]}
                onChange={handleInput}
              />
            </label>
          ))}

          <label className="mc-label">
            Intended Use
            <textarea
              className="mc-textarea"
              name="intended_use"
              value={form.intended_use}
              onChange={handleInput}
              rows={3}
            />
          </label>

          <label className="mc-label">
            Out-of-Scope Use
            <textarea
              className="mc-textarea"
              name="out_of_scope_use"
              value={form.out_of_scope_use}
              onChange={handleInput}
              rows={2}
            />
          </label>

          <label className="mc-label">
            Additional Notes
            <textarea
              className="mc-textarea"
              name="custom_notes"
              value={form.custom_notes}
              onChange={handleInput}
              rows={3}
              placeholder="Fairness constraints, data governance policies, training details…"
            />
          </label>

          <button
            className="mc-btn-generate"
            onClick={generate}
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader2 size={16} className="mc-spin" />
                Auditing…
              </>
            ) : (
              <>
                <Shield size={16} />
                Run Compliance Audit
              </>
            )}
          </button>

          {error && <div className="mc-error">{error}</div>}
        </div>

        {/* ── Right: Results ── */}
        <div className="mc-results-panel">
          {!card && !loading && (
            <div className="mc-empty">
              <Shield size={54} className="mc-empty-icon" />
              <p>Fill in model metadata and run the audit to generate a compliance report.</p>
              <ul className="mc-framework-list">
                <li>🇪🇺 EU AI Act — Risk classification & transparency</li>
                <li>🛡️ NIST AI RMF — Govern, Map, Measure, Manage</li>
                <li>📋 ISO/IEC 42001 — AI Management System</li>
              </ul>
            </div>
          )}

          {loading && (
            <div className="mc-loading">
              <Loader2 size={42} className="mc-spin" />
              <p>Running regulatory compliance audit…</p>
            </div>
          )}

          {card && (
            <>
              {/* ── Overall Score Bar ── */}
              <div className="mc-overall-card">
                <div className="mc-overall-left">
                  <ScoreRing
                    score={card.overall_compliance_score}
                    size={110}
                    risk={card.highest_risk_level}
                  />
                  <div className="mc-overall-meta">
                    <h2 className="mc-overall-title">
                      {card.model_overview.name}
                    </h2>
                    <p className="mc-overall-sub">
                      v{card.model_overview.version} &nbsp;·&nbsp;{" "}
                      {card.model_overview.type}
                    </p>
                    <p className="mc-overall-id">
                      Report {card.metadata.report_id}
                    </p>
                    <div className="mc-risk-badge" style={{ background: riskColor[card.highest_risk_level] ?? "#6366f1" }}>
                      {card.highest_risk_level} Risk
                    </div>
                  </div>
                </div>
                <button
                  className="mc-btn-download"
                  onClick={downloadMarkdown}
                  disabled={downloading}
                >
                  {downloading ? (
                    <Loader2 size={15} className="mc-spin" />
                  ) : (
                    <Download size={15} />
                  )}
                  Export .md
                </button>
              </div>

              {/* ── Executive Summary ── */}
              <div className="mc-exec-summary">
                <div className="mc-section-label">
                  <FileText size={14} />
                  Executive Summary
                </div>
                <p>{card.executive_summary}</p>
              </div>

              {/* ── Recommendations ── */}
              {card.recommendations.length > 0 && (
                <div className="mc-recommendations">
                  <div className="mc-section-label">
                    <AlertTriangle size={14} />
                    Top Recommendations
                  </div>
                  <ul>
                    {card.recommendations.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* ── Framework Panels ── */}
              <div className="mc-section-label" style={{ marginTop: "1.25rem" }}>
                <Award size={14} />
                Framework Audits
              </div>
              {card.compliance_audits.map((a) => (
                <FrameworkPanel key={a.framework} audit={a} />
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
