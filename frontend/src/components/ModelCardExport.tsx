import "./ModelCardExport.css";
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
  Award,
  Info,
} from "lucide-react";

// ── Types matching the REAL backend schema ─────────────────────────────────────

interface EuArticleAudit {
  article: string;
  status: "PASS" | "WARNING" | "FAIL";
  evidence: string;
  score: number;
}

interface NistFunction {
  status: string;
  evidence: string;
  score: number;
}

interface IsoControl {
  clause: string;
  name: string;
  status: string;
  details: string;
}

interface CardData {
  metadata: {
    report_id: string;
    generated_at: string;
    overall_compliance_score: number;
    overall_compliance_status: string;
    compliance_rating: string;
  };
  model_overview: {
    name: string;
    version: string;
    type: string;
    developer: string;
    intended_use: string;
    out_of_scope_use: string;
    custom_notes: string;
  };
  regulatory_compliance: {
    eu_ai_act: {
      framework: string;
      assigned_risk_tier: string;
      score: number;
      status: string;
      article_audits: EuArticleAudit[];
    };
    nist_ai_rmf: {
      framework: string;
      score: number;
      core_functions: Record<string, NistFunction>;
    };
    iso_42001: {
      framework: string;
      score: number;
      controls: IsoControl[];
    };
  };
  performance_and_robustness: {
    rag_evaluation: Record<string, unknown>;
    distribution_drift: Record<string, unknown>;
    security_guardrails: Record<string, unknown>;
  };
  mitigations_and_safeguards: string[];
}

// ── Helpers ────────────────────────────────────────────────────────────────────

const statusColor = (s: string) =>
  s === "PASS" || s === "Compliant" || s === "Implemented" || s === "Verified" || s === "Active"
    ? "#4ade80"
    : s === "WARNING" || s === "Conditional Pass"
    ? "#fbbf24"
    : "#f87171";

const StatusPill = ({ status }: { status: string }) => {
  const color = statusColor(status);
  const icon =
    status === "PASS" || status === "Compliant" || status === "Implemented" || status === "Verified" || status === "Active" ? (
      <CheckCircle size={12} />
    ) : status === "WARNING" || status === "Conditional Pass" ? (
      <AlertTriangle size={12} />
    ) : (
      <XCircle size={12} />
    );
  return (
    <span className="mc-status-pill" style={{ color, borderColor: color + "44", background: color + "14" }}>
      {icon} {status}
    </span>
  );
};

const ScoreRing = ({ score, size = 100, label }: { score: number; size?: number; label?: string }) => {
  const r = (size - 14) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ - (score / 100) * circ;
  const color = score >= 85 ? "#22c55e" : score >= 65 ? "#f59e0b" : "#ef4444";
  return (
    <div className="mc-score-ring" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} stroke="#1e293b" strokeWidth={12} fill="none" />
        <circle
          cx={size / 2} cy={size / 2} r={r}
          stroke={color} strokeWidth={12} fill="none"
          strokeDasharray={circ} strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transform: "rotate(-90deg)", transformOrigin: "50% 50%", transition: "stroke-dashoffset 0.6s ease" }}
        />
      </svg>
      <div className="mc-score-ring-label">
        <span className="mc-score-num" style={{ color }}>{score}</span>
        {label && <span className="mc-score-sub">{label}</span>}
      </div>
    </div>
  );
};

// ── Section blocks ─────────────────────────────────────────────────────────────

const EuPanel = ({ eu }: { eu: CardData["regulatory_compliance"]["eu_ai_act"] }) => {
  const [open, setOpen] = useState(true);
  return (
    <div className="mc-fw-panel">
      <button className="mc-fw-header" onClick={() => setOpen(v => !v)}>
        <div className="mc-fw-left">
          <Award size={16} />
          <div>
            <span className="mc-fw-name">EU AI Act</span>
            <span className="mc-fw-ver">Regulation 2024/1689</span>
          </div>
        </div>
        <div className="mc-fw-right">
          <span className="mc-fw-score-badge">{eu.score}%</span>
          <StatusPill status={eu.status} />
          {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        </div>
      </button>
      {open && (
        <div className="mc-fw-body">
          <p className="mc-fw-tier">Risk Tier: <strong>{eu.assigned_risk_tier}</strong></p>
          <table className="mc-table">
            <thead>
              <tr><th>Article</th><th>Result</th><th>Score</th><th>Evidence</th></tr>
            </thead>
            <tbody>
              {eu.article_audits.map((a, i) => (
                <tr key={i}>
                  <td className="mc-td-article">{a.article}</td>
                  <td><StatusPill status={a.status} /></td>
                  <td className="mc-td-score">{a.score}%</td>
                  <td className="mc-td-evidence">{a.evidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

const NistPanel = ({ nist }: { nist: CardData["regulatory_compliance"]["nist_ai_rmf"] }) => {
  const [open, setOpen] = useState(true);
  return (
    <div className="mc-fw-panel">
      <button className="mc-fw-header" onClick={() => setOpen(v => !v)}>
        <div className="mc-fw-left">
          <Shield size={16} />
          <div>
            <span className="mc-fw-name">NIST AI RMF</span>
            <span className="mc-fw-ver">AI RMF 1.0</span>
          </div>
        </div>
        <div className="mc-fw-right">
          <span className="mc-fw-score-badge">{nist.score}%</span>
          {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        </div>
      </button>
      {open && (
        <div className="mc-fw-body">
          <div className="mc-nist-grid">
            {Object.entries(nist.core_functions).map(([fn, data]) => (
              <div key={fn} className="mc-nist-card">
                <div className="mc-nist-fn">{fn}</div>
                <div className="mc-nist-status">
                  <StatusPill status={data.status} />
                </div>
                <div className="mc-nist-score">{data.score}%</div>
                <p className="mc-nist-evidence">{data.evidence}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const IsoPanel = ({ iso }: { iso: CardData["regulatory_compliance"]["iso_42001"] }) => {
  const [open, setOpen] = useState(true);
  return (
    <div className="mc-fw-panel">
      <button className="mc-fw-header" onClick={() => setOpen(v => !v)}>
        <div className="mc-fw-left">
          <Info size={16} />
          <div>
            <span className="mc-fw-name">ISO/IEC 42001</span>
            <span className="mc-fw-ver">AI Management System 2023</span>
          </div>
        </div>
        <div className="mc-fw-right">
          <span className="mc-fw-score-badge">{iso.score}%</span>
          {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        </div>
      </button>
      {open && (
        <div className="mc-fw-body">
          <div className="mc-iso-list">
            {iso.controls.map((ctrl, i) => (
              <div key={i} className="mc-iso-row">
                <div className="mc-iso-clause">§{ctrl.clause}</div>
                <div className="mc-iso-info">
                  <span className="mc-iso-name">{ctrl.name}</span>
                  <p className="mc-iso-details">{ctrl.details}</p>
                </div>
                <StatusPill status={ctrl.status} />
              </div>
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
    intended_use: "Production AI model diagnostics, drift tracking, retrieval evaluation, and adversarial security auditing.",
    out_of_scope_use: "Autonomous safety-critical decisions without human verification.",
    custom_notes: "",
  });

  const [card, setCard] = useState<CardData | null>(null);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInput = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
    setForm(f => ({ ...f, [e.target.name]: e.target.value }));

  const runAudit = useCallback(async () => {
    setLoading(true); setError(null); setCard(null);
    try {
      const res = await fetch(`${API}/api/v1/compliance/model-card`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail ?? `HTTP ${res.status}`); }
      setCard(await res.json());
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, [form]);

  const downloadMd = useCallback(async () => {
    setDownloading(true);
    try {
      const res = await fetch(`${API}/api/v1/compliance/model-card/markdown`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const cd = res.headers.get("content-disposition") ?? "";
      const m = cd.match(/filename=([^\s;]+)/);
      a.download = m?.[1] ?? "model-card.md";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setDownloading(false); }
  }, [form]);

  const ratingColor: Record<string, string> = { A: "#22c55e", B: "#f59e0b", C: "#ef4444" };

  return (
    <div className="mc-root">
      {/* Header */}
      <div className="mc-header">
        <div className="mc-header-icon"><Shield size={26} /></div>
        <div>
          <h1 className="mc-title">Model Card & Regulatory Compliance</h1>
          <p className="mc-subtitle">
            Generate an audited model card aligned with EU AI Act, NIST AI RMF, and ISO/IEC 42001
          </p>
        </div>
      </div>

      <div className="mc-layout">
        {/* ── Form ── */}
        <div className="mc-form-panel">
          <div className="mc-section-label"><ClipboardList size={14} />Model Metadata</div>

          {(["model_name","model_version","model_type","developer_or_team"] as (keyof FormState)[]).map(name => (
            <label key={name} className="mc-label">
              {name.replace(/_/g," ").replace(/\b\w/g,c=>c.toUpperCase())}
              <input className="mc-input" name={name} value={form[name]} onChange={handleInput} />
            </label>
          ))}

          <label className="mc-label">
            Intended Use
            <textarea className="mc-textarea" name="intended_use" value={form.intended_use} onChange={handleInput} rows={3} />
          </label>

          <label className="mc-label">
            Out-of-Scope Use
            <textarea className="mc-textarea" name="out_of_scope_use" value={form.out_of_scope_use} onChange={handleInput} rows={2} />
          </label>

          <label className="mc-label">
            Additional Notes
            <textarea className="mc-textarea" name="custom_notes" value={form.custom_notes} onChange={handleInput} rows={2}
              placeholder="Fairness constraints, data governance, training details…" />
          </label>

          <button className="mc-btn-generate" onClick={runAudit} disabled={loading}>
            {loading ? <><Loader2 size={15} className="mc-spin" /> Auditing…</> : <><Shield size={15} /> Run Compliance Audit</>}
          </button>

          {error && <div className="mc-error">{error}</div>}
        </div>

        {/* ── Results ── */}
        <div className="mc-results-panel">
          {!card && !loading && (
            <div className="mc-empty">
              <Shield size={52} className="mc-empty-icon" />
              <p>Configure model metadata and run the audit to generate a full compliance report.</p>
              <ul className="mc-framework-list">
                <li>🇪🇺 <strong>EU AI Act</strong> — Risk classification, transparency, human oversight</li>
                <li>🛡️ <strong>NIST AI RMF</strong> — Govern, Map, Measure, Manage</li>
                <li>📋 <strong>ISO/IEC 42001</strong> — AI Management System controls</li>
              </ul>
            </div>
          )}

          {loading && (
            <div className="mc-loading">
              <Loader2 size={44} className="mc-spin" />
              <p>Running regulatory compliance audit…</p>
            </div>
          )}

          {card && (
            <>
              {/* Overall badge */}
              <div className="mc-overall-card">
                <div className="mc-overall-left">
                  <ScoreRing score={card.metadata.overall_compliance_score} size={112} label={card.metadata.overall_compliance_status.replace(/_/g," ")} />
                  <div className="mc-overall-meta">
                    <h2 className="mc-overall-title">{card.model_overview.name}</h2>
                    <p className="mc-overall-sub">v{card.model_overview.version} · {card.model_overview.type}</p>
                    <p className="mc-overall-id">Report: {card.metadata.report_id} · {card.metadata.generated_at.slice(0,10)}</p>
                    <span className="mc-rating-badge" style={{ background: ratingColor[card.metadata.compliance_rating] ?? "#6366f1" }}>
                      Rating {card.metadata.compliance_rating}
                    </span>
                  </div>
                </div>
                <button className="mc-btn-dl" onClick={downloadMd} disabled={downloading}>
                  {downloading ? <Loader2 size={14} className="mc-spin" /> : <Download size={14} />}
                  Export .md
                </button>
              </div>

              {/* Framework audits */}
              <div className="mc-section-label" style={{ marginTop: "1rem" }}>
                <Award size={14} /> Framework Audits
              </div>
              <EuPanel eu={card.regulatory_compliance.eu_ai_act} />
              <NistPanel nist={card.regulatory_compliance.nist_ai_rmf} />
              <IsoPanel iso={card.regulatory_compliance.iso_42001} />

              {/* Mitigations */}
              <div className="mc-mitigations">
                <div className="mc-section-label"><FileText size={14} />Mitigations &amp; Safeguards</div>
                <ul>{card.mitigations_and_safeguards.map((m, i) => <li key={i}>{m}</li>)}</ul>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
