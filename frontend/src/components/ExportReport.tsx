
import { useCallback, useState } from "react";
import axios from "axios";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileText,
  Loader2,
  Printer,
  RefreshCw,
  XCircle,
} from "lucide-react";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

interface MetricSummary {
  average: number | null;
  latest_value: number | null;
  available_runs: number;
}

interface ReportData {
  report_title: string;
  generated_at: string;
  service_version: string;
  rag: {
    runs_included: number;
    finding_summary: {
      runs_with_findings: number;
      runs_without_findings: number;
      runs_with_missing_finding_data: number;
    };
    metrics: Record<string, MetricSummary>;
    recent_runs: Array<{
      run_id: string;
      query: string;
      created_at: string;
      summary?: Record<string, number>;
    }>;
  };
  alerts: {
    total: number;
    by_severity: Record<string, number>;
    recent: Array<{
      alert_id: string;
      feature: string;
      severity: string;
      title: string;
      message: string;
      effect_size: number;
      created_at: string;
    }>;
  };
}

// ─── HTML Report Generator ────────────────────────────────────────────────────

function buildHtmlReport(data: ReportData): string {
  const ts = new Date(data.generated_at).toLocaleString();
  const fmt = (v: number | null | undefined) =>
    v != null && Number.isFinite(v) ? (v * 100).toFixed(1) + "%" : "—";
  const fmtRaw = (v: number | null | undefined) =>
    v != null && Number.isFinite(v) ? v.toFixed(4) : "—";

  const sevColor: Record<string, string> = {
    high: "#f87171", medium: "#fbbf24", low: "#a78bfa", none: "#475569",
  };

  const metricRows = Object.entries(data.rag.metrics)
    .map(([key, m]) => `
      <tr>
        <td>${key.replaceAll("_", " ")}</td>
        <td>${fmt(m.average)}</td>
        <td>${fmt(m.latest_value)}</td>
        <td>${m.available_runs}</td>
      </tr>`)
    .join("");

  const ragRunRows = data.rag.recent_runs
    .slice(0, 15)
    .map((run) => {
      const s = run.summary ?? {};
      return `<tr>
        <td class="mono">${run.run_id.slice(0, 14)}…</td>
        <td>${(run.query ?? "").slice(0, 60)}${(run.query ?? "").length > 60 ? "…" : ""}</td>
        <td>${fmt(s["precision_at_k"])}</td>
        <td>${fmt(s["recall_at_k"])}</td>
        <td>${fmt(s["answer_relevance_score"])}</td>
        <td>${new Date(run.created_at).toLocaleDateString()}</td>
      </tr>`;
    })
    .join("");

  const alertRows = data.alerts.recent
    .slice(0, 15)
    .map((a) => `
      <tr>
        <td>${a.feature}</td>
        <td><span class="badge" style="color:${sevColor[a.severity] ?? "#94a3b8"}">${a.severity.toUpperCase()}</span></td>
        <td>${a.title}</td>
        <td>${fmtRaw(a.effect_size)}</td>
        <td>${new Date(a.created_at).toLocaleDateString()}</td>
      </tr>`)
    .join("");

  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>${data.report_title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Inter',sans-serif;background:#f8fafc;color:#1e293b;font-size:13px;line-height:1.6}
  .page{max-width:900px;margin:0 auto;padding:40px 32px}
  h1{font-size:22px;font-weight:700;color:#0f172a;margin-bottom:4px}
  h2{font-size:14px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#475569;margin:32px 0 12px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}
  h3{font-size:13px;font-weight:600;color:#334155;margin-bottom:8px}
  .meta{font-size:11px;color:#94a3b8;margin-bottom:32px}
  .kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
  .kpi{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px}
  .kpi-label{font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#94a3b8;margin-bottom:6px}
  .kpi-val{font-size:22px;font-weight:700;color:#0f172a}
  .kpi-sub{font-size:10px;color:#94a3b8;margin-top:2px}
  table{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.06);margin-bottom:24px}
  th{background:#f1f5f9;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#64748b;padding:10px 14px;text-align:left;font-weight:600}
  td{padding:9px 14px;border-bottom:1px solid #f1f5f9;color:#334155;font-size:12px}
  tr:last-child td{border-bottom:none}
  tr:hover td{background:#f8fafc}
  .badge{font-size:10px;font-weight:700;letter-spacing:0.05em}
  .mono{font-family:monospace;font-size:11px;color:#64748b}
  .sev-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
  .sev-card{border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;background:#fff}
  .sev-num{font-size:20px;font-weight:700}
  .sev-label{font-size:10px;text-transform:uppercase;letter-spacing:0.05em;color:#94a3b8;margin-top:2px}
  .footer{border-top:1px solid #e2e8f0;padding-top:20px;margin-top:40px;font-size:10px;color:#94a3b8;display:flex;justify-content:space-between}
  .disclaimer{background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:12px 16px;font-size:11px;color:#92400e;margin-bottom:24px;line-height:1.55}
  @media print{body{background:#fff}.page{padding:20px}button{display:none!important}}
</style>
</head>
<body>
<div class="page">
  <h1>${data.report_title}</h1>
  <p class="meta">Generated: ${ts} &nbsp;·&nbsp; Service v${data.service_version} &nbsp;·&nbsp; ModelAutopsy AI</p>

  <div class="disclaimer">
    ⚠ <strong>Diagnostic use only.</strong> Metrics are computed using embedding similarity heuristics and statistical thresholds.
    They are not a substitute for human evaluation or calibrated production monitoring.
    Findings require expert review before acting on any recommendation.
  </div>

  <h2>RAG Evaluation Summary</h2>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Runs evaluated</div>
      <div class="kpi-val">${data.rag.runs_included}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Runs with findings</div>
      <div class="kpi-val" style="color:#f59e0b">${data.rag.finding_summary.runs_with_findings}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Clean runs</div>
      <div class="kpi-val" style="color:#10b981">${data.rag.finding_summary.runs_without_findings}</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Missing data</div>
      <div class="kpi-val" style="color:#94a3b8">${data.rag.finding_summary.runs_with_missing_finding_data}</div>
      <div class="kpi-sub">runs</div>
    </div>
  </div>

  <h2>Metric Averages</h2>
  <table>
    <thead><tr><th>Metric</th><th>Average</th><th>Latest</th><th>Runs with data</th></tr></thead>
    <tbody>${metricRows}</tbody>
  </table>

  <h2>Recent Evaluation Runs</h2>
  <table>
    <thead><tr><th>Run ID</th><th>Query</th><th>Precision@K</th><th>Recall@K</th><th>Answer Relevance</th><th>Created</th></tr></thead>
    <tbody>${ragRunRows || '<tr><td colspan="6" style="text-align:center;color:#94a3b8">No runs recorded yet.</td></tr>'}</tbody>
  </table>

  <h2>Drift Alert Summary</h2>
  <div class="sev-grid" style="margin-bottom:20px">
    ${Object.entries(data.alerts.by_severity).map(([sev, count]) => `
    <div class="sev-card">
      <div class="sev-num" style="color:${sevColor[sev] ?? "#94a3b8"}">${count}</div>
      <div class="sev-label">${sev} severity</div>
    </div>`).join("")}
  </div>

  <h2>Recent Drift Alerts</h2>
  <table>
    <thead><tr><th>Feature</th><th>Severity</th><th>Title</th><th>Effect size</th><th>Date</th></tr></thead>
    <tbody>${alertRows || '<tr><td colspan="5" style="text-align:center;color:#94a3b8">No alerts recorded.</td></tr>'}</tbody>
  </table>

  <div class="footer">
    <span>ModelAutopsy AI · Diagnostic Audit Report</span>
    <span>Generated ${ts}</span>
  </div>
</div>
</body>
</html>`;
}

// ─── Stat Card ────────────────────────────────────────────────────────────────

function StatCard({ label, value, color = "#e2e8f0" }: { label: string; value: string | number; color?: string }) {
  return (
    <div style={{
      background: "rgba(0,0,0,0.28)",
      border: "1px solid rgba(255,255,255,0.07)",
      borderRadius: "0.6rem",
      padding: "0.7rem 0.85rem",
    }}>
      <p style={{ fontSize: "0.64rem", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.25rem" }}>{label}</p>
      <p style={{ fontSize: "1.15rem", fontWeight: 700, color }}>{value}</p>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function ExportReport() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [report, setReport] = useState<ReportData | null>(null);

  const fetchReport = useCallback(async () => {
    setLoading(true);
    setError("");
    setReport(null);
    try {
      const res = await axios.get<ReportData>(`${API_BASE}/api/v1/reports/generate?rag_limit=20&alert_limit=50`);
      setReport(res.data);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(typeof detail === "string" ? detail : `Report fetch failed (${err.response?.status ?? "network error"}).`);
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  function downloadHtml() {
    if (!report) return;
    const html = buildHtmlReport(report);
    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `modelautopsy-report-${new Date().toISOString().slice(0, 10)}.html`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function printReport() {
    if (!report) return;
    const html = buildHtmlReport(report);
    const win = window.open("", "_blank");
    if (!win) return;
    win.document.write(html);
    win.document.close();
    win.focus();
    setTimeout(() => win.print(), 400);
  }

  const fmt = (v: number | null | undefined) =>
    v != null && Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—";

  const SEV_COLOR: Record<string, string> = { high: "#f87171", medium: "#fbbf24", low: "#a78bfa", none: "#475569" };

  return (
    <div style={{
      borderRadius: "1rem",
      border: "1px solid rgba(100,116,139,0.2)",
      background: "#0d1424",
      padding: "1.35rem 1.4rem",
    }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "1rem", marginBottom: "1.1rem", flexWrap: "wrap" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.35rem" }}>
            <FileText size={16} style={{ color: "#818cf8" }} />
            <h3 style={{ fontSize: "0.95rem", fontWeight: 600, color: "#f1f5f9" }}>
              Export Diagnostic Report
            </h3>
          </div>
          <p style={{ fontSize: "0.78rem", color: "#64748b", lineHeight: 1.5 }}>
            Generate a full audit report covering RAG evaluation metrics, run history, and drift alerts.
            Download as a self-contained HTML file or print directly to PDF.
          </p>
        </div>

        <button
          type="button"
          onClick={fetchReport}
          disabled={loading}
          style={{
            display: "inline-flex", alignItems: "center", gap: "0.45rem",
            padding: "0.55rem 1.1rem", borderRadius: "0.55rem",
            fontSize: "0.8rem", fontWeight: 600, cursor: loading ? "not-allowed" : "pointer",
            background: "linear-gradient(135deg,#6366f1 0%,#818cf8 100%)",
            color: "#fff", border: "none", opacity: loading ? 0.6 : 1,
            flexShrink: 0, whiteSpace: "nowrap",
          }}
        >
          {loading
            ? <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
            : <RefreshCw size={14} />}
          {loading ? "Generating…" : report ? "Regenerate" : "Generate Report"}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start", fontSize: "0.78rem", color: "#fca5a5", padding: "0.55rem 0.7rem", borderRadius: "0.5rem", background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.25)", marginBottom: "1rem" }}>
          <XCircle size={15} style={{ flexShrink: 0, marginTop: "0.05rem" }} />
          {error}
        </div>
      )}

      {/* Preview */}
      {report && (
        <div>
          {/* Timestamp + download bar */}
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "space-between",
            flexWrap: "wrap", gap: "0.6rem",
            background: "rgba(99,102,241,0.07)", border: "1px solid rgba(99,102,241,0.2)",
            borderRadius: "0.65rem", padding: "0.65rem 0.9rem", marginBottom: "1.1rem",
          }}>
            <div>
              <p style={{ fontSize: "0.8rem", fontWeight: 600, color: "#a5b4fc" }}>{report.report_title}</p>
              <p style={{ fontSize: "0.7rem", color: "#64748b" }}>
                Generated: {new Date(report.generated_at).toLocaleString()} · v{report.service_version}
              </p>
            </div>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button type="button" onClick={downloadHtml}
                style={{
                  display: "inline-flex", alignItems: "center", gap: "0.4rem",
                  padding: "0.45rem 0.9rem", borderRadius: "0.5rem",
                  fontSize: "0.76rem", fontWeight: 600, cursor: "pointer",
                  background: "rgba(99,102,241,0.15)", border: "1px solid rgba(99,102,241,0.35)",
                  color: "#a5b4fc",
                }}>
                <Download size={13} /> Download HTML
              </button>
              <button type="button" onClick={printReport}
                style={{
                  display: "inline-flex", alignItems: "center", gap: "0.4rem",
                  padding: "0.45rem 0.9rem", borderRadius: "0.5rem",
                  fontSize: "0.76rem", fontWeight: 600, cursor: "pointer",
                  background: "rgba(52,211,153,0.1)", border: "1px solid rgba(52,211,153,0.25)",
                  color: "#34d399",
                }}>
                <Printer size={13} /> Print / PDF
              </button>
            </div>
          </div>

          {/* RAG KPIs */}
          <p style={{ fontSize: "0.68rem", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.55rem" }}>
            RAG Evaluation Overview
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(130px,1fr))", gap: "0.55rem", marginBottom: "1.1rem" }}>
            <StatCard label="Runs evaluated" value={report.rag.runs_included} />
            <StatCard label="With findings" value={report.rag.finding_summary.runs_with_findings} color="#fbbf24" />
            <StatCard label="Clean runs" value={report.rag.finding_summary.runs_without_findings} color="#34d399" />
            <StatCard label="Total alerts" value={report.alerts.total} color={report.alerts.total > 0 ? "#f87171" : "#34d399"} />
          </div>

          {/* Metric table preview */}
          <p style={{ fontSize: "0.68rem", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
            Metric Averages
          </p>
          <div style={{ borderRadius: "0.65rem", border: "1px solid rgba(255,255,255,0.07)", overflow: "hidden", marginBottom: "1.1rem" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "rgba(0,0,0,0.3)" }}>
                  {["Metric", "Average", "Latest", "Runs"].map((h) => (
                    <th key={h} style={{ padding: "0.45rem 0.75rem", fontSize: "0.63rem", color: "#475569", textTransform: "uppercase", letterSpacing: "0.05em", textAlign: "left", fontWeight: 600 }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(report.rag.metrics).map(([key, m]) => (
                  <tr key={key} style={{ borderTop: "1px solid rgba(255,255,255,0.04)" }}>
                    <td style={{ padding: "0.42rem 0.75rem", fontSize: "0.75rem", color: "#94a3b8" }}>{key.replaceAll("_", " ")}</td>
                    <td style={{ padding: "0.42rem 0.75rem", fontSize: "0.75rem", fontWeight: 600, color: (m.average ?? 0) >= 0.5 ? "#34d399" : "#fbbf24" }}>{fmt(m.average)}</td>
                    <td style={{ padding: "0.42rem 0.75rem", fontSize: "0.75rem", color: "#94a3b8" }}>{fmt(m.latest_value)}</td>
                    <td style={{ padding: "0.42rem 0.75rem", fontSize: "0.75rem", color: "#64748b" }}>{m.available_runs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Alert severity chips */}
          <p style={{ fontSize: "0.68rem", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
            Drift Alerts by Severity
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.55rem", marginBottom: "1.1rem" }}>
            {Object.entries(report.alerts.by_severity).map(([sev, count]) => (
              <div key={sev} style={{
                display: "flex", alignItems: "center", gap: "0.45rem",
                padding: "0.4rem 0.8rem", borderRadius: "0.5rem",
                border: `1px solid ${SEV_COLOR[sev]}33`,
                background: `${SEV_COLOR[sev]}0d`,
              }}>
                {sev === "high" ? <XCircle size={13} style={{ color: SEV_COLOR[sev] }} />
                  : sev === "none" ? <CheckCircle2 size={13} style={{ color: SEV_COLOR[sev] }} />
                    : <AlertTriangle size={13} style={{ color: SEV_COLOR[sev] }} />}
                <span style={{ fontSize: "0.75rem", fontWeight: 700, color: SEV_COLOR[sev] }}>{count}</span>
                <span style={{ fontSize: "0.72rem", color: "#64748b", textTransform: "capitalize" }}>{sev}</span>
              </div>
            ))}
          </div>

          {/* Recent runs preview */}
          {report.rag.recent_runs.length > 0 && (
            <>
              <p style={{ fontSize: "0.68rem", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
                Recent RAG Runs (preview)
              </p>
              <div style={{ borderRadius: "0.65rem", border: "1px solid rgba(255,255,255,0.07)", overflow: "hidden", overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 560 }}>
                  <thead>
                    <tr style={{ background: "rgba(0,0,0,0.3)" }}>
                      {["Run ID", "Query", "Precision@K", "Answer Rel.", "Date"].map((h) => (
                        <th key={h} style={{ padding: "0.45rem 0.75rem", fontSize: "0.63rem", color: "#475569", textTransform: "uppercase", letterSpacing: "0.05em", textAlign: "left", fontWeight: 600, whiteSpace: "nowrap" }}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {report.rag.recent_runs.slice(0, 8).map((run) => (
                      <tr key={run.run_id} style={{ borderTop: "1px solid rgba(255,255,255,0.04)" }}>
                        <td style={{ padding: "0.42rem 0.75rem", fontSize: "0.7rem", fontFamily: "monospace", color: "#22d3ee", whiteSpace: "nowrap" }}>
                          {run.run_id.slice(0, 14)}…
                        </td>
                        <td style={{ padding: "0.42rem 0.75rem", fontSize: "0.72rem", color: "#94a3b8", maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          {run.query}
                        </td>
                        <td style={{ padding: "0.42rem 0.75rem", fontSize: "0.72rem", color: "#e2e8f0" }}>
                          {fmt(run.summary?.["precision_at_k"])}
                        </td>
                        <td style={{ padding: "0.42rem 0.75rem", fontSize: "0.72rem", color: "#e2e8f0" }}>
                          {fmt(run.summary?.["answer_relevance_score"])}
                        </td>
                        <td style={{ padding: "0.42rem 0.75rem", fontSize: "0.7rem", color: "#64748b", whiteSpace: "nowrap" }}>
                          {new Date(run.created_at).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}

      {!report && !loading && !error && (
        <div style={{ textAlign: "center", padding: "2rem 0", color: "#475569" }}>
          <FileText size={32} style={{ margin: "0 auto 0.75rem", opacity: 0.4 }} />
          <p style={{ fontSize: "0.82rem" }}>Click <strong style={{ color: "#a5b4fc" }}>Generate Report</strong> to collect diagnostic data and build the audit report.</p>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
