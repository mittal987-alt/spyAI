
import { useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileUp,
  Loader2,
  Upload,
  XCircle,
} from "lucide-react";
import axios from "axios";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// ─── Types ───────────────────────────────────────────────────────────────────

interface PsiBin {
  bin_index: number;
  label: string;
  expected_pct: number;
  actual_pct: number;
  sub_psi: number;
}

interface FeatureResult {
  feature: string;
  ks_statistic: number;
  p_value: number;
  adjusted_p_value: number;
  effect_size: number;
  statistically_significant: boolean;
  drift_detected: boolean;
  severity: "none" | "low" | "medium" | "high";
  psi?: {
    psi_score: number;
    risk_level: "stable" | "moderate" | "significant";
    bins: PsiBin[];
  };
  wasserstein_distance?: number;
  summary_stats?: {
    reference_mean: number;
    current_mean: number;
    reference_std: number;
    current_std: number;
  };
}

interface DriftUploadResult {
  baseline_rows: number;
  current_rows: number;
  features_tested: number;
  features_drifted: number;
  alpha: number;
  effect_threshold: number;
  results: FeatureResult[];
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const SEVERITY_COLOR: Record<string, string> = {
  high:   "#f87171",
  medium: "#fbbf24",
  low:    "#a78bfa",
  none:   "#475569",
};

const SEVERITY_BG: Record<string, string> = {
  high:   "rgba(248,113,113,0.08)",
  medium: "rgba(251,191,36,0.07)",
  low:    "rgba(167,139,250,0.07)",
  none:   "rgba(71,85,105,0.07)",
};

function SeverityBadge({ severity }: { severity: string }) {
  const col = SEVERITY_COLOR[severity] ?? "#475569";
  return (
    <span
      style={{
        fontSize: "0.62rem",
        fontWeight: 700,
        textTransform: "uppercase",
        letterSpacing: "0.06em",
        padding: "0.1rem 0.5rem",
        borderRadius: "999px",
        border: `1px solid ${col}55`,
        background: `${col}18`,
        color: col,
      }}
    >
      {severity}
    </span>
  );
}

function Bar({ value }: { value: number }) {
  const pct = Math.min(100, Math.round(value * 100));
  const col =
    value >= 0.3 ? "#f87171" : value >= 0.1 ? "#fbbf24" : "#34d399";
  return (
    <div
      style={{
        height: 5,
        borderRadius: 4,
        background: "rgba(255,255,255,0.07)",
        overflow: "hidden",
        width: "100%",
      }}
    >
      <div
        style={{
          height: "100%",
          width: `${pct}%`,
          borderRadius: 4,
          background: col,
          transition: "width 0.4s ease",
        }}
      />
    </div>
  );
}

// ─── File Drop Zone ───────────────────────────────────────────────────────────

function DropZone({
  label,
  file,
  onFile,
  id,
}: {
  label: string;
  file: File | null;
  onFile: (f: File) => void;
  id: string;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  }

  return (
    <div>
      <label
        htmlFor={id}
        style={{
          display: "block",
          fontSize: "0.78rem",
          color: "#94a3b8",
          marginBottom: "0.45rem",
          fontWeight: 500,
        }}
      >
        {label}
      </label>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${dragging ? "#22d3ee" : file ? "rgba(52,211,153,0.5)" : "rgba(100,116,139,0.4)"}`,
          borderRadius: "0.75rem",
          padding: "1.1rem",
          textAlign: "center",
          cursor: "pointer",
          background: dragging
            ? "rgba(34,211,238,0.04)"
            : file
              ? "rgba(52,211,153,0.04)"
              : "rgba(0,0,0,0.2)",
          transition: "all 0.15s ease",
        }}
      >
        <input
          id={id}
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
          }}
        />
        {file ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "0.5rem" }}>
            <CheckCircle2 size={16} style={{ color: "#34d399" }} />
            <span style={{ fontSize: "0.78rem", color: "#e2e8f0", fontFamily: "monospace" }}>{file.name}</span>
            <span style={{ fontSize: "0.68rem", color: "#64748b" }}>({(file.size / 1024).toFixed(1)} KB)</span>
          </div>
        ) : (
          <div>
            <FileUp size={20} style={{ color: "#475569", margin: "0 auto 0.4rem" }} />
            <p style={{ fontSize: "0.78rem", color: "#64748b" }}>
              Drop a <span style={{ color: "#94a3b8" }}>.csv</span> file or click to browse
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Feature Row ─────────────────────────────────────────────────────────────

function FeatureRow({ result }: { result: FeatureResult }) {
  const [open, setOpen] = useState(false);

  const psi = result.psi;
  const psiColor =
    psi?.risk_level === "significant"
      ? "#f87171"
      : psi?.risk_level === "moderate"
      ? "#fbbf24"
      : "#34d399";

  return (
    <div
      style={{
        borderRadius: "0.55rem",
        border: `1px solid ${result.drift_detected ? `${SEVERITY_COLOR[result.severity]}35` : "rgba(255,255,255,0.05)"}`,
        background: result.drift_detected ? SEVERITY_BG[result.severity] : "rgba(0,0,0,0.15)",
        overflow: "hidden",
        marginBottom: "0.35rem",
      }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: "0.6rem",
          padding: "0.55rem 0.75rem",
          background: "transparent",
          border: "none",
          cursor: "pointer",
          textAlign: "left",
        }}
      >
        {/* Feature name */}
        <span style={{ flex: 1, fontSize: "0.8rem", fontFamily: "monospace", color: "#e2e8f0", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {result.feature}
        </span>

        {/* KS bar */}
        <div style={{ width: 70, flexShrink: 0 }}>
          <Bar value={result.effect_size} />
        </div>

        {/* Effect size */}
        <span style={{ width: 42, fontSize: "0.72rem", color: "#94a3b8", textAlign: "right", flexShrink: 0 }}>
          {(result.effect_size * 100).toFixed(1)}%
        </span>

        {/* PSI Badge */}
        {psi && (
          <span
            style={{
              fontSize: "0.6rem",
              fontWeight: 600,
              padding: "0.1rem 0.4rem",
              borderRadius: "4px",
              background: `${psiColor}15`,
              border: `1px solid ${psiColor}40`,
              color: psiColor,
              flexShrink: 0,
            }}
            title={`Population Stability Index: ${psi.psi_score} (${psi.risk_level.toUpperCase()})`}
          >
            PSI {psi.psi_score.toFixed(3)}
          </span>
        )}

        <SeverityBadge severity={result.severity} />

        {open
          ? <ChevronUp size={13} style={{ color: "#64748b", flexShrink: 0 }} />
          : <ChevronDown size={13} style={{ color: "#64748b", flexShrink: 0 }} />}
      </button>

      {open && (
        <div
          style={{
            padding: "0.65rem 0.8rem 0.8rem",
            borderTop: "1px solid rgba(255,255,255,0.05)",
          }}
        >
          {/* Metrics summary grid */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(135px, 1fr))",
              gap: "0.45rem",
              marginBottom: "0.75rem",
            }}
          >
            {[
              ["KS Statistic", result.ks_statistic.toFixed(4)],
              ["Adjusted p-val", result.adjusted_p_value.toExponential(3)],
              ["PSI Score", psi ? `${psi.psi_score.toFixed(4)} (${psi.risk_level})` : "—"],
              ["Wasserstein Dist", result.wasserstein_distance != null ? result.wasserstein_distance.toFixed(4) : "—"],
              ["Baseline Mean", result.summary_stats ? result.summary_stats.reference_mean.toFixed(3) : "—"],
              ["Current Mean", result.summary_stats ? result.summary_stats.current_mean.toFixed(3) : "—"],
            ].map(([k, v]) => (
              <div key={k} style={{ background: "rgba(0,0,0,0.25)", borderRadius: "0.4rem", padding: "0.35rem 0.5rem" }}>
                <p style={{ fontSize: "0.62rem", color: "#64748b", marginBottom: "0.15rem", textTransform: "uppercase", letterSpacing: "0.04em" }}>{k}</p>
                <p style={{ fontSize: "0.78rem", fontWeight: 600, color: "#e2e8f0", fontFamily: "monospace" }}>{v}</p>
              </div>
            ))}
          </div>

          {/* Quantile Bucket Distribution (PSI Bins) */}
          {psi && psi.bins.length > 0 && (
            <div style={{ background: "rgba(0,0,0,0.2)", borderRadius: "0.5rem", padding: "0.6rem 0.7rem", border: "1px solid rgba(255,255,255,0.04)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.45rem" }}>
                <p style={{ fontSize: "0.68rem", fontWeight: 600, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Quantile Distribution Shift (Expected vs Actual %)
                </p>
                <span style={{ fontSize: "0.65rem", color: "#64748b" }}>
                  <span style={{ color: "#38bdf8" }}>■ Baseline</span> &nbsp;vs&nbsp; <span style={{ color: "#f43f5e" }}>■ Current</span>
                </span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "0.3rem" }}>
                {psi.bins.slice(0, 8).map((b) => (
                  <div key={b.bin_index} style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.68rem" }}>
                    <span style={{ width: 85, fontFamily: "monospace", color: "#64748b", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {b.label}
                    </span>
                    <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "0.15rem" }}>
                      {/* Expected (Baseline) bar */}
                      <div style={{ height: 4, borderRadius: 2, background: "rgba(255,255,255,0.05)", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${Math.min(100, b.expected_pct * 3)}%`, background: "#38bdf8" }} />
                      </div>
                      {/* Actual (Current) bar */}
                      <div style={{ height: 4, borderRadius: 2, background: "rgba(255,255,255,0.05)", overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${Math.min(100, b.actual_pct * 3)}%`, background: "#f43f5e" }} />
                      </div>
                    </div>
                    <span style={{ width: 68, textAlign: "right", fontFamily: "monospace", color: "#94a3b8" }}>
                      {b.expected_pct}% / {b.actual_pct}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function CSVDriftUpload() {
  const [baselineFile, setBaselineFile] = useState<File | null>(null);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [alpha, setAlpha] = useState(0.05);
  const [effectThreshold, setEffectThreshold] = useState(0.1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<DriftUploadResult | null>(null);
  const [filter, setFilter] = useState<"all" | "drifted">("all");

  async function runAnalysis(e: React.FormEvent) {
    e.preventDefault();
    if (!baselineFile || !currentFile) return;

    setError("");
    setResult(null);
    setLoading(true);

    const form = new FormData();
    form.append("baseline", baselineFile);
    form.append("current", currentFile);

    try {
      const res = await axios.post<DriftUploadResult>(
        `${API_BASE}/api/v1/drift/upload?alpha=${alpha}&effect_threshold=${effectThreshold}`,
        form,
        { headers: { "Content-Type": "multipart/form-data" } },
      );
      setResult(res.data);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(typeof detail === "string" ? detail : `Upload failed (${err.response?.status ?? "network error"}).`);
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  }

  const displayed =
    result?.results.filter((r) => filter === "all" || r.drift_detected) ?? [];

  const driftedCount = result?.features_drifted ?? 0;
  const totalCount  = result?.features_tested ?? 0;

  return (
    <div
      style={{
        borderRadius: "1rem",
        border: "1px solid rgba(100,116,139,0.2)",
        background: "#0d1424",
        padding: "1.35rem 1.4rem",
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: "1.1rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.35rem" }}>
          <Upload size={16} style={{ color: "#38bdf8" }} />
          <h3 style={{ fontSize: "0.95rem", fontWeight: 600, color: "#f1f5f9" }}>
            CSV Dataset Drift Upload
          </h3>
        </div>
        <p style={{ fontSize: "0.78rem", color: "#64748b", lineHeight: 1.55 }}>
          Upload your own baseline and current CSV datasets to detect feature distribution drift
          using KS tests with Benjamini-Hochberg correction.
        </p>
      </div>

      <form onSubmit={runAnalysis} style={{ display: "flex", flexDirection: "column", gap: "0.9rem" }}>
        {/* Drop zones */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
          <DropZone id="baseline-csv" label="Baseline CSV" file={baselineFile} onFile={setBaselineFile} />
          <DropZone id="current-csv"  label="Current CSV"  file={currentFile}  onFile={setCurrentFile}  />
        </div>

        {/* Parameters */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
          <div>
            <label style={{ display: "block", fontSize: "0.75rem", color: "#94a3b8", marginBottom: "0.35rem" }}>
              Significance level (α)
            </label>
            <input
              type="number"
              min={0.001}
              max={0.2}
              step={0.001}
              value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))}
              style={{
                width: "100%",
                borderRadius: "0.5rem",
                border: "1px solid rgba(100,116,139,0.35)",
                background: "rgba(0,0,0,0.3)",
                color: "#e2e8f0",
                padding: "0.5rem 0.7rem",
                fontSize: "0.82rem",
                outline: "none",
              }}
            />
          </div>
          <div>
            <label style={{ display: "block", fontSize: "0.75rem", color: "#94a3b8", marginBottom: "0.35rem" }}>
              Min effect size threshold
            </label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={effectThreshold}
              onChange={(e) => setEffectThreshold(Number(e.target.value))}
              style={{
                width: "100%",
                borderRadius: "0.5rem",
                border: "1px solid rgba(100,116,139,0.35)",
                background: "rgba(0,0,0,0.3)",
                color: "#e2e8f0",
                padding: "0.5rem 0.7rem",
                fontSize: "0.82rem",
                outline: "none",
              }}
            />
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start", fontSize: "0.78rem", color: "#fca5a5", padding: "0.55rem 0.7rem", borderRadius: "0.5rem", background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.25)" }}>
            <XCircle size={15} style={{ flexShrink: 0, marginTop: "0.05rem" }} />
            {error}
          </div>
        )}

        {/* Submit */}
        <button
          type="submit"
          disabled={loading || !baselineFile || !currentFile}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.5rem",
            padding: "0.6rem 1.2rem",
            borderRadius: "0.55rem",
            fontSize: "0.82rem",
            fontWeight: 600,
            cursor: loading || !baselineFile || !currentFile ? "not-allowed" : "pointer",
            opacity: !baselineFile || !currentFile ? 0.45 : 1,
            background: "linear-gradient(135deg, #0ea5e9 0%, #38bdf8 100%)",
            color: "#0f172a",
            border: "none",
            alignSelf: "flex-start",
            transition: "opacity 0.15s",
          }}
        >
          {loading ? <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} /> : <Upload size={14} />}
          {loading ? "Analysing…" : "Detect Drift"}
        </button>
      </form>

      {/* Results */}
      {result && (
        <div style={{ marginTop: "1.5rem" }}>
          {/* Summary bar */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))",
              gap: "0.55rem",
              marginBottom: "1.1rem",
            }}
          >
            {[
              ["Baseline rows", result.baseline_rows],
              ["Current rows",  result.current_rows],
              ["Features tested", result.features_tested],
              ["Features drifted", result.features_drifted],
            ].map(([label, val]) => (
              <div
                key={String(label)}
                style={{
                  background: "rgba(0,0,0,0.3)",
                  borderRadius: "0.55rem",
                  border: "1px solid rgba(255,255,255,0.06)",
                  padding: "0.55rem 0.7rem",
                }}
              >
                <p style={{ fontSize: "0.64rem", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "0.2rem" }}>{label}</p>
                <p
                  style={{
                    fontSize: "1.15rem",
                    fontWeight: 700,
                    color:
                      label === "Features drifted"
                        ? Number(val) === 0
                          ? "#34d399"
                          : "#f87171"
                        : "#e2e8f0",
                  }}
                >
                  {val}
                </p>
              </div>
            ))}
          </div>

          {/* Alert bar */}
          {driftedCount > 0 ? (
            <div style={{ display: "flex", alignItems: "flex-start", gap: "0.5rem", fontSize: "0.78rem", color: "#fcd34d", padding: "0.55rem 0.7rem", borderRadius: "0.5rem", background: "rgba(251,191,36,0.07)", border: "1px solid rgba(251,191,36,0.2)", marginBottom: "0.9rem" }}>
              <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: "0.05rem" }} />
              {driftedCount} of {totalCount} feature{totalCount !== 1 ? "s" : ""} show statistically significant drift. Investigate data pipelines and feature sources before relying on model predictions.
            </div>
          ) : (
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.78rem", color: "#6ee7b7", padding: "0.5rem 0.7rem", borderRadius: "0.5rem", background: "rgba(52,211,153,0.06)", border: "1px solid rgba(52,211,153,0.2)", marginBottom: "0.9rem" }}>
              <CheckCircle2 size={14} />
              No significant drift detected across all {totalCount} feature{totalCount !== 1 ? "s" : ""}.
            </div>
          )}

          {/* Filter tabs */}
          <div style={{ display: "flex", gap: "0.4rem", marginBottom: "0.65rem" }}>
            {(["all", "drifted"] as const).map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setFilter(f)}
                style={{
                  padding: "0.28rem 0.75rem",
                  borderRadius: "999px",
                  fontSize: "0.73rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  border: "1px solid",
                  borderColor: filter === f ? "#38bdf8" : "rgba(100,116,139,0.3)",
                  background: filter === f ? "rgba(56,189,248,0.12)" : "transparent",
                  color: filter === f ? "#38bdf8" : "#64748b",
                  transition: "all 0.12s",
                }}
              >
                {f === "all" ? `All (${totalCount})` : `Drifted (${driftedCount})`}
              </button>
            ))}
          </div>

          {/* Table header */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", padding: "0 0.75rem", marginBottom: "0.35rem" }}>
            <span style={{ flex: 1, fontSize: "0.63rem", color: "#475569", textTransform: "uppercase", letterSpacing: "0.05em" }}>Feature</span>
            <span style={{ width: 80, fontSize: "0.63rem", color: "#475569", textTransform: "uppercase", letterSpacing: "0.05em" }}>KS</span>
            <span style={{ width: 42, fontSize: "0.63rem", color: "#475569", textTransform: "uppercase", textAlign: "right", letterSpacing: "0.05em" }}>Effect</span>
            <span style={{ width: 58, fontSize: "0.63rem", color: "#475569", textTransform: "uppercase", textAlign: "center", letterSpacing: "0.05em" }}>Severity</span>
            <span style={{ width: 13 }} />
          </div>

          {/* Feature rows */}
          <div>
            {displayed.length === 0 ? (
              <p style={{ fontSize: "0.78rem", color: "#475569", textAlign: "center", padding: "1rem 0" }}>
                {filter === "drifted" ? "No drifted features." : "No results."}
              </p>
            ) : (
              displayed.map((r) => <FeatureRow key={r.feature} result={r} />)
            )}
          </div>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
