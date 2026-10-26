
import { useCallback, useRef, useState } from "react";
import axios from "axios";
import {
  Activity,
  AlertTriangle,
  Loader2,
  Play,
  Scan,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

interface PlotPoint {
  x: number;
  y: number;
  cluster: number;
  source: "reference" | "current";
  text: string;
}

interface ProjectionResult {
  points: PlotPoint[];
  n_clusters: number;
  pca_variance_explained: number[];
  reference_count: number;
  current_count: number;
}

interface MonitorResult {
  status: string;
  semantic_drift: {
    drift_detected: boolean;
    effect_size: number;
    p_value: number;
    permutation_p_value: number;
  };
  topic_shift: {
    clusters: Array<{
      cluster_id: number;
      reference_proportion: number;
      current_proportion: number;
      proportion_change: number;
    }>;
  };
}

// ─── Palette ─────────────────────────────────────────────────────────────────

const CLUSTER_COLORS = [
  "#22d3ee", "#a78bfa", "#34d399", "#fb923c",
  "#f472b6", "#facc15", "#60a5fa", "#f87171",
  "#4ade80", "#c084fc",
];

function clusterColor(id: number) {
  return CLUSTER_COLORS[id % CLUSTER_COLORS.length];
}

// ─── SVG Scatter Plot ─────────────────────────────────────────────────────────

const PAD = 32;

function ScatterPlot({
  points,
  nClusters,
  variance,
}: {
  points: PlotPoint[];
  nClusters: number;
  variance: number[];
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number; y: number; point: PlotPoint;
  } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef<{ mx: number; my: number; px: number; py: number } | null>(null);
  const [activeCluster, setActiveCluster] = useState<number | null>(null);

  const W = 540, H = 380;
  const innerW = W - PAD * 2, innerH = H - PAD * 2;

  // Data range
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  function toSvg(px: number, py: number) {
    const sx = PAD + ((px - xMin) / xRange) * innerW;
    const sy = PAD + (1 - (py - yMin) / yRange) * innerH;
    return {
      x: sx * zoom + pan.x,
      y: sy * zoom + pan.y,
    };
  }

  function onMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    if (dragging && dragStart.current) {
      const dx = e.clientX - dragStart.current.mx;
      const dy = e.clientY - dragStart.current.my;
      setPan({ x: dragStart.current.px + dx, y: dragStart.current.py + dy });
    }
  }

  function onMouseDown(e: React.MouseEvent<SVGSVGElement>) {
    setDragging(true);
    dragStart.current = { mx: e.clientX, my: e.clientY, px: pan.x, py: pan.y };
  }

  const resetView = useCallback(() => { setZoom(1); setPan({ x: 0, y: 0 }); }, []);

  const visiblePoints = points.filter(
    (p) => activeCluster === null || p.cluster === activeCluster,
  );

  const clusterIds = Array.from({ length: nClusters }, (_, i) => i);

  return (
    <div style={{ userSelect: "none" }}>
      {/* Toolbar */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.6rem", flexWrap: "wrap" }}>
        <span style={{ fontSize: "0.72rem", color: "#64748b" }}>
          PCA1 {(variance[0] * 100).toFixed(1)}% · PCA2 {(variance[1] * 100).toFixed(1)}% variance explained
        </span>
        <div style={{ marginLeft: "auto", display: "flex", gap: "0.35rem" }}>
          <button type="button" onClick={() => setZoom((z) => Math.min(z * 1.4, 6))}
            style={toolBtn}><ZoomIn size={13} /></button>
          <button type="button" onClick={() => setZoom((z) => Math.max(z / 1.4, 0.4))}
            style={toolBtn}><ZoomOut size={13} /></button>
          <button type="button" onClick={resetView}
            style={toolBtn}><Scan size={13} /></button>
        </div>
      </div>

      {/* Cluster legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginBottom: "0.65rem" }}>
        <button type="button"
          onClick={() => setActiveCluster(null)}
          style={{
            ...legendBtn,
            borderColor: activeCluster === null ? "#94a3b8" : "rgba(148,163,184,0.2)",
            color: activeCluster === null ? "#e2e8f0" : "#64748b",
          }}
        >All clusters</button>
        {clusterIds.map((id) => (
          <button key={id} type="button"
            onClick={() => setActiveCluster(activeCluster === id ? null : id)}
            style={{
              ...legendBtn,
              borderColor: activeCluster === id ? clusterColor(id) : `${clusterColor(id)}44`,
              color: clusterColor(id),
              background: activeCluster === id ? `${clusterColor(id)}18` : "transparent",
            }}
          >
            <span style={{ width: 7, height: 7, borderRadius: "50%", background: clusterColor(id), display: "inline-block" }} />
            C{id}
          </button>
        ))}
        <span style={{ marginLeft: "auto", fontSize: "0.68rem", color: "#475569", alignSelf: "center" }}>
          ● reference &nbsp; ◆ current
        </span>
      </div>

      {/* SVG canvas */}
      <div style={{ position: "relative", borderRadius: "0.75rem", overflow: "hidden", border: "1px solid rgba(255,255,255,0.07)", background: "#060c18" }}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          style={{ display: "block", cursor: dragging ? "grabbing" : "grab", maxHeight: 380 }}
          onMouseMove={onMouseMove}
          onMouseDown={onMouseDown}
          onMouseUp={() => setDragging(false)}
          onMouseLeave={() => { setDragging(false); setTooltip(null); }}
        >
          {/* Grid lines */}
          {[0.25, 0.5, 0.75].map((t) => (
            <g key={t}>
              <line
                x1={PAD + t * innerW} y1={PAD} x2={PAD + t * innerW} y2={H - PAD}
                stroke="rgba(255,255,255,0.04)" strokeWidth={1}
              />
              <line
                x1={PAD} y1={PAD + t * innerH} x2={W - PAD} y2={PAD + t * innerH}
                stroke="rgba(255,255,255,0.04)" strokeWidth={1}
              />
            </g>
          ))}

          {/* Points */}
          {visiblePoints.map((p, i) => {
            const { x, y } = toSvg(p.x, p.y);
            const col = clusterColor(p.cluster);
            const isRef = p.source === "reference";
            return (
              <g key={i}
                onMouseEnter={(e) => {
                  const rect = svgRef.current!.getBoundingClientRect();
                  setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, point: p });
                }}
                onMouseLeave={() => setTooltip(null)}
              >
                {isRef ? (
                  <circle
                    cx={x} cy={y} r={5}
                    fill={`${col}cc`}
                    stroke={col}
                    strokeWidth={1}
                  />
                ) : (
                  <polygon
                    points={`${x},${y - 6} ${x + 5.2},${y + 3} ${x - 5.2},${y + 3}`}
                    fill={`${col}99`}
                    stroke={col}
                    strokeWidth={1}
                  />
                )}
              </g>
            );
          })}

          {/* Axis labels */}
          <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="#334155">PCA 1</text>
          <text x={8} y={H / 2} textAnchor="middle" fontSize={9} fill="#334155"
            transform={`rotate(-90, 8, ${H / 2})`}>PCA 2</text>
        </svg>

        {/* Tooltip */}
        {tooltip && (
          <div style={{
            position: "absolute",
            left: Math.min(tooltip.x + 12, W - 200),
            top: Math.max(tooltip.y - 10, 4),
            background: "rgba(15,23,42,0.96)",
            border: `1px solid ${clusterColor(tooltip.point.cluster)}55`,
            borderRadius: "0.45rem",
            padding: "0.45rem 0.65rem",
            pointerEvents: "none",
            maxWidth: 200,
            zIndex: 10,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.35rem", marginBottom: "0.3rem" }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: clusterColor(tooltip.point.cluster), display: "inline-block", flexShrink: 0 }} />
              <span style={{ fontSize: "0.67rem", fontWeight: 600, color: clusterColor(tooltip.point.cluster), textTransform: "uppercase" }}>
                Cluster {tooltip.point.cluster}
              </span>
              <span style={{ fontSize: "0.63rem", color: "#64748b", marginLeft: "auto" }}>
                {tooltip.point.source}
              </span>
            </div>
            <p style={{ fontSize: "0.72rem", color: "#cbd5e1", lineHeight: 1.45 }}>
              {tooltip.point.text}{tooltip.point.text.length >= 120 ? "…" : ""}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

const toolBtn: React.CSSProperties = {
  display: "flex", alignItems: "center", justifyContent: "center",
  width: 26, height: 26, borderRadius: "0.35rem",
  border: "1px solid rgba(100,116,139,0.3)",
  background: "rgba(0,0,0,0.3)", color: "#94a3b8",
  cursor: "pointer",
};

const legendBtn: React.CSSProperties = {
  display: "inline-flex", alignItems: "center", gap: "0.3rem",
  fontSize: "0.68rem", fontWeight: 600,
  padding: "0.18rem 0.55rem", borderRadius: "999px",
  border: "1px solid", cursor: "pointer",
  transition: "all 0.12s",
};

// ─── Drift Summary Card ───────────────────────────────────────────────────────

function DriftSummary({ data }: { data: MonitorResult }) {
  const drift = data.semantic_drift;
  const drifted = drift.drift_detected;

  return (
    <div style={{
      borderRadius: "0.8rem",
      border: `1px solid ${drifted ? "rgba(248,113,113,0.3)" : "rgba(52,211,153,0.25)"}`,
      background: drifted ? "rgba(248,113,113,0.05)" : "rgba(52,211,153,0.04)",
      padding: "1rem 1.1rem",
      marginBottom: "1rem",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.6rem" }}>
        <span style={{
          fontSize: "0.72rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.07em",
          padding: "0.15rem 0.6rem", borderRadius: "999px",
          background: drifted ? "rgba(248,113,113,0.15)" : "rgba(52,211,153,0.12)",
          color: drifted ? "#f87171" : "#34d399",
          border: `1px solid ${drifted ? "rgba(248,113,113,0.35)" : "rgba(52,211,153,0.3)"}`,
        }}>
          {drifted ? "⚠ Drift Detected" : "✓ No Significant Drift"}
        </span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(120px,1fr))", gap: "0.5rem" }}>
        {[
          ["Effect Size", drift.effect_size?.toFixed(4)],
          ["Permutation p", drift.permutation_p_value?.toExponential(3)],
          ["p-value", drift.p_value?.toExponential(3)],
        ].map(([k, v]) => (
          <div key={k} style={{ background: "rgba(0,0,0,0.25)", borderRadius: "0.45rem", padding: "0.4rem 0.55rem" }}>
            <p style={{ fontSize: "0.63rem", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "0.18rem" }}>{k}</p>
            <p style={{ fontSize: "0.88rem", fontWeight: 700, color: "#e2e8f0", fontFamily: "monospace" }}>{v ?? "—"}</p>
          </div>
        ))}
      </div>

      {/* Cluster proportions */}
      {data.topic_shift?.clusters?.length > 0 && (
        <div style={{ marginTop: "0.85rem" }}>
          <p style={{ fontSize: "0.68rem", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: "0.5rem" }}>
            Cluster proportion shifts
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
            {data.topic_shift.clusters.map((c) => {
              const change = c.proportion_change;
              const col = change > 0.05 ? "#34d399" : change < -0.05 ? "#f87171" : "#64748b";
              return (
                <div key={c.cluster_id} style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: clusterColor(c.cluster_id), flexShrink: 0 }} />
                  <span style={{ fontSize: "0.72rem", color: "#94a3b8", width: 60 }}>Cluster {c.cluster_id}</span>
                  <div style={{ flex: 1, height: 5, borderRadius: 4, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
                    <div style={{
                      height: "100%",
                      width: `${Math.min(100, Math.abs(change) * 400)}%`,
                      borderRadius: 4,
                      background: col,
                    }} />
                  </div>
                  <span style={{ fontSize: "0.7rem", fontWeight: 600, color: col, width: 52, textAlign: "right", fontFamily: "monospace" }}>
                    {change >= 0 ? "+" : ""}{(change * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function SemanticMonitoring() {
  const [referenceText, setReferenceText] = useState("");
  const [currentText, setCurrentText] = useState("");
  const [nClusters, setNClusters] = useState(5);
  const [nPermutations, setNPermutations] = useState(199);
  const [alpha, setAlpha] = useState(0.05);
  const [randomState, setRandomState] = useState(42);

  const [monitorResult, setMonitorResult] = useState<MonitorResult | null>(null);
  const [projResult, setProjResult] = useState<ProjectionResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<"scatter" | "stats">("scatter");

  async function runMonitoring(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setMonitorResult(null);
    setProjResult(null);

    const reference_texts = referenceText.split("\n").map((t) => t.trim()).filter(Boolean);
    const current_texts = currentText.split("\n").map((t) => t.trim()).filter(Boolean);

    if (reference_texts.length < 5 || current_texts.length < 5) {
      setError("Provide at least five non-empty lines in each collection.");
      return;
    }

    const payload = { reference_texts, current_texts, n_clusters: nClusters, n_permutations: nPermutations, alpha, random_state: randomState };

    setLoading(true);
    try {
      const [monRes, projRes] = await Promise.all([
        axios.post(`${API_BASE}/api/v1/semantic/monitor`, payload),
        axios.post<ProjectionResult>(`${API_BASE}/api/v1/semantic/project`, payload),
      ]);
      setMonitorResult(monRes.data as MonitorResult);
      setProjResult(projRes.data);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(typeof detail === "string" ? detail : `Request failed (${err.response?.status ?? "network error"}).`);
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  }

  const hasResults = monitorResult && projResult;

  return (
    <section className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5 sm:p-6">
      {/* Header */}
      <div className="mb-6 flex items-center gap-3">
        <div className="rounded-xl bg-emerald-400/10 p-3 text-emerald-300">
          <Activity size={23} />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-white">Semantic Drift Monitoring</h2>
          <p className="mt-1 text-sm text-slate-400">
            Compare text distributions and visualize embeddings in 2D PCA space.
          </p>
        </div>
      </div>

      {/* Form */}
      <form onSubmit={runMonitoring} className="space-y-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className="mb-2 block text-sm text-slate-300">Reference texts</label>
            <textarea
              required
              value={referenceText}
              onChange={(e) => setReferenceText(e.target.value)}
              placeholder={"One text per line (≥5 lines required)\nExample: Customers ask about account balances."}
              className="min-h-36 w-full rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm leading-6 text-slate-200 outline-none focus:border-emerald-400"
            />
          </div>
          <div>
            <label className="mb-2 block text-sm text-slate-300">Current texts</label>
            <textarea
              required
              value={currentText}
              onChange={(e) => setCurrentText(e.target.value)}
              placeholder={"One text per line (≥5 lines required)\nExample: Customers now ask about international transfers."}
              className="min-h-36 w-full rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm leading-6 text-slate-200 outline-none focus:border-emerald-400"
            />
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-4">
          {[
            { label: "Clusters", val: nClusters, set: setNClusters, min: 2, max: 20, step: 1 },
            { label: "Permutations", val: nPermutations, set: setNPermutations, min: 19, max: 999, step: 1 },
            { label: "Alpha", val: alpha, set: setAlpha, min: 0.001, max: 0.2, step: 0.005 },
            { label: "Random seed", val: randomState, set: setRandomState, min: 0, max: 9999, step: 1 },
          ].map(({ label, val, set, min, max, step }) => (
            <label key={label} className="block text-sm text-slate-300">
              {label}
              <input
                type="number" required min={min} max={max} step={step} value={val}
                onChange={(e) => set(Number(e.target.value))}
                className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-2.5 text-sm text-slate-200 outline-none focus:border-emerald-400"
              />
            </label>
          ))}
        </div>

        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">
            <AlertTriangle size={16} className="mt-0.5 shrink-0" />{error}
          </div>
        )}

        <button
          type="submit" disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-emerald-400 px-5 py-3 text-sm font-semibold text-slate-950 hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
          {loading ? "Analysing…" : "Run semantic analysis"}
        </button>
      </form>

      {/* Results */}
      {hasResults && (
        <div style={{ marginTop: "1.75rem" }}>
          {/* Tab switcher */}
          <div style={{ display: "flex", gap: "0.35rem", marginBottom: "1rem" }}>
            {(["scatter", "stats"] as const).map((tab) => (
              <button key={tab} type="button" onClick={() => setActiveTab(tab)}
                style={{
                  padding: "0.3rem 0.9rem",
                  borderRadius: "999px",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  border: "1px solid",
                  borderColor: activeTab === tab ? "#34d399" : "rgba(100,116,139,0.3)",
                  background: activeTab === tab ? "rgba(52,211,153,0.12)" : "transparent",
                  color: activeTab === tab ? "#34d399" : "#64748b",
                  transition: "all 0.12s",
                }}
              >
                {tab === "scatter" ? "📍 Embedding Scatter Plot" : "📊 Drift Statistics"}
              </button>
            ))}
          </div>

          {activeTab === "scatter" && (
            <div>
              <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.75rem", flexWrap: "wrap" }}>
                <span style={{ fontSize: "0.73rem", color: "#64748b" }}>
                  {projResult.reference_count} reference · {projResult.current_count} current · {projResult.n_clusters} clusters
                </span>
              </div>
              <ScatterPlot
                points={projResult.points}
                nClusters={projResult.n_clusters}
                variance={projResult.pca_variance_explained}
              />
            </div>
          )}

          {activeTab === "stats" && (
            <DriftSummary data={monitorResult} />
          )}
        </div>
      )}
    </section>
  );
}
