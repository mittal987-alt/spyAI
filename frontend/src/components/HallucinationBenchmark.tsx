import "./HallucinationBenchmark.css";
import { useState, useCallback } from "react";
import {
  Microscope,
  FileText,
  Loader2,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronRight,
  Plus,
  Trash2,
  BarChart3,
  Zap,
} from "lucide-react";
import {
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Cell,
} from "recharts";

// ── Types ──────────────────────────────────────────────────────────────────────

interface RougeScores {
  precision: number; recall: number; f1: number;
}
interface HallucinationResult {
  rouge_scores: { rouge1: RougeScores; rouge2: RougeScores; rougeL: RougeScores & { lcs_length: number } };
  token_metrics: { precision: number; recall: number; f1: number; hypothesis_length: number; reference_length: number; length_ratio: number };
  novelty: { bigram_novelty_rate: number; interpretation: string };
  claim_analysis: { total_claims: number; supported_claims: number; unsupported_claims: number; support_rate: number; claim_details: { claim: string; supported: boolean; support_ratio: number }[] };
  hallucination_risk: { risk_score: number; risk_level: string; verdict: string; faithfulness_score: number };
}
interface BatchResult extends HallucinationResult { pair_index: number; status: "ok" | "error"; error?: string }
interface BatchResponse {
  results: BatchResult[];
  aggregate_stats: { total_pairs: number; evaluated: number; errors: number; avg_hallucination_risk: number; avg_faithfulness_score: number; avg_rouge1_f1: number; high_risk_count: number; high_risk_rate: number };
}

// ── Helpers ────────────────────────────────────────────────────────────────────

const API = "http://localhost:8000";

const riskColor = (level: string) => {
  if (level === "Low") return "#22c55e";
  if (level === "Moderate") return "#f59e0b";
  if (level === "High") return "#f97316";
  return "#ef4444";
};

const RiskGauge = ({ score, level }: { score: number; level: string }) => {
  const color = riskColor(level);
  const size = 120;
  const r = 46;
  const circ = Math.PI * r; // half circle
  const offset = circ - (score / 100) * circ;
  return (
    <div className="hb-gauge">
      <svg width={size} height={size / 2 + 14}>
        <path d={`M 10,${size / 2} A ${r} ${r} 0 0 1 ${size - 10},${size / 2}`} fill="none" stroke="#1e293b" strokeWidth={12} />
        <path
          d={`M 10,${size / 2} A ${r} ${r} 0 0 1 ${size - 10},${size / 2}`}
          fill="none" stroke={color} strokeWidth={12}
          strokeDasharray={circ} strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.7s ease" }}
        />
      </svg>
      <div className="hb-gauge-label">
        <span className="hb-gauge-score" style={{ color }}>{score}</span>
        <span className="hb-gauge-level" style={{ color }}>{level}</span>
        <span className="hb-gauge-sub">Hallucination Risk</span>
      </div>
    </div>
  );
};

const MetricBar = ({ label, value, color = "#6366f1" }: { label: string; value: number; color?: string }) => (
  <div className="hb-metric-bar">
    <div className="hb-metric-top">
      <span className="hb-metric-label">{label}</span>
      <span className="hb-metric-value">{(value * 100).toFixed(1)}%</span>
    </div>
    <div className="hb-bar-bg">
      <div className="hb-bar-fill" style={{ width: `${value * 100}%`, background: color, transition: "width 0.5s ease" }} />
    </div>
  </div>
);

const ClaimRow = ({ claim }: { claim: { claim: string; supported: boolean; support_ratio: number } }) => (
  <div className={`hb-claim-row ${claim.supported ? "hb-claim-ok" : "hb-claim-bad"}`}>
    {claim.supported ? <CheckCircle size={13} className="hb-icon-ok" /> : <XCircle size={13} className="hb-icon-bad" />}
    <p className="hb-claim-text">{claim.claim}</p>
    <span className="hb-claim-ratio">{(claim.support_ratio * 100).toFixed(0)}%</span>
  </div>
);

const SingleResult = ({ result }: { result: HallucinationResult }) => {
  const [claimsOpen, setClaimsOpen] = useState(false);

  const radarData = [
    { metric: "ROUGE-1", value: result.rouge_scores.rouge1.f1 * 100 },
    { metric: "ROUGE-2", value: result.rouge_scores.rouge2.f1 * 100 },
    { metric: "ROUGE-L", value: result.rouge_scores.rougeL.f1 * 100 },
    { metric: "Token F1", value: result.token_metrics.f1 * 100 },
    { metric: "Claim Support", value: result.claim_analysis.support_rate * 100 },
    { metric: "Faithfulness", value: result.hallucination_risk.faithfulness_score },
  ];

  return (
    <div className="hb-result-card">
      {/* Top row */}
      <div className="hb-result-top">
        <RiskGauge score={result.hallucination_risk.risk_score} level={result.hallucination_risk.risk_level} />
        <div className="hb-result-verdict">
          <p className="hb-verdict-text">{result.hallucination_risk.verdict}</p>
          <div className="hb-verdict-stats">
            <div className="hb-vstat"><span>Faithfulness</span><strong>{result.hallucination_risk.faithfulness_score}%</strong></div>
            <div className="hb-vstat"><span>Novelty Rate</span><strong>{(result.novelty.bigram_novelty_rate * 100).toFixed(1)}%</strong></div>
            <div className="hb-vstat"><span>Claims</span><strong>{result.claim_analysis.supported_claims}/{result.claim_analysis.total_claims} supported</strong></div>
            <div className="hb-vstat"><span>Length Ratio</span><strong>{result.token_metrics.length_ratio.toFixed(2)}×</strong></div>
          </div>
          <p className="hb-novelty-note">{result.novelty.interpretation}</p>
        </div>
      </div>

      {/* Radar */}
      <div className="hb-radar-wrapper">
        <ResponsiveContainer width="100%" height={220}>
          <RadarChart data={radarData} margin={{ top: 8, right: 30, bottom: 8, left: 30 }}>
            <PolarGrid stroke="#1e293b" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 11 }} />
            <Radar name="Score" dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.25} />
            <Tooltip formatter={(v) => [typeof v === "number" ? `${v.toFixed(1)}%` : String(v ?? ""), "Score"]} contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8 }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Metric bars */}
      <div className="hb-metrics-grid">
        <MetricBar label="ROUGE-1 F1" value={result.rouge_scores.rouge1.f1} color="#6366f1" />
        <MetricBar label="ROUGE-2 F1" value={result.rouge_scores.rouge2.f1} color="#8b5cf6" />
        <MetricBar label="ROUGE-L F1" value={result.rouge_scores.rougeL.f1} color="#06b6d4" />
        <MetricBar label="Token Precision" value={result.token_metrics.precision} color="#10b981" />
        <MetricBar label="Token Recall" value={result.token_metrics.recall} color="#f59e0b" />
        <MetricBar label="Claim Support Rate" value={result.claim_analysis.support_rate} color="#ec4899" />
      </div>

      {/* Claims */}
      {result.claim_analysis.claim_details.length > 0 && (
        <div className="hb-claims-section">
          <button className="hb-claims-toggle" onClick={() => setClaimsOpen(v => !v)}>
            {claimsOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            Claim-Level Analysis ({result.claim_analysis.total_claims} claims)
          </button>
          {claimsOpen && (
            <div className="hb-claims-list">
              {result.claim_analysis.claim_details.map((c, i) => (
                <ClaimRow key={i} claim={c} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── Main Component ─────────────────────────────────────────────────────────────

interface Pair { generated: string; reference: string }

export default function HallucinationBenchmark() {
  const [mode, setMode] = useState<"single" | "batch">("single");
  const [generated, setGenerated] = useState("");
  const [reference, setReference] = useState("");
  const [pairs, setPairs] = useState<Pair[]>([{ generated: "", reference: "" }]);
  const [result, setResult] = useState<HallucinationResult | null>(null);
  const [batchResult, setBatchResult] = useState<BatchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runSingle = useCallback(async () => {
    if (!generated.trim() || !reference.trim()) { setError("Both fields are required."); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await fetch(`${API}/api/v1/hallucination/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ generated, reference }),
      });
      if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail ?? `HTTP ${res.status}`); }
      setResult(await res.json());
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, [generated, reference]);

  const runBatch = useCallback(async () => {
    const valid = pairs.filter(p => p.generated.trim() && p.reference.trim());
    if (!valid.length) { setError("Add at least one pair with both fields filled."); return; }
    setLoading(true); setError(null); setBatchResult(null);
    try {
      const res = await fetch(`${API}/api/v1/hallucination/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pairs: valid }),
      });
      if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.detail ?? `HTTP ${res.status}`); }
      setBatchResult(await res.json());
    } catch (e: unknown) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setLoading(false); }
  }, [pairs]);

  const addPair = () => setPairs(p => [...p, { generated: "", reference: "" }]);
  const removePair = (i: number) => setPairs(p => p.filter((_, idx) => idx !== i));
  const updatePair = (i: number, field: keyof Pair, val: string) =>
    setPairs(p => p.map((pair, idx) => idx === i ? { ...pair, [field]: val } : pair));

  const preloadExample = () => {
    setGenerated("The Eiffel Tower is a wrought-iron structure located in Paris, France. It was constructed between 1887 and 1889 and stands 330 metres tall.");
    setReference("The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. It was constructed between 1887 and 1889 as the centerpiece of the 1889 World's Fair. The tower stands 330 metres tall.");
    setMode("single");
  };

  const batchBarData = batchResult?.results
    .filter(r => r.status === "ok")
    .map(r => ({
      name: `Pair ${r.pair_index + 1}`,
      risk: r.hallucination_risk.risk_score,
      faithfulness: r.hallucination_risk.faithfulness_score,
    })) ?? [];

  return (
    <div className="hb-root">
      {/* Header */}
      <div className="hb-header">
        <div className="hb-header-icon"><Microscope size={26} /></div>
        <div>
          <h1 className="hb-title">LLM Hallucination Benchmarking</h1>
          <p className="hb-subtitle">
            Score LLM outputs for factual consistency using ROUGE, novelty rate, and claim-level entailment
          </p>
        </div>
        <button className="hb-example-btn" onClick={preloadExample}>
          <Zap size={14} /> Load Example
        </button>
      </div>

      {/* Mode tabs */}
      <div className="hb-tabs">
        <button className={`hb-tab ${mode === "single" ? "hb-tab-active" : ""}`} onClick={() => setMode("single")}>
          <FileText size={14} /> Single Evaluation
        </button>
        <button className={`hb-tab ${mode === "batch" ? "hb-tab-active" : ""}`} onClick={() => setMode("batch")}>
          <BarChart3 size={14} /> Batch Benchmark
        </button>
      </div>

      <div className="hb-layout">
        {/* ── Single mode ── */}
        {mode === "single" && (
          <div className="hb-single-layout">
            <div className="hb-input-panel">
              <label className="hb-label">
                LLM-Generated Output
                <textarea className="hb-textarea hb-ta-lg" value={generated} onChange={e => setGenerated(e.target.value)}
                  rows={7} placeholder="Paste the model's answer here…" />
              </label>
              <label className="hb-label">
                Reference / Ground-Truth Document
                <textarea className="hb-textarea hb-ta-lg" value={reference} onChange={e => setReference(e.target.value)}
                  rows={7} placeholder="Paste the ground-truth context or answer here…" />
              </label>
              <button className="hb-btn-run" onClick={runSingle} disabled={loading}>
                {loading ? <><Loader2 size={15} className="hb-spin" /> Evaluating…</> : <><Microscope size={15} /> Evaluate Hallucination</>}
              </button>
              {error && <div className="hb-error"><AlertTriangle size={14} />{error}</div>}
            </div>

            <div className="hb-result-panel">
              {!result && !loading && (
                <div className="hb-empty">
                  <Microscope size={52} className="hb-empty-icon" />
                  <p>Paste generated text and reference, then run the evaluation.</p>
                  <ul className="hb-feature-list">
                    <li>📊 ROUGE-1 / ROUGE-2 / ROUGE-L overlap scores</li>
                    <li>🔍 Bigram novelty rate — unsupported new content</li>
                    <li>🧩 Claim-level entailment check per sentence</li>
                    <li>⚠️ Aggregated 0–100 hallucination risk score</li>
                  </ul>
                </div>
              )}
              {loading && <div className="hb-loading"><Loader2 size={42} className="hb-spin" /><p>Analysing factual consistency…</p></div>}
              {result && <SingleResult result={result} />}
            </div>
          </div>
        )}

        {/* ── Batch mode ── */}
        {mode === "batch" && (
          <div className="hb-batch-layout">
            <div className="hb-batch-inputs">
              <div className="hb-section-label"><BarChart3 size={14} /> Pairs ({pairs.length}/50)</div>
              {pairs.map((pair, i) => (
                <div key={i} className="hb-pair-card">
                  <div className="hb-pair-header">
                    <span className="hb-pair-num">Pair {i + 1}</span>
                    {pairs.length > 1 && (
                      <button className="hb-remove-btn" onClick={() => removePair(i)}><Trash2 size={13} /></button>
                    )}
                  </div>
                  <label className="hb-label">
                    Generated
                    <textarea className="hb-textarea" rows={3} value={pair.generated}
                      onChange={e => updatePair(i, "generated", e.target.value)}
                      placeholder="LLM output…" />
                  </label>
                  <label className="hb-label">
                    Reference
                    <textarea className="hb-textarea" rows={3} value={pair.reference}
                      onChange={e => updatePair(i, "reference", e.target.value)}
                      placeholder="Ground-truth…" />
                  </label>
                </div>
              ))}

              <button className="hb-add-pair-btn" onClick={addPair} disabled={pairs.length >= 50}>
                <Plus size={14} /> Add Pair
              </button>

              <button className="hb-btn-run" onClick={runBatch} disabled={loading}>
                {loading ? <><Loader2 size={15} className="hb-spin" /> Running…</> : <><BarChart3 size={15} /> Run Batch Benchmark</>}
              </button>
              {error && <div className="hb-error"><AlertTriangle size={14} />{error}</div>}
            </div>

            {/* Batch results */}
            <div className="hb-batch-results">
              {!batchResult && !loading && (
                <div className="hb-empty">
                  <BarChart3 size={52} className="hb-empty-icon" />
                  <p>Add multiple generated/reference pairs and run the batch benchmark to compare hallucination scores across outputs.</p>
                </div>
              )}
              {loading && <div className="hb-loading"><Loader2 size={42} className="hb-spin" /><p>Evaluating batch…</p></div>}
              {batchResult && (
                <>
                  {/* Aggregate summary */}
                  <div className="hb-agg-card">
                    <div className="hb-agg-grid">
                      <div className="hb-agg-stat">
                        <span className="hb-agg-label">Avg Risk Score</span>
                        <span className="hb-agg-val" style={{ color: "#ef4444" }}>{batchResult.aggregate_stats.avg_hallucination_risk}</span>
                      </div>
                      <div className="hb-agg-stat">
                        <span className="hb-agg-label">Avg Faithfulness</span>
                        <span className="hb-agg-val" style={{ color: "#22c55e" }}>{batchResult.aggregate_stats.avg_faithfulness_score}%</span>
                      </div>
                      <div className="hb-agg-stat">
                        <span className="hb-agg-label">ROUGE-1 F1 Avg</span>
                        <span className="hb-agg-val" style={{ color: "#6366f1" }}>{(batchResult.aggregate_stats.avg_rouge1_f1 * 100).toFixed(1)}%</span>
                      </div>
                      <div className="hb-agg-stat">
                        <span className="hb-agg-label">High Risk Pairs</span>
                        <span className="hb-agg-val" style={{ color: "#f97316" }}>
                          {batchResult.aggregate_stats.high_risk_count} / {batchResult.aggregate_stats.evaluated}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Bar chart */}
                  {batchBarData.length > 0 && (
                    <div className="hb-batch-chart">
                      <div className="hb-section-label"><BarChart3 size={14} /> Risk & Faithfulness per Pair</div>
                      <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={batchBarData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} domain={[0, 100]} />
                          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8 }} />
                          <Bar dataKey="risk" name="Hallucination Risk" radius={[4, 4, 0, 0]}>
                            {batchBarData.map((d, i) => (
                              <Cell key={i} fill={d.risk > 40 ? "#ef4444" : d.risk > 20 ? "#f59e0b" : "#22c55e"} />
                            ))}
                          </Bar>
                          <Bar dataKey="faithfulness" name="Faithfulness" fill="#6366f1" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* Per-pair cards */}
                  {batchResult.results.map((r, i) =>
                    r.status === "ok" ? (
                      <details key={i} className="hb-batch-detail">
                        <summary className="hb-batch-summary">
                          <span>Pair {r.pair_index + 1}</span>
                          <span style={{ color: riskColor(r.hallucination_risk.risk_level) }}>
                            Risk {r.hallucination_risk.risk_score} — {r.hallucination_risk.risk_level}
                          </span>
                        </summary>
                        <SingleResult result={r} />
                      </details>
                    ) : (
                      <div key={i} className="hb-error">Pair {r.pair_index + 1}: {r.error}</div>
                    )
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
