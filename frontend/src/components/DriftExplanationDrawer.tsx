import { useState } from "react";
import {
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  Brain,
  RefreshCw,
  X,
} from "lucide-react";
import { explainDrift } from "../api";
import type { DriftExplanation } from "../types";

interface DriftExplanationDrawerProps {
  initialSummary?: Record<string, any>;
  initialFeatures?: Record<string, any>[];
  isOpen?: boolean;
  onClose?: () => void;
}

const statusColors: Record<string, { bg: string; text: string; border: string }> = {
  CRITICAL: {
    bg: "bg-rose-500/10",
    text: "text-rose-400",
    border: "border-rose-500/30",
  },
  WARNING: {
    bg: "bg-amber-500/10",
    text: "text-amber-400",
    border: "border-amber-500/30",
  },
  MONITOR: {
    bg: "bg-cyan-500/10",
    text: "text-cyan-400",
    border: "border-cyan-500/30",
  },
  HEALTHY: {
    bg: "bg-emerald-500/10",
    text: "text-emerald-400",
    border: "border-emerald-500/30",
  },
  ANALYZED: {
    bg: "bg-indigo-500/10",
    text: "text-indigo-400",
    border: "border-indigo-500/30",
  },
};

export default function DriftExplanationDrawer({
  initialSummary,
  initialFeatures,
  isOpen = true,
  onClose,
}: DriftExplanationDrawerProps) {
  const [apiKey, setApiKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [explanation, setExplanation] = useState<DriftExplanation | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRunExplanation = async () => {
    setLoading(true);
    setError(null);

    const summary = initialSummary || {
      total_features: 12,
      drifted_features_count: 3,
      dataset: "production_inferences_v2",
    };

    const features = initialFeatures || [
      {
        feature_name: "debt_to_income_ratio",
        ks_statistic: 0.384,
        psi_score: 0.28,
        severity: "high",
        drift_detected: true,
      },
      {
        feature_name: "credit_inquiry_count",
        ks_statistic: 0.215,
        psi_score: 0.16,
        severity: "medium",
        drift_detected: true,
      },
      {
        feature_name: "revolving_utilization",
        ks_statistic: 0.192,
        psi_score: 0.13,
        severity: "medium",
        drift_detected: true,
      },
    ];

    try {
      const res = await explainDrift(summary, features, apiKey || undefined);
      setExplanation(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to generate explanation");
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const currentStatus = explanation?.health_status || "MONITOR";
  const statusTheme = statusColors[currentStatus] || statusColors.MONITOR;

  return (
    <div className="rounded-xl border border-white/[0.08] bg-[#0c0e15] p-5 shadow-sm">
      <div className="flex flex-col justify-between gap-3 border-b border-white/[0.06] pb-4 sm:flex-row sm:items-center">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-md bg-indigo-500/20 text-indigo-400">
              <Brain size={14} />
            </span>
            <h3 className="text-sm font-semibold text-white">
              Executive AI Diagnostic & Root-Cause Explanation
            </h3>
          </div>
          <p className="mt-1 text-xs text-zinc-400">
            Translates raw Kolmogorov-Smirnov and PSI divergence stats into plain-English root causes and remediation playbooks.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="password"
            placeholder="Optional OpenAI API Key (or heuristic fallback)"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            className="linear-input w-56 px-2.5 py-1 text-[11px]"
          />
          <button
            type="button"
            onClick={() => void handleRunExplanation()}
            disabled={loading}
            className="linear-button-primary flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold"
          >
            {loading ? (
              <RefreshCw size={12} className="animate-spin" />
            ) : (
              <Sparkles size={12} />
            )}
            <span>Explain Drift</span>
          </button>

          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="rounded p-1 text-zinc-500 hover:text-white"
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
          {error}
        </div>
      )}

      {!explanation && !loading && (
        <div className="mt-4 rounded-lg border border-dashed border-white/[0.06] p-8 text-center text-xs text-zinc-500">
          <Sparkles size={20} className="mx-auto mb-2 text-zinc-600" />
          Click &ldquo;Explain Drift&rdquo; to synthesize an operational diagnosis from current telemetry.
        </div>
      )}

      {explanation && (
        <div className="mt-5 space-y-4 text-xs">
          {/* Header Banner */}
          <div
            className={`flex flex-col justify-between gap-2 rounded-lg border p-4 sm:flex-row sm:items-center ${statusTheme.bg} ${statusTheme.border}`}
          >
            <div>
              <span
                className={`font-mono text-[10px] font-bold uppercase tracking-wider ${statusTheme.text}`}
              >
                Health Status: {explanation.health_status}
              </span>
              <p className="mt-0.5 text-sm font-semibold text-zinc-100">
                {explanation.headline}
              </p>
            </div>
            <span className="rounded-full border border-white/[0.1] bg-white/[0.05] px-2 py-0.5 text-[10px] font-mono text-zinc-400">
              Engine: {explanation.engine}
            </span>
          </div>

          {/* Narrative Summary */}
          <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-4">
            <h4 className="text-[11px] font-semibold text-zinc-300">
              Diagnostic Narrative
            </h4>
            <p className="mt-1.5 leading-relaxed text-zinc-400">
              {explanation.summary}
            </p>
          </div>

          {/* Key Statistical Findings */}
          {explanation.key_findings && explanation.key_findings.length > 0 && (
            <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-4">
              <h4 className="text-[11px] font-semibold text-zinc-300">
                Key Statistical Divergence Points
              </h4>
              <ul className="mt-2 space-y-1.5">
                {explanation.key_findings.map((f, i) => (
                  <li key={i} className="flex items-start gap-2 text-zinc-400">
                    <AlertTriangle size={13} className="mt-0.5 shrink-0 text-amber-400" />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Recommended Actions */}
          <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-4">
            <h4 className="text-[11px] font-semibold text-zinc-300">
              Recommended Remediation Steps
            </h4>
            <div className="mt-2 space-y-2">
              {explanation.recommended_actions.map((act, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2.5 rounded border border-white/[0.04] bg-[#0e1017] p-2.5 text-zinc-300"
                >
                  <CheckCircle2 size={14} className="mt-0.5 shrink-0 text-emerald-400" />
                  <span className="flex-1 text-[11px]">{act}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
