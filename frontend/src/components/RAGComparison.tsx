
import { useState } from "react";
import axios from "axios";
import { ArrowDownRight, ArrowUpRight, GitCompare, Loader2 } from "lucide-react";
import { compareRagRuns } from "../api";
import type { RagHistoryItem, RagComparisonResponse } from "../types";

export default function RAGComparison({
  history,
}: {
  history: RagHistoryItem[];
}) {
  const [baseline, setBaseline] = useState("");
  const [candidate, setCandidate] = useState("");
  const [result, setResult] = useState<RagComparisonResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function runComparison() {
    setError("");
    setResult(null);

    if (!baseline || !candidate) {
      setError("Choose both a baseline run and a candidate run.");
      return;
    }

    if (baseline === candidate) {
      setError("Choose two different evaluation runs.");
      return;
    }

    setLoading(true);
    try {
      setResult(await compareRagRuns(baseline, candidate));
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(
          typeof detail === "string"
            ? detail
            : `Comparison failed (${err.response?.status ?? "network error"}).`,
        );
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5 sm:p-6">
      <div className="mb-5 flex items-center gap-3">
        <div className="rounded-lg bg-violet-400/10 p-2 text-violet-300">
          <GitCompare size={20} />
        </div>
        <div>
          <h2 className="font-semibold text-white">Compare evaluation runs</h2>
          <p className="mt-1 text-sm text-slate-500">
            Compare metric changes and inspect potential regressions.
          </p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <label className="text-sm text-slate-300">
          Baseline run
          <select
            value={baseline}
            onChange={(e) => setBaseline(e.target.value)}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3"
          >
            <option value="">Select baseline</option>
            {history.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id.slice(0, 8)} — {run.query.slice(0, 55)}
              </option>
            ))}
          </select>
        </label>

        <label className="text-sm text-slate-300">
          Candidate run
          <select
            value={candidate}
            onChange={(e) => setCandidate(e.target.value)}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3"
          >
            <option value="">Select candidate</option>
            {history.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id.slice(0, 8)} — {run.query.slice(0, 55)}
              </option>
            ))}
          </select>
        </label>
      </div>

      <button
        onClick={() => void runComparison()}
        disabled={loading || history.length < 2}
        className="mt-5 flex items-center gap-2 rounded-lg border border-violet-400/30 bg-violet-400/10 px-4 py-3 text-sm text-violet-200 hover:bg-violet-400/20 disabled:opacity-40"
      >
        {loading ? <Loader2 size={16} className="animate-spin" /> : <GitCompare size={16} />}
        Compare runs
      </button>

      {history.length < 2 && (
        <p className="mt-3 text-xs text-slate-500">
          Save at least two evaluation runs to enable comparisons.
        </p>
      )}

      {error && (
        <p className="mt-4 rounded-lg bg-rose-500/10 p-3 text-sm text-rose-200">
          {error}
        </p>
      )}

      {result && (
        <div className="mt-6 space-y-4">
          <div
            className={`rounded-xl border p-4 ${
              result.regression_detected
                ? "border-rose-500/30 bg-rose-500/10"
                : "border-emerald-500/30 bg-emerald-500/10"
            }`}
          >
            <p className="font-medium text-white">
              {result.regression_detected
                ? "Potential regression detected"
                : "No regression detected by configured checks"}
            </p>
            <p className="mt-1 text-sm text-slate-400">
              This is a metric-based comparison, not a guarantee of answer correctness.
            </p>
          </div>

          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full min-w-[560px] text-left text-sm">
              <thead className="bg-slate-900 text-xs text-slate-400">
                <tr>
                  <th className="p-3">Metric</th>
                  <th className="p-3">Baseline</th>
                  <th className="p-3">Candidate</th>
                  <th className="p-3">Change</th>
                  <th className="p-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.metrics).map(([name, metric]) => {
                  const change = metric.change;
                  const isRegression = metric.status.toLowerCase().includes("regression");
                  return (
                    <tr key={name} className="border-t border-slate-800">
                      <td className="p-3 text-slate-300">
                        {name.replaceAll("_", " ")}
                      </td>
                      <td className="p-3">{formatScore(metric.baseline)}</td>
                      <td className="p-3">{formatScore(metric.candidate)}</td>
                      <td className="p-3">
                        <span className="flex items-center gap-1">
                          {change != null && change > 0 ? (
                            <ArrowUpRight size={14} className="text-emerald-300" />
                          ) : change != null && change < 0 ? (
                            <ArrowDownRight size={14} className="text-rose-300" />
                          ) : null}
                          {formatScore(change)}
                        </span>
                      </td>
                      <td className={`p-3 ${isRegression ? "text-rose-300" : "text-slate-400"}`}>
                        {metric.status}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {result.limitations.length > 0 && (
            <div className="space-y-2">
              {result.limitations.map((item, index) => (
                <p key={index} className="text-xs leading-5 text-slate-500">
                  {item}
                </p>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function formatScore(value: number | null) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(4);
}
