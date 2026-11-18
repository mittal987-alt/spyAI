import { useState, useEffect } from "react";
import {
  Activity,
  DollarSign,
  Clock,
  Zap,
  Play,
  CheckCircle2,
  XCircle,
  Cpu,
  RefreshCw,
  Layers,
} from "lucide-react";
import {
  getLLMTraces,
  getLLMTraceStats,
  getLLMProviders,
  generateWithLLM,
} from "../api";
import type {
  LLMTrace,
  LLMTraceStats,
  LLMProviderInfo,
} from "../types";

export function LLMTracingPanel() {
  const [traces, setTraces] = useState<LLMTrace[]>([]);
  const [stats, setStats] = useState<LLMTraceStats | null>(null);
  const [providers, setProviders] = useState<LLMProviderInfo[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>("auto");
  const [testPrompt, setTestPrompt] = useState<string>("Summarize the key causes of model drift in production.");
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [selectedTrace, setSelectedTrace] = useState<LLMTrace | null>(null);
  const [lastOutput, setLastOutput] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setIsLoading(true);
      const [tracesData, statsData, providersData] = await Promise.all([
        getLLMTraces(30),
        getLLMTraceStats(),
        getLLMProviders(),
      ]);
      setTraces(tracesData);
      setStats(statsData);
      setProviders(providersData.providers);
      if (providersData.recommended_provider && selectedProvider === "auto") {
        setSelectedProvider(providersData.recommended_provider);
      }
    } catch (err) {
      console.error("Failed to load LLM traces or stats", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleRunTest = async () => {
    if (!testPrompt.trim()) return;
    setIsRunning(true);
    setLastOutput(null);
    try {
      const res = await generateWithLLM({
        prompt: testPrompt,
        provider: selectedProvider,
      });
      setLastOutput(res.text);
      await loadData();
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : "Inference failed";
      setLastOutput(`Error: ${errorMsg}`);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner & Refresh */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-4 rounded-xl bg-slate-900 border border-slate-800">
        <div>
          <h2 className="text-xl font-semibold text-slate-100 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-400" />
            LLM Tracing & Token Cost Telemetry
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Real-time observability into LLM inference latency, token usage, and cumulative dollar spend across local and cloud models.
          </p>
        </div>
        <button
          onClick={loadData}
          disabled={isLoading}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Inferences */}
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Total Inferences</span>
            <Zap className="w-4 h-4 text-blue-400" />
          </div>
          <p className="text-2xl font-bold text-slate-100 mt-2">
            {stats ? stats.total_calls.toLocaleString() : "--"}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            {stats ? `${stats.total_tokens.toLocaleString()} tokens tracked` : "Zero calls"}
          </p>
        </div>

        {/* Total Cost */}
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Cumulative Spend</span>
            <DollarSign className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="text-2xl font-bold text-emerald-400 mt-2">
            ${stats ? stats.total_cost_usd.toFixed(4) : "0.0000"}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            Ollama / Heuristic: <span className="text-emerald-400 font-semibold">$0.00 (100% Free)</span>
          </p>
        </div>

        {/* Avg Latency */}
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Average Latency</span>
            <Clock className="w-4 h-4 text-amber-400" />
          </div>
          <p className="text-2xl font-bold text-slate-100 mt-2">
            {stats ? `${stats.avg_latency_ms} ms` : "--"}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            Min: {stats ? `${stats.min_latency_ms} ms` : "--"} • Max: {stats ? `${stats.max_latency_ms} ms` : "--"}
          </p>
        </div>

        {/* Reliability / Success Rate */}
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span>Success Rate</span>
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="text-2xl font-bold text-slate-100 mt-2">
            {stats ? `${stats.success_rate}%` : "--"}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            Zero network timeouts recorded
          </p>
        </div>
      </div>

      {/* Live Interactive Query & Trace Runner */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-blue-400" />
            Live Trace Generator (Test Any Provider)
          </h3>
          <span className="text-xs text-slate-500">
            Traces spans, latency, and exact token pricing automatically
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {providers.map((p) => {
            const isSelected = selectedProvider === p.id;
            return (
              <button
                key={p.id}
                onClick={() => setSelectedProvider(p.id)}
                className={`p-3 rounded-lg border text-left transition ${
                  isSelected
                    ? "border-blue-500 bg-blue-500/10 text-white"
                    : "border-slate-800 bg-slate-950/40 text-slate-300 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center justify-between text-xs font-semibold mb-1">
                  <span>{p.name}</span>
                  <span
                    className={`px-1.5 py-0.5 rounded text-[10px] ${
                      p.available ? "bg-emerald-500/10 text-emerald-400" : "bg-slate-800 text-slate-500"
                    }`}
                  >
                    {p.available ? "Available" : "Offline"}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 line-clamp-2">
                  {p.description}
                </p>
              </button>
            );
          })}
        </div>

        <div className="space-y-2">
          <label className="text-xs font-medium text-slate-300">
            Prompt / Task
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={testPrompt}
              onChange={(e) => setTestPrompt(e.target.value)}
              placeholder="Enter diagnostic question or prompt..."
              className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none focus:border-blue-500"
            />
            <button
              onClick={handleRunTest}
              disabled={isRunning || !testPrompt.trim()}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white transition"
            >
              {isRunning ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {isRunning ? "Running..." : "Run & Trace"}
            </button>
          </div>
        </div>

        {lastOutput && (
          <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs">
            <span className="text-slate-400 font-semibold block mb-1">
              Latest Response:
            </span>
            <p className="text-slate-200 whitespace-pre-wrap font-mono text-[11px]">
              {lastOutput}
            </p>
          </div>
        )}
      </div>

      {/* Traces Table & Span Waterfall */}
      <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Layers className="w-4 h-4 text-blue-400" />
            Execution Traces ({traces.length})
          </h3>
          <span className="text-xs text-slate-500">
            Click any row to inspect span breakdown
          </span>
        </div>

        {traces.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-sm">
            No LLM calls recorded yet. Use the test runner above to generate your first trace.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/60 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Timestamp</th>
                  <th className="py-2.5 px-3">Provider & Model</th>
                  <th className="py-2.5 px-3">Prompt Sample</th>
                  <th className="py-2.5 px-3 text-right">Tokens</th>
                  <th className="py-2.5 px-3 text-right">Latency</th>
                  <th className="py-2.5 px-3 text-right">Est. Cost</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {traces.map((t) => {
                  const isSelected = selectedTrace?.trace_id === t.trace_id;
                  return (
                    <tr
                      key={t.trace_id}
                      onClick={() => setSelectedTrace(isSelected ? null : t)}
                      className={`cursor-pointer transition hover:bg-slate-800/40 ${
                        isSelected ? "bg-blue-950/30 border-l-2 border-blue-500" : ""
                      }`}
                    >
                      <td className="py-2.5 px-3">
                        {t.success ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        ) : (
                          <XCircle className="w-4 h-4 text-rose-400" />
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-slate-400 font-mono text-[11px]">
                        {new Date(t.created_at).toLocaleTimeString()}
                      </td>
                      <td className="py-2.5 px-3">
                        <span className="font-semibold text-slate-200 capitalize">
                          {t.provider}
                        </span>
                        <span className="text-slate-500 block text-[10px] font-mono">
                          {t.model}
                        </span>
                      </td>
                      <td className="py-2.5 px-3 max-w-[200px] truncate text-slate-400">
                        {t.prompt}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-slate-300">
                        {t.total_tokens.toLocaleString()}
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono text-amber-300">
                        {t.latency_ms.toFixed(1)} ms
                      </td>
                      <td className="py-2.5 px-3 text-right font-mono">
                        {t.estimated_cost_usd === 0 ? (
                          <span className="text-emerald-400 font-semibold">$0.00</span>
                        ) : (
                          <span className="text-slate-200">
                            ${t.estimated_cost_usd.toFixed(6)}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Selected Trace Span Waterfall */}
        {selectedTrace && (
          <div className="mt-4 p-4 rounded-xl bg-slate-950 border border-blue-900/50 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <span className="text-xs font-semibold text-blue-400 uppercase tracking-wide">
                Trace Inspector • ID: {selectedTrace.trace_id.slice(0, 8)}...
              </span>
              <button
                onClick={() => setSelectedTrace(null)}
                className="text-slate-400 hover:text-slate-200 text-xs"
              >
                Close
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
              <div className="p-2 rounded bg-slate-900 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">Prompt Tokens</span>
                <span className="font-mono text-slate-200">{selectedTrace.prompt_tokens}</span>
              </div>
              <div className="p-2 rounded bg-slate-900 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">Completion Tokens</span>
                <span className="font-mono text-slate-200">{selectedTrace.completion_tokens}</span>
              </div>
              <div className="p-2 rounded bg-slate-900 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">Total Latency</span>
                <span className="font-mono text-amber-300">{selectedTrace.latency_ms} ms</span>
              </div>
              <div className="p-2 rounded bg-slate-900 border border-slate-800">
                <span className="text-slate-500 block text-[10px]">Estimated Cost</span>
                <span className="font-mono text-emerald-400 font-semibold">
                  ${selectedTrace.estimated_cost_usd.toFixed(6)}
                </span>
              </div>
            </div>

            <div>
              <span className="text-[11px] font-semibold text-slate-400 block mb-2">
                Execution Spans:
              </span>
              {selectedTrace.spans && selectedTrace.spans.length > 0 ? (
                <div className="space-y-1.5">
                  {selectedTrace.spans.map((s, idx) => {
                    const dur = s.duration_ms || (s.end_ms ? s.end_ms - s.start_ms : 0);
                    return (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-2 rounded bg-slate-900 border border-slate-800/80 text-xs"
                      >
                        <span className="font-mono text-slate-300">{s.name}</span>
                        <span className="font-mono text-amber-400 text-[11px]">
                          {dur > 0 ? `${dur.toFixed(1)} ms` : "< 1 ms"}
                        </span>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="text-xs text-slate-500 italic">
                  Single-span monolithic execution recorded.
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
