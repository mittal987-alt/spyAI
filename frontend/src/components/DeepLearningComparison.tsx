import { useState } from "react";
import axios from "axios";
import {
  GitCompare,
  Loader2,
  Trophy,
  Award,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Layers,
  Sparkles,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
} from "recharts";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

interface CandidateConfig {
  name: string;
  hidden_units: number;
  learning_rate: number;
  dropout: number;
  batch_size: number;
}

interface CandidateResult {
  rank: number;
  name: string;
  config: CandidateConfig;
  best_validation_loss: number;
  best_validation_accuracy: number;
  train_accuracy_at_best_epoch: number;
  accuracy_gap: number;
  best_epoch: number;
  epochs_completed: number;
  early_stopping_triggered: boolean;
  training_time_seconds: number;
}

interface ComparisonResponse {
  experiment: string;
  run_id: string;
  created_at: string;
  ranking_criteria: string;
  models_compared: number;
  best_model: CandidateResult;
  leaderboard: CandidateResult[];
  limitations: string[];
}

export default function DeepLearningComparison() {
  const [epochs, setEpochs] = useState(15);
  const [samples, setSamples] = useState(1500);
  const [features, setFeatures] = useState(10);
  const [randomState, setRandomState] = useState(42);
  const [result, setResult] = useState<ComparisonResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showRawJson, setShowRawJson] = useState(false);
  const [selectedCandidate, setSelectedCandidate] = useState<string | null>(null);
  const [chartMetric, setChartMetric] = useState<"accuracy" | "loss" | "efficiency">(
    "accuracy",
  );

  function applyPreset(preset: {
    epochs: number;
    samples: number;
    features: number;
  }) {
    setEpochs(preset.epochs);
    setSamples(preset.samples);
    setFeatures(preset.features);
  }

  async function compareModels(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const response = await axios.post<ComparisonResponse>(
        `${API_BASE}/api/v1/deep-learning/compare`,
        {
          epochs,
          n_samples: samples,
          n_features: features,
          random_state: randomState,
        },
      );

      setResult(response.data);
      if (response.data.leaderboard.length > 0) {
        setSelectedCandidate(response.data.leaderboard[0].name);
      }
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(
          typeof detail === "string"
            ? detail
            : `Model comparison failed (${err.response?.status ?? "network error"}).`,
        );
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  }

  function copyJson() {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  // Format data for Recharts
  const chartData = result?.leaderboard.map((item) => ({
    name: formatModelName(item.name),
    rawName: item.name,
    rank: item.rank,
    val_accuracy: Number((item.best_validation_accuracy * 100).toFixed(1)),
    train_accuracy: Number((item.train_accuracy_at_best_epoch * 100).toFixed(1)),
    accuracy_gap: Number((item.accuracy_gap * 100).toFixed(1)),
    val_loss: Number(item.best_validation_loss.toFixed(4)),
    training_time: Number(item.training_time_seconds.toFixed(2)),
    best_epoch: item.best_epoch,
  })) ?? [];

  return (
    <section className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5 sm:p-6 shadow-xl">
      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-cyan-500/10 p-3 text-cyan-400 border border-cyan-500/20 shadow-inner">
            <GitCompare size={24} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-white tracking-tight">
                Neural Architecture Arena & Comparison Leaderboard
              </h2>
              <span className="rounded-full bg-cyan-400/10 px-2 py-0.5 text-xs font-semibold text-cyan-300 border border-cyan-400/20">
                Parallel Candidates
              </span>
            </div>
            <p className="mt-1 text-sm text-slate-400">
              Benchmark baseline vs larger capacity, adaptive learning rates, and higher dropout configurations on identical synthetic distributions.
            </p>
          </div>
        </div>

        {/* Quick Presets */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-slate-500">Presets:</span>
          <button
            type="button"
            onClick={() =>
              applyPreset({ epochs: 8, samples: 800, features: 8 })
            }
            className="rounded-lg border border-slate-700 bg-slate-900/80 px-2.5 py-1 text-slate-300 hover:border-cyan-500/50 hover:text-white transition-colors cursor-pointer"
          >
            ⚡ Fast Check (8 ep)
          </button>
          <button
            type="button"
            onClick={() =>
              applyPreset({ epochs: 15, samples: 1500, features: 12 })
            }
            className="rounded-lg border border-slate-700 bg-slate-900/80 px-2.5 py-1 text-slate-300 hover:border-cyan-500/50 hover:text-white transition-colors cursor-pointer"
          >
            🎯 Standard Arena (15 ep)
          </button>
          <button
            type="button"
            onClick={() =>
              applyPreset({ epochs: 25, samples: 2500, features: 20 })
            }
            className="rounded-lg border border-slate-700 bg-slate-900/80 px-2.5 py-1 text-slate-300 hover:border-cyan-500/50 hover:text-white transition-colors cursor-pointer"
          >
            🔬 Deep Benchmark (25 ep)
          </button>
        </div>
      </div>

      {/* Form Controls */}
      <form onSubmit={compareModels} className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <NumberField
            label="Epochs per Candidate"
            value={epochs}
            min={2}
            max={80}
            onChange={setEpochs}
          />
          <NumberField
            label="Synthetic Samples"
            value={samples}
            min={200}
            max={10000}
            step={100}
            onChange={setSamples}
          />
          <NumberField
            label="Feature Count"
            value={features}
            min={4}
            max={60}
            onChange={setFeatures}
          />
          <NumberField
            label="Random Seed"
            value={randomState}
            min={0}
            max={2147483647}
            onChange={setRandomState}
          />
        </div>

        <div className="flex items-center justify-between pt-1">
          <p className="text-xs text-slate-500 hidden sm:block">
            Trains 4 architecture variants in parallel runs logged to MLflow
          </p>

          <button
            type="submit"
            disabled={loading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-5 py-2.5 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-500/20 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-50 transition-all cursor-pointer font-sans"
          >
            {loading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <GitCompare size={16} />
            )}
            {loading ? "Training & Evaluating Arena..." : "Start Comparison Arena"}
          </button>
        </div>
      </form>

      {/* Error Callout */}
      {error && (
        <div className="mt-5 flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">
          <AlertCircle size={18} className="shrink-0 mt-0.5 text-rose-400" />
          <div>
            <p className="font-semibold text-rose-300">Arena Execution Failed</p>
            <p className="text-xs text-rose-200/90 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Arena Results Dashboard */}
      {result && (
        <div className="mt-8 space-y-6 border-t border-slate-800/80 pt-6">
          {/* Header Summary */}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-slate-900/60 p-4 border border-slate-800">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">
                  Arena Experiment Run
                </span>
                <code className="rounded bg-slate-800 px-2 py-0.5 text-xs text-cyan-300 font-mono">
                  {result.run_id}
                </code>
                <span className="rounded-full bg-cyan-500/15 border border-cyan-500/30 px-2.5 py-0.5 text-[11px] font-semibold text-cyan-300">
                  {result.models_compared} Candidates Evaluated
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Ranking criteria: {result.ranking_criteria}
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={copyJson}
                className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs text-slate-300 hover:text-white hover:bg-slate-700 transition-colors cursor-pointer"
              >
                {copied ? (
                  <Check size={13} className="text-emerald-400" />
                ) : (
                  <Copy size={13} />
                )}
                {copied ? "Copied" : "Copy Report"}
              </button>
              <button
                type="button"
                onClick={() => setShowRawJson(!showRawJson)}
                className="flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs text-slate-300 hover:text-white transition-colors cursor-pointer"
              >
                {showRawJson ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                JSON
              </button>
            </div>
          </div>

          {/* Winner Crown Podium Card */}
          {result.best_model && (
            <div className="relative overflow-hidden rounded-2xl border-2 border-amber-500/40 bg-gradient-to-r from-amber-500/10 via-slate-900/90 to-cyan-500/10 p-5 sm:p-6 shadow-2xl">
              <div className="absolute top-0 right-0 p-4 opacity-10 pointer-events-none">
                <Trophy size={140} className="text-amber-400" />
              </div>

              <div className="relative z-10 flex flex-wrap items-center justify-between gap-4">
                <div className="space-y-1.5">
                  <div className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/20 border border-amber-500/40 px-3 py-1 text-xs font-bold text-amber-300">
                    <Trophy size={14} className="text-amber-400" />
                    <span>#1 CHAMPION ARCHITECTURE</span>
                  </div>
                  <h3 className="text-2xl font-bold text-white capitalize tracking-tight flex items-center gap-2">
                    {formatModelName(result.best_model.name)}
                    <span className="text-xs font-normal text-slate-400">
                      ({result.best_model.config.hidden_units} hidden units, lr {result.best_model.config.learning_rate}, dropout {result.best_model.config.dropout})
                    </span>
                  </h3>
                  <p className="text-xs text-slate-300">
                    Achieved lowest validation loss of{" "}
                    <strong className="text-cyan-300 font-mono">
                      {result.best_model.best_validation_loss.toFixed(4)}
                    </strong>{" "}
                    at checkpoint epoch {result.best_model.best_epoch}.
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <div className="rounded-xl bg-slate-950/70 border border-amber-500/30 p-3 text-center min-w-[110px]">
                    <span className="text-[10px] uppercase font-semibold text-slate-400">
                      Val Accuracy
                    </span>
                    <div className="text-2xl font-black text-amber-300 font-mono">
                      {(result.best_model.best_validation_accuracy * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-950/70 border border-slate-800 p-3 text-center min-w-[110px]">
                    <span className="text-[10px] uppercase font-semibold text-slate-400">
                      Val Loss
                    </span>
                    <div className="text-xl font-bold text-white font-mono">
                      {result.best_model.best_validation_loss.toFixed(4)}
                    </div>
                  </div>
                  <div className="rounded-xl bg-slate-950/70 border border-slate-800 p-3 text-center min-w-[110px]">
                    <span className="text-[10px] uppercase font-semibold text-slate-400">
                      Accuracy Gap
                    </span>
                    <div className="text-xl font-bold text-emerald-400 font-mono">
                      {(result.best_model.accuracy_gap * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Leaderboard Table */}
          <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold text-white">
                  Candidate Architecture Leaderboard
                </h4>
                <p className="text-xs text-slate-400">
                  Select a candidate to view configuration details and hyperparameters
                </p>
              </div>
            </div>

            <div className="overflow-x-auto rounded-lg border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3">Rank</th>
                    <th className="py-2.5 px-3">Architecture Variant</th>
                    <th className="py-2.5 px-3">Hidden / LR / Dropout</th>
                    <th className="py-2.5 px-3">Best Val Accuracy</th>
                    <th className="py-2.5 px-3">Best Val Loss</th>
                    <th className="py-2.5 px-3">Accuracy Gap</th>
                    <th className="py-2.5 px-3">Best Epoch</th>
                    <th className="py-2.5 px-3">Latency</th>
                    <th className="py-2.5 px-3">Early Stop</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {result.leaderboard.map((candidate) => {
                    const isSelected = selectedCandidate === candidate.name;
                    const isWinner = candidate.rank === 1;

                    return (
                      <tr
                        key={candidate.name}
                        onClick={() => setSelectedCandidate(candidate.name)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? "bg-cyan-500/10 text-white font-medium"
                            : "hover:bg-slate-900/40 text-slate-300"
                        }`}
                      >
                        <td className="py-2.5 px-3">
                          <RankBadge rank={candidate.rank} />
                        </td>
                        <td className="py-2.5 px-3 font-sans font-semibold text-white">
                          <div className="flex items-center gap-1.5">
                            <span>{formatModelName(candidate.name)}</span>
                            {isWinner && (
                              <Award size={14} className="text-amber-400 shrink-0" />
                            )}
                          </div>
                        </td>
                        <td className="py-2.5 px-3 text-slate-400">
                          {candidate.config.hidden_units}u • lr {candidate.config.learning_rate} • {candidate.config.dropout} drop
                        </td>
                        <td className="py-2.5 px-3">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-white">
                              {(candidate.best_validation_accuracy * 100).toFixed(1)}%
                            </span>
                            <div className="h-1.5 w-16 rounded-full bg-slate-800 overflow-hidden hidden md:block">
                              <div
                                className={`h-full rounded-full ${
                                  isWinner ? "bg-amber-400" : "bg-cyan-400"
                                }`}
                                style={{
                                  width: `${candidate.best_validation_accuracy * 100}%`,
                                }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="py-2.5 px-3 text-cyan-300 font-semibold">
                          {candidate.best_validation_loss.toFixed(4)}
                        </td>
                        <td className="py-2.5 px-3">
                          <span
                            className={
                              candidate.accuracy_gap > 0.15
                                ? "text-rose-400 font-semibold"
                                : "text-emerald-400"
                            }
                          >
                            {(candidate.accuracy_gap * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-slate-300">
                          {candidate.best_epoch} / {candidate.epochs_completed}
                        </td>
                        <td className="py-2.5 px-3 text-slate-400">
                          {candidate.training_time_seconds.toFixed(2)}s
                        </td>
                        <td className="py-2.5 px-3 font-sans text-[11px]">
                          {candidate.early_stopping_triggered ? (
                            <span className="rounded bg-amber-500/15 border border-amber-500/30 px-1.5 py-0.5 text-amber-300">
                              Yes
                            </span>
                          ) : (
                            <span className="text-slate-500">No</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Visual Benchmark Charts */}
          <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h4 className="text-sm font-semibold text-white">
                  Comparative Performance Visualizer
                </h4>
                <p className="text-xs text-slate-400">
                  Contrast candidate accuracy, loss bounds, and training efficiency
                </p>
              </div>

              {/* Metric Selector Buttons */}
              <div className="flex items-center gap-1.5 rounded-lg bg-slate-900 p-1 border border-slate-800 text-xs">
                <button
                  type="button"
                  onClick={() => setChartMetric("accuracy")}
                  className={`rounded-md px-3 py-1 font-semibold transition-colors cursor-pointer ${
                    chartMetric === "accuracy"
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  Accuracy & Gap
                </button>
                <button
                  type="button"
                  onClick={() => setChartMetric("loss")}
                  className={`rounded-md px-3 py-1 font-semibold transition-colors cursor-pointer ${
                    chartMetric === "loss"
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  Validation Loss
                </button>
                <button
                  type="button"
                  onClick={() => setChartMetric("efficiency")}
                  className={`rounded-md px-3 py-1 font-semibold transition-colors cursor-pointer ${
                    chartMetric === "efficiency"
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  Training Latency
                </button>
              </div>
            </div>

            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                {chartMetric === "accuracy" ? (
                  <BarChart
                    data={chartData}
                    margin={{ top: 10, right: 20, left: -10, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
                    <YAxis stroke="#64748b" fontSize={11} domain={[0, 100]} tickLine={false} />
                    <Tooltip
                      formatter={(val) => [`${val}%`, ""]}
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        border: "1px solid #334155",
                        borderRadius: "8px",
                        fontSize: "12px",
                      }}
                    />
                    <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: "12px" }} />
                    <Bar
                      dataKey="val_accuracy"
                      name="Validation Accuracy (%)"
                      fill="#38bdf8"
                      radius={[4, 4, 0, 0]}
                    />
                    <Bar
                      dataKey="train_accuracy"
                      name="Train Accuracy (%)"
                      fill="#34d399"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                ) : chartMetric === "loss" ? (
                  <BarChart
                    data={chartData}
                    margin={{ top: 10, right: 20, left: -10, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
                    <YAxis stroke="#64748b" fontSize={11} tickLine={false} />
                    <Tooltip
                      formatter={(val) => [Number(val).toFixed(4), "Val Loss"]}
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        border: "1px solid #334155",
                        borderRadius: "8px",
                        fontSize: "12px",
                      }}
                    />
                    <Bar
                      dataKey="val_loss"
                      name="Best Validation Loss"
                      radius={[4, 4, 0, 0]}
                    >
                      {chartData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.rank === 1 ? "#f59e0b" : "#818cf8"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                ) : (
                  <BarChart
                    data={chartData}
                    margin={{ top: 10, right: 20, left: -10, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} />
                    <YAxis stroke="#64748b" fontSize={11} tickLine={false} />
                    <Tooltip
                      formatter={(val) => [`${val}s`, "Training Time"]}
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        border: "1px solid #334155",
                        borderRadius: "8px",
                        fontSize: "12px",
                      }}
                    />
                    <Bar
                      dataKey="training_time"
                      name="Training Time (seconds)"
                      fill="#a78bfa"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                )}
              </ResponsiveContainer>
            </div>
          </div>

          {/* Drilldown on Selected Candidate */}
          {selectedCandidate && (() => {
            const cand = result.leaderboard.find((c) => c.name === selectedCandidate);
            if (!cand) return null;

            return (
              <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers size={16} className="text-cyan-400" />
                    <h4 className="text-sm font-semibold text-white">
                      Selected Candidate Spec: {formatModelName(cand.name)}
                    </h4>
                  </div>
                  <RankBadge rank={cand.rank} />
                </div>

                <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4 text-xs">
                  <div className="rounded-lg bg-slate-900/80 p-3 border border-slate-800">
                    <span className="text-slate-400">Network Structure</span>
                    <p className="text-sm font-bold text-white mt-1">
                      {features} → {cand.config.hidden_units} → {Math.floor(cand.config.hidden_units / 2)} → 2
                    </p>
                  </div>
                  <div className="rounded-lg bg-slate-900/80 p-3 border border-slate-800">
                    <span className="text-slate-400">Regularization & LR</span>
                    <p className="text-sm font-bold text-white mt-1">
                      Dropout: {cand.config.dropout} • LR: {cand.config.learning_rate}
                    </p>
                  </div>
                  <div className="rounded-lg bg-slate-900/80 p-3 border border-slate-800">
                    <span className="text-slate-400">Epoch Checkpoint</span>
                    <p className="text-sm font-bold text-white mt-1">
                      Epoch {cand.best_epoch} (of {cand.epochs_completed})
                    </p>
                  </div>
                  <div className="rounded-lg bg-slate-900/80 p-3 border border-slate-800">
                    <span className="text-slate-400">Convergence & Speed</span>
                    <p className="text-sm font-bold text-white mt-1">
                      {cand.training_time_seconds.toFixed(2)}s • {cand.early_stopping_triggered ? "Early Stopped" : "Full Run"}
                    </p>
                  </div>
                </div>
              </div>
            );
          })()}

          {/* Limitations & Methodology Callout */}
          <div className="rounded-xl border border-slate-800 bg-slate-950 p-4 space-y-2 text-xs text-slate-400">
            <div className="flex items-center gap-2 font-semibold text-slate-300">
              <Sparkles size={14} className="text-cyan-400" />
              <span>Diagnostic Arena Methodology & Limitations</span>
            </div>
            <ul className="list-disc pl-5 space-y-1 text-slate-400">
              {result.limitations.map((lim, idx) => (
                <li key={idx}>{lim}</li>
              ))}
            </ul>
          </div>

          {/* Collapsible Raw JSON Block */}
          {showRawJson && (
            <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
              <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-300">Raw Comparison JSON Report</span>
                <span>{JSON.stringify(result).length} bytes</span>
              </div>
              <pre className="max-h-[350px] overflow-auto rounded-lg bg-slate-900/90 p-4 text-xs font-mono text-cyan-300 leading-5">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function formatModelName(name: string): string {
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function RankBadge({ rank }: { rank: number }) {
  switch (rank) {
    case 1:
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/20 border border-amber-500/40 px-2 py-0.5 text-[10px] font-black text-amber-300">
          🥇 #1
        </span>
      );
    case 2:
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-slate-400/20 border border-slate-400/30 px-2 py-0.5 text-[10px] font-bold text-slate-300">
          🥈 #2
        </span>
      );
    case 3:
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-amber-700/20 border border-amber-700/30 px-2 py-0.5 text-[10px] font-bold text-amber-500">
          🥉 #3
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1 rounded-full bg-slate-800 border border-slate-700 px-2 py-0.5 text-[10px] font-medium text-slate-400">
          #{rank}
        </span>
      );
  }
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block text-xs font-medium text-slate-300">
      {label}
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        required
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-900/90 p-2 text-xs text-white outline-none focus:border-cyan-400 transition-colors"
      />
    </label>
  );
}
