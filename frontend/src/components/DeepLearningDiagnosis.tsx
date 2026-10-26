import { useState } from "react";
import axios from "axios";
import {
  BrainCircuit,
  Loader2,
  Play,
  AlertTriangle,
  CheckCircle2,
  TrendingDown,
  Activity,
  Layers,
  BarChart3,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  ShieldAlert,
  Flame,
  Info,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  BarChart,
  Bar,
  Cell,
} from "recharts";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

interface DiagnosisConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  n_samples: number;
  n_features: number;
  hidden_units: number;
  dropout: number;
  random_state: number;
  explain: boolean;
}

interface ClassMetric {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

interface Finding {
  type: string;
  severity: "high" | "medium" | "low" | string;
  message: string;
  recommendation: string;
  class?: string;
}

interface LocalFeatureContribution {
  feature: string;
  shap_value: number;
  direction: "positive" | "negative" | "neutral" | string;
}

interface LocalShapSample {
  sample_index: number;
  class_explanations: Array<{
    class_index: number;
    top_features: LocalFeatureContribution[];
  }>;
}

interface DiagnosisResult {
  experiment: string;
  run_id: string;
  created_at: string;
  dataset: {
    n_samples: number;
    n_features: number;
    class_ratio: number;
    train_size: number;
    val_size: number;
  };
  config: {
    epochs: number;
    batch_size: number;
    learning_rate: number;
    hidden_units: number;
    dropout: number;
    random_state: number;
  };
  training_summary: {
    epochs_completed: number;
    best_epoch: number;
    best_validation_loss: number;
    final_validation_loss: number;
    final_validation_accuracy: number;
    training_time_seconds: number;
    early_stopping_triggered: boolean;
  };
  training_history: {
    train_loss: number[];
    val_loss: number[];
    train_accuracy: number[];
    val_accuracy: number[];
  };
  diagnostics: {
    final_train_loss: number;
    final_validation_loss: number;
    final_train_accuracy: number;
    final_validation_accuracy: number;
    accuracy_gap: number;
    best_validation_loss: number;
    best_validation_epoch: number;
    overfitting_detected: boolean;
    training_instability_detected: boolean;
    convergence_status: string;
    findings: Finding[];
    prediction_analysis: {
      sample_count: number;
      class_count: number;
      accuracy: number;
      confusion_matrix: number[][];
      class_metrics: Record<string, ClassMetric>;
      confidence: {
        mean: number;
        median: number;
        minimum: number;
        maximum: number;
        low_confidence_count: number;
        high_confidence_threshold: number;
      };
      errors: {
        error_count: number;
        error_rate: number;
        high_confidence_error_count: number;
        incorrect_sample_indices: number[];
        high_confidence_error_indices: number[];
      };
    };
    prediction_findings: Finding[];
    explainability?: {
      method: string;
      background_sample_count: number;
      explained_sample_count: number;
      feature_count: number;
      global_feature_importance: Array<{
        feature: string;
        mean_absolute_shap: number;
      }>;
      local_explanations: LocalShapSample[];
    } | null;
  };
}

const initialConfig: DiagnosisConfig = {
  epochs: 15,
  batch_size: 64,
  learning_rate: 0.001,
  n_samples: 2000,
  n_features: 16,
  hidden_units: 64,
  dropout: 0.2,
  random_state: 42,
  explain: true,
};

export default function DeepLearningDiagnosis() {
  const [config, setConfig] = useState<DiagnosisConfig>(initialConfig);
  const [result, setResult] = useState<DiagnosisResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<
    "trajectories" | "confusion" | "explainability" | "findings"
  >("trajectories");
  const [copied, setCopied] = useState(false);
  const [showRawJson, setShowRawJson] = useState(false);
  const [selectedSampleIndex, setSelectedSampleIndex] = useState(0);

  function updateConfig<K extends keyof DiagnosisConfig>(
    key: K,
    value: DiagnosisConfig[K],
  ) {
    setConfig((current) => ({ ...current, [key]: value }));
  }

  function applyPreset(
    preset: Partial<DiagnosisConfig> & { label: string },
  ) {
    setConfig((current) => ({ ...current, ...preset }));
  }

  async function runDiagnosis(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const response = await axios.post<DiagnosisResult>(
        `${API_BASE}/api/v1/deep-learning/diagnose`,
        config,
      );

      setResult(response.data);
      setSelectedSampleIndex(0);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(
          typeof detail === "string"
            ? detail
            : `Diagnosis failed (${err.response?.status ?? "network error"}).`,
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

  // Format trajectory data for Recharts
  const trajectoryData = result?.training_history
    ? result.training_history.train_loss.map((loss, idx) => ({
        epoch: idx + 1,
        train_loss: Number(loss.toFixed(4)),
        val_loss: Number(
          (result.training_history.val_loss[idx] ?? 0).toFixed(4),
        ),
        train_accuracy: Number(
          ((result.training_history.train_accuracy[idx] ?? 0) * 100).toFixed(
            2,
          ),
        ),
        val_accuracy: Number(
          ((result.training_history.val_accuracy[idx] ?? 0) * 100).toFixed(2),
        ),
        gap: Number(
          Math.abs(
            (result.training_history.val_loss[idx] ?? 0) - loss,
          ).toFixed(4),
        ),
      }))
    : [];

  const allFindings = [
    ...(result?.diagnostics?.findings ?? []),
    ...(result?.diagnostics?.prediction_findings ?? []),
  ];

  return (
    <section className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5 sm:p-6 shadow-xl">
      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-violet-500/10 p-3 text-violet-400 border border-violet-500/20 shadow-inner">
            <BrainCircuit size={24} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-white tracking-tight">
                Deep Learning Visual Diagnostics Studio
              </h2>
              <span className="rounded-full bg-violet-400/10 px-2 py-0.5 text-xs font-semibold text-violet-300 border border-violet-400/20">
                PyTorch + SHAP
              </span>
            </div>
            <p className="mt-1 text-sm text-slate-400">
              Train neural network architectures, inspect loss & accuracy
              trajectories, slice error confusion matrices, and compute SHAP
              feature attributions.
            </p>
          </div>
        </div>

        {/* Quick Presets */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-slate-500">Presets:</span>
          <button
            type="button"
            onClick={() =>
              applyPreset({
                epochs: 10,
                n_samples: 1200,
                n_features: 12,
                dropout: 0.2,
                label: "Fast",
              })
            }
            className="rounded-lg border border-slate-700 bg-slate-900/80 px-2.5 py-1 text-slate-300 hover:border-violet-500/50 hover:text-white transition-colors"
          >
            ⚡ Fast Check
          </button>
          <button
            type="button"
            onClick={() =>
              applyPreset({
                epochs: 20,
                n_samples: 3000,
                n_features: 16,
                dropout: 0.25,
                label: "Balanced",
              })
            }
            className="rounded-lg border border-slate-700 bg-slate-900/80 px-2.5 py-1 text-slate-300 hover:border-violet-500/50 hover:text-white transition-colors"
          >
            🎯 Standard Diagnostic
          </button>
          <button
            type="button"
            onClick={() =>
              applyPreset({
                epochs: 35,
                n_samples: 1000,
                n_features: 30,
                hidden_units: 128,
                dropout: 0.0,
                label: "Overfit",
              })
            }
            className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-amber-200 hover:border-amber-400 transition-colors"
            title="Small dataset with large capacity and zero dropout to test overfit detectors"
          >
            🔥 Induce Overfit
          </button>
        </div>
      </div>

      {/* Configuration Form */}
      <form onSubmit={runDiagnosis} className="space-y-5">
        <div className="grid gap-3.5 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-7">
          <NumberField
            label="Epochs"
            value={config.epochs}
            min={2}
            max={100}
            onChange={(value) => updateConfig("epochs", value)}
          />
          <NumberField
            label="Batch Size"
            value={config.batch_size}
            min={8}
            max={512}
            onChange={(value) => updateConfig("batch_size", value)}
          />
          <NumberField
            label="Learning Rate"
            value={config.learning_rate}
            min={0.00001}
            max={0.5}
            step={0.0005}
            onChange={(value) => updateConfig("learning_rate", value)}
          />
          <NumberField
            label="Samples"
            value={config.n_samples}
            min={200}
            max={20000}
            step={100}
            onChange={(value) => updateConfig("n_samples", value)}
          />
          <NumberField
            label="Features"
            value={config.n_features}
            min={4}
            max={100}
            onChange={(value) => updateConfig("n_features", value)}
          />
          <NumberField
            label="Hidden Units"
            value={config.hidden_units}
            min={8}
            max={256}
            onChange={(value) => updateConfig("hidden_units", value)}
          />
          <NumberField
            label="Dropout"
            value={config.dropout}
            min={0}
            max={0.9}
            step={0.05}
            onChange={(value) => updateConfig("dropout", value)}
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-4 pt-1">
          <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={config.explain}
              onChange={(event) =>
                updateConfig("explain", event.target.checked)
              }
              className="h-4 w-4 rounded border-slate-700 bg-slate-900 text-violet-500 focus:ring-violet-400"
            />
            <span className="font-medium">
              Compute SHAP GradientExplainer Feature Attributions
            </span>
            <span className="text-slate-500 text-[11px]">
              (captures global importance & sample-level drivers)
            </span>
          </label>

          <button
            type="submit"
            disabled={loading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-violet-500 to-indigo-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-violet-500/20 hover:from-violet-400 hover:to-indigo-500 disabled:opacity-50 transition-all cursor-pointer"
          >
            {loading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Play size={16} fill="currentColor" />
            )}
            {loading ? "Training & Diagnosing..." : "Run Diagnostic Studio"}
          </button>
        </div>
      </form>

      {/* Error Banner */}
      {error && (
        <div className="mt-5 flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">
          <AlertTriangle size={18} className="shrink-0 mt-0.5 text-rose-400" />
          <div>
            <p className="font-semibold text-rose-300">Diagnosis Execution Error</p>
            <p className="text-xs text-rose-200/90 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Results Dashboard */}
      {result && (
        <div className="mt-8 space-y-6 border-t border-slate-800/80 pt-6">
          {/* Executive Header Banner */}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-slate-900/60 p-4 border border-slate-800">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">
                  Run ID
                </span>
                <code className="rounded bg-slate-800 px-2 py-0.5 text-xs text-violet-300 font-mono">
                  {result.run_id}
                </code>
                {result.training_summary.early_stopping_triggered ? (
                  <span className="inline-flex items-center gap-1 rounded-full bg-amber-500/15 border border-amber-500/30 px-2.5 py-0.5 text-[11px] font-semibold text-amber-300">
                    <Flame size={12} /> Early Stopped @ Epoch{" "}
                    {result.training_summary.best_epoch}
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-0.5 text-[11px] font-semibold text-emerald-300">
                    <CheckCircle2 size={12} /> Full {result.training_summary.epochs_completed} Epochs
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-500">
                Trained in {result.training_summary.training_time_seconds.toFixed(2)}s •{" "}
                {result.dataset.train_size} train / {result.dataset.val_size} validation samples •{" "}
                {result.dataset.n_features} features
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={copyJson}
                className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs text-slate-300 hover:text-white hover:bg-slate-700 transition-colors"
              >
                {copied ? (
                  <Check size={13} className="text-emerald-400" />
                ) : (
                  <Copy size={13} />
                )}
                {copied ? "Copied" : "Copy Result"}
              </button>
              <button
                type="button"
                onClick={() => setShowRawJson(!showRawJson)}
                className="flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs text-slate-300 hover:text-white transition-colors"
              >
                {showRawJson ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                JSON
              </button>
            </div>
          </div>

          {/* Metric KPI Cards */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {/* KPI 1: Validation Accuracy */}
            <div className="rounded-xl border border-slate-800 bg-[#111a2e] p-4 shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Validation Accuracy</span>
                <CheckCircle2 size={16} className="text-emerald-400" />
              </div>
              <div className="mt-2 text-2xl font-bold text-white tracking-tight">
                {(
                  result.diagnostics.final_validation_accuracy * 100
                ).toFixed(1)}
                %
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800">
                <span>Train: {(result.diagnostics.final_train_accuracy * 100).toFixed(1)}%</span>
                <span
                  className={
                    result.diagnostics.accuracy_gap >= 0.1
                      ? "text-rose-400 font-semibold"
                      : "text-slate-400"
                  }
                >
                  Gap: {(result.diagnostics.accuracy_gap * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* KPI 2: Best Loss Checkpoint */}
            <div className="rounded-xl border border-slate-800 bg-[#111a2e] p-4 shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Best Validation Loss</span>
                <TrendingDown size={16} className="text-violet-400" />
              </div>
              <div className="mt-2 text-2xl font-bold text-white tracking-tight">
                {result.diagnostics.best_validation_loss.toFixed(4)}
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800">
                <span>Epoch {result.diagnostics.best_validation_epoch}</span>
                <span className="text-slate-400">
                  Final: {result.diagnostics.final_validation_loss.toFixed(4)}
                </span>
              </div>
            </div>

            {/* KPI 3: Overfitting & Convergence */}
            <div className="rounded-xl border border-slate-800 bg-[#111a2e] p-4 shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Convergence Status</span>
                <Activity size={16} className="text-cyan-400" />
              </div>
              <div className="mt-2 flex items-center gap-2">
                {result.diagnostics.overfitting_detected ? (
                  <span className="rounded-md bg-rose-500/20 border border-rose-500/30 px-2.5 py-1 text-sm font-semibold text-rose-300">
                    Overfitting Detected
                  </span>
                ) : (
                  <span className="rounded-md bg-emerald-500/20 border border-emerald-500/30 px-2.5 py-1 text-sm font-semibold text-emerald-300">
                    Stable Convergence
                  </span>
                )}
              </div>
              <div className="mt-2 text-xs text-slate-400 pt-2 border-t border-slate-800 truncate">
                State: <span className="text-slate-300 capitalize">{result.diagnostics.convergence_status.replace(/_/g, " ")}</span>
              </div>
            </div>

            {/* KPI 4: Reliability & Error Slices */}
            <div className="rounded-xl border border-slate-800 bg-[#111a2e] p-4 shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Mean Confidence</span>
                <ShieldAlert size={16} className="text-amber-400" />
              </div>
              <div className="mt-2 text-2xl font-bold text-white tracking-tight">
                {(
                  result.diagnostics.prediction_analysis.confidence.mean * 100
                ).toFixed(1)}
                %
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800">
                <span>
                  High-Conf Errors:{" "}
                  <strong
                    className={
                      result.diagnostics.prediction_analysis.errors
                        .high_confidence_error_count > 0
                        ? "text-rose-400"
                        : "text-slate-300"
                    }
                  >
                    {
                      result.diagnostics.prediction_analysis.errors
                        .high_confidence_error_count
                    }
                  </strong>
                </span>
                <span>
                  Low Conf:{" "}
                  {
                    result.diagnostics.prediction_analysis.confidence
                      .low_confidence_count
                  }
                </span>
              </div>
            </div>
          </div>

          {/* Studio Navigation Tabs */}
          <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-2">
            <TabButton
              active={activeTab === "trajectories"}
              onClick={() => setActiveTab("trajectories")}
              icon={<TrendingDown size={15} />}
              label="Learning Curves (Trajectories)"
            />
            <TabButton
              active={activeTab === "confusion"}
              onClick={() => setActiveTab("confusion")}
              icon={<Layers size={15} />}
              label="Confusion Matrix & Slices"
            />
            {result.diagnostics.explainability && (
              <TabButton
                active={activeTab === "explainability"}
                onClick={() => setActiveTab("explainability")}
                icon={<BarChart3 size={15} />}
                label="SHAP Feature Importance"
              />
            )}
            <TabButton
              active={activeTab === "findings"}
              onClick={() => setActiveTab("findings")}
              icon={<AlertTriangle size={15} />}
              label={`Findings & Actions (${allFindings.length})`}
              badge={allFindings.length > 0 ? allFindings.length : undefined}
            />
          </div>

          {/* TAB 1: LEARNING TRAJECTORIES */}
          {activeTab === "trajectories" && (
            <div className="space-y-6">
              <div className="grid gap-6 lg:grid-cols-2">
                {/* Chart 1: Loss curves */}
                <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div>
                      <h4 className="text-sm font-semibold text-white">
                        Loss Convergence Trajectory
                      </h4>
                      <p className="text-xs text-slate-400">
                        Binary Cross-Entropy Loss per epoch (Lower is better)
                      </p>
                    </div>
                    <span className="rounded bg-violet-500/10 border border-violet-500/20 px-2 py-0.5 text-[11px] text-violet-300">
                      Best: Epoch {result.diagnostics.best_validation_epoch}
                    </span>
                  </div>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={trajectoryData}
                        margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="#1e293b"
                          vertical={false}
                        />
                        <XAxis
                          dataKey="epoch"
                          stroke="#64748b"
                          fontSize={11}
                          tickLine={false}
                        />
                        <YAxis stroke="#64748b" fontSize={11} tickLine={false} />
                        <Tooltip content={<CustomTrajectoryTooltip />} />
                        <Legend
                          verticalAlign="top"
                          height={30}
                          wrapperStyle={{ fontSize: "12px" }}
                        />
                        <ReferenceLine
                          x={result.diagnostics.best_validation_epoch}
                          stroke="#8b5cf6"
                          strokeDasharray="4 4"
                          label={{
                            value: "Checkpoint",
                            fill: "#a78bfa",
                            fontSize: 10,
                            position: "top",
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="train_loss"
                          name="Train Loss"
                          stroke="#38bdf8"
                          strokeWidth={2}
                          dot={{ r: 2 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="val_loss"
                          name="Val Loss"
                          stroke="#f43f5e"
                          strokeWidth={2}
                          dot={{ r: 3 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Chart 2: Accuracy curves */}
                <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div>
                      <h4 className="text-sm font-semibold text-white">
                        Accuracy Progression
                      </h4>
                      <p className="text-xs text-slate-400">
                        Accuracy % across training & validation sets
                      </p>
                    </div>
                    <span className="rounded bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 text-[11px] text-emerald-300">
                      Gap: {(result.diagnostics.accuracy_gap * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={trajectoryData}
                        margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="#1e293b"
                          vertical={false}
                        />
                        <XAxis
                          dataKey="epoch"
                          stroke="#64748b"
                          fontSize={11}
                          tickLine={false}
                        />
                        <YAxis
                          stroke="#64748b"
                          fontSize={11}
                          tickLine={false}
                          domain={[0, 100]}
                        />
                        <Tooltip
                          formatter={(val) => [`${val}%`, ""]}
                          content={<CustomAccuracyTooltip />}
                        />
                        <Legend
                          verticalAlign="top"
                          height={30}
                          wrapperStyle={{ fontSize: "12px" }}
                        />
                        <Line
                          type="monotone"
                          dataKey="train_accuracy"
                          name="Train Accuracy (%)"
                          stroke="#34d399"
                          strokeWidth={2}
                          dot={{ r: 2 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="val_accuracy"
                          name="Val Accuracy (%)"
                          stroke="#818cf8"
                          strokeWidth={2}
                          dot={{ r: 3 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Epoch Details Data Table */}
              <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
                  Epoch History Breakdown
                </h4>
                <div className="max-h-48 overflow-y-auto rounded-lg border border-slate-800/80">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="sticky top-0 bg-slate-900 text-slate-400 border-b border-slate-800">
                      <tr>
                        <th className="py-2 px-3">Epoch</th>
                        <th className="py-2 px-3">Train Loss</th>
                        <th className="py-2 px-3">Val Loss</th>
                        <th className="py-2 px-3">Loss Gap</th>
                        <th className="py-2 px-3">Train Acc</th>
                        <th className="py-2 px-3">Val Acc</th>
                        <th className="py-2 px-3">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {trajectoryData.map((row) => {
                        const isBest =
                          row.epoch ===
                          result.diagnostics.best_validation_epoch;
                        return (
                          <tr
                            key={row.epoch}
                            className={
                              isBest
                                ? "bg-violet-500/10 text-white font-semibold"
                                : "hover:bg-slate-900/40"
                            }
                          >
                            <td className="py-1.5 px-3">
                              {row.epoch}{" "}
                              {isBest && (
                                <span className="ml-1 text-[10px] text-violet-400 font-sans">
                                  ★ Best
                                </span>
                              )}
                            </td>
                            <td className="py-1.5 px-3 text-sky-300">
                              {row.train_loss}
                            </td>
                            <td className="py-1.5 px-3 text-rose-300">
                              {row.val_loss}
                            </td>
                            <td className="py-1.5 px-3 text-slate-400">
                              {row.gap}
                            </td>
                            <td className="py-1.5 px-3 text-emerald-300">
                              {row.train_accuracy}%
                            </td>
                            <td className="py-1.5 px-3 text-indigo-300">
                              {row.val_accuracy}%
                            </td>
                            <td className="py-1.5 px-3 font-sans text-[11px]">
                              {isBest ? (
                                <span className="text-violet-300">Optimal checkpoint</span>
                              ) : row.gap > 0.2 ? (
                                <span className="text-amber-400">Diverging</span>
                              ) : (
                                <span className="text-slate-500">Normal</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: CONFUSION MATRIX & SLICES */}
          {activeTab === "confusion" && (
            <div className="space-y-6">
              <div className="grid gap-6 lg:grid-cols-2">
                {/* 2x2 Confusion Matrix Heatmap */}
                <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4">
                  <div className="mb-4">
                    <h4 className="text-sm font-semibold text-white">
                      Validation Confusion Matrix Heatmap
                    </h4>
                    <p className="text-xs text-slate-400">
                      Actual vs. Predicted classes on validation split (Total:{" "}
                      {result.diagnostics.prediction_analysis.sample_count} samples)
                    </p>
                  </div>

                  {(() => {
                    const matrix =
                      result.diagnostics.prediction_analysis.confusion_matrix;
                    const tn = matrix[0]?.[0] ?? 0;
                    const fp = matrix[0]?.[1] ?? 0;
                    const fn = matrix[1]?.[0] ?? 0;
                    const tp = matrix[1]?.[1] ?? 0;
                    const total = tn + fp + fn + tp || 1;

                    return (
                      <div className="space-y-3">
                        <div className="grid grid-cols-3 gap-2 text-center text-xs">
                          {/* Top-left header */}
                          <div className="flex items-center justify-center p-2 text-slate-500 text-[11px] font-semibold">
                            Actual \ Pred
                          </div>
                          <div className="rounded-lg bg-slate-900/90 p-2 font-semibold text-slate-300 border border-slate-800">
                            Pred Class 0
                          </div>
                          <div className="rounded-lg bg-slate-900/90 p-2 font-semibold text-slate-300 border border-slate-800">
                            Pred Class 1
                          </div>

                          {/* Row 1: Actual Class 0 */}
                          <div className="flex items-center justify-center rounded-lg bg-slate-900/90 p-2 font-semibold text-slate-300 border border-slate-800">
                            Actual Class 0
                          </div>
                          {/* TN */}
                          <div className="flex flex-col items-center justify-center rounded-xl border border-emerald-500/30 bg-emerald-500/15 p-4 transition-all hover:bg-emerald-500/25">
                            <span className="text-[10px] uppercase tracking-wider text-emerald-300 font-semibold">
                              True Negative (TN)
                            </span>
                            <span className="text-2xl font-bold text-white mt-1">
                              {tn}
                            </span>
                            <span className="text-[11px] text-emerald-400 font-mono">
                              {((tn / total) * 100).toFixed(1)}%
                            </span>
                          </div>
                          {/* FP */}
                          <div className="flex flex-col items-center justify-center rounded-xl border border-rose-500/30 bg-rose-500/15 p-4 transition-all hover:bg-rose-500/25">
                            <span className="text-[10px] uppercase tracking-wider text-rose-300 font-semibold">
                              False Positive (FP)
                            </span>
                            <span className="text-2xl font-bold text-white mt-1">
                              {fp}
                            </span>
                            <span className="text-[11px] text-rose-400 font-mono">
                              {((fp / total) * 100).toFixed(1)}%
                            </span>
                          </div>

                          {/* Row 2: Actual Class 1 */}
                          <div className="flex items-center justify-center rounded-lg bg-slate-900/90 p-2 font-semibold text-slate-300 border border-slate-800">
                            Actual Class 1
                          </div>
                          {/* FN */}
                          <div className="flex flex-col items-center justify-center rounded-xl border border-rose-500/30 bg-rose-500/15 p-4 transition-all hover:bg-rose-500/25">
                            <span className="text-[10px] uppercase tracking-wider text-rose-300 font-semibold">
                              False Negative (FN)
                            </span>
                            <span className="text-2xl font-bold text-white mt-1">
                              {fn}
                            </span>
                            <span className="text-[11px] text-rose-400 font-mono">
                              {((fn / total) * 100).toFixed(1)}%
                            </span>
                          </div>
                          {/* TP */}
                          <div className="flex flex-col items-center justify-center rounded-xl border border-emerald-500/30 bg-emerald-500/15 p-4 transition-all hover:bg-emerald-500/25">
                            <span className="text-[10px] uppercase tracking-wider text-emerald-300 font-semibold">
                              True Positive (TP)
                            </span>
                            <span className="text-2xl font-bold text-white mt-1">
                              {tp}
                            </span>
                            <span className="text-[11px] text-emerald-400 font-mono">
                              {((tp / total) * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>

                        <div className="flex items-center justify-between text-xs text-slate-400 px-1 pt-1">
                          <span>Total Errors: {fp + fn} ({(((fp + fn) / total) * 100).toFixed(1)}%)</span>
                          <span>Total Correct: {tn + tp} ({(((tn + tp) / total) * 100).toFixed(1)}%)</span>
                        </div>
                      </div>
                    );
                  })()}
                </div>

                {/* Per-Class Precision / Recall / F1 Table & Confidence Breakdown */}
                <div className="space-y-4">
                  <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4">
                    <h4 className="text-sm font-semibold text-white mb-2">
                      Class Performance Slices
                    </h4>
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="py-2 px-3">Class</th>
                          <th className="py-2 px-3">Precision</th>
                          <th className="py-2 px-3">Recall</th>
                          <th className="py-2 px-3">F1-Score</th>
                          <th className="py-2 px-3">Support</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 font-mono">
                        {Object.entries(
                          result.diagnostics.prediction_analysis.class_metrics,
                        ).map(([className, metrics]) => (
                          <tr key={className} className="hover:bg-slate-900/40">
                            <td className="py-2 px-3 font-semibold text-white">
                              Class {className}
                            </td>
                            <td className="py-2 px-3 text-sky-300">
                              {(metrics.precision * 100).toFixed(1)}%
                            </td>
                            <td className="py-2 px-3 text-indigo-300">
                              {(metrics.recall * 100).toFixed(1)}%
                            </td>
                            <td className="py-2 px-3 text-emerald-300 font-semibold">
                              {(metrics.f1_score * 100).toFixed(1)}%
                            </td>
                            <td className="py-2 px-3 text-slate-400">
                              {metrics.support}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Confidence Distribution Overview */}
                  <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4 space-y-3">
                    <h4 className="text-sm font-semibold text-white">
                      Softmax Confidence Distribution
                    </h4>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="rounded-lg bg-slate-900 p-2.5 border border-slate-800">
                        <span className="text-slate-400">Median Confidence</span>
                        <p className="text-base font-bold text-white mt-1">
                          {(
                            result.diagnostics.prediction_analysis.confidence
                              .median * 100
                          ).toFixed(1)}
                          %
                        </p>
                      </div>
                      <div className="rounded-lg bg-slate-900 p-2.5 border border-slate-800">
                        <span className="text-slate-400">Min / Max Range</span>
                        <p className="text-sm font-semibold text-slate-200 mt-1">
                          {(
                            result.diagnostics.prediction_analysis.confidence
                              .minimum * 100
                          ).toFixed(0)}
                          % -{" "}
                          {(
                            result.diagnostics.prediction_analysis.confidence
                              .maximum * 100
                          ).toFixed(0)}
                          %
                        </p>
                      </div>
                    </div>

                    {result.diagnostics.prediction_analysis.errors
                      .high_confidence_error_count > 0 && (
                      <div className="flex items-start gap-2.5 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-200">
                        <AlertTriangle
                          size={16}
                          className="text-rose-400 shrink-0 mt-0.5"
                        />
                        <div>
                          <p className="font-semibold text-rose-300">
                            High-Confidence Prediction Error Warning
                          </p>
                          <p className="text-rose-200/90 mt-0.5">
                            {
                              result.diagnostics.prediction_analysis.errors
                                .high_confidence_error_count
                            }{" "}
                            validation sample(s) were misclassified despite the
                            model assigning ≥ 90% confidence. Review dataset labels
                            for these sample indices: [
                            {result.diagnostics.prediction_analysis.errors.high_confidence_error_indices
                              .slice(0, 8)
                              .join(", ")}
                            {result.diagnostics.prediction_analysis.errors
                              .high_confidence_error_indices.length > 8
                              ? "..."
                              : ""}
                            ].
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: SHAP FEATURE IMPORTANCE */}
          {activeTab === "explainability" && result.diagnostics.explainability && (
            <div className="space-y-6">
              <div className="grid gap-6 lg:grid-cols-2">
                {/* Global Feature Importance Bar Chart */}
                <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4">
                  <div className="mb-4">
                    <h4 className="text-sm font-semibold text-white">
                      Global Feature Importance (SHAP GradientExplainer)
                    </h4>
                    <p className="text-xs text-slate-400">
                      Mean absolute SHAP value across {result.diagnostics.explainability.explained_sample_count} evaluated samples
                    </p>
                  </div>
                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={result.diagnostics.explainability.global_feature_importance.slice(
                          0,
                          10,
                        )}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="#1e293b"
                          horizontal={false}
                        />
                        <XAxis
                          type="number"
                          stroke="#64748b"
                          fontSize={11}
                          tickLine={false}
                        />
                        <YAxis
                          type="category"
                          dataKey="feature"
                          stroke="#94a3b8"
                          fontSize={11}
                          tickLine={false}
                        />
                        <Tooltip
                          formatter={(value) => [
                            Number(value).toFixed(4),
                            "Mean |SHAP|",
                          ]}
                          contentStyle={{
                            backgroundColor: "#0f172a",
                            border: "1px solid #334155",
                            borderRadius: "8px",
                            fontSize: "12px",
                          }}
                        />
                        <Bar
                          dataKey="mean_absolute_shap"
                          radius={[0, 4, 4, 0]}
                          name="Importance"
                        >
                          {result.diagnostics.explainability.global_feature_importance
                            .slice(0, 10)
                            .map((_, index) => (
                              <Cell
                                key={`cell-${index}`}
                                fill={
                                  index === 0
                                    ? "#a855f7"
                                    : index === 1
                                    ? "#8b5cf6"
                                    : index < 4
                                    ? "#6366f1"
                                    : "#3b82f6"
                                }
                              />
                            ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Local Sample Attribution Explorer */}
                <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4 flex flex-col">
                  <div className="mb-3 flex items-center justify-between">
                    <div>
                      <h4 className="text-sm font-semibold text-white">
                        Local Sample Attribution Explorer
                      </h4>
                      <p className="text-xs text-slate-400">
                        Inspect feature contributions pushing specific test samples
                      </p>
                    </div>

                    {/* Sample Selector */}
                    <div className="flex items-center gap-1.5 text-xs">
                      <span className="text-slate-400">Sample:</span>
                      <select
                        value={selectedSampleIndex}
                        onChange={(e) =>
                          setSelectedSampleIndex(Number(e.target.value))
                        }
                        className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-white outline-none focus:border-violet-500"
                      >
                        {result.diagnostics.explainability.local_explanations.map(
                          (sample) => (
                            <option
                              key={sample.sample_index}
                              value={sample.sample_index}
                            >
                              Sample #{sample.sample_index}
                            </option>
                          ),
                        )}
                      </select>
                    </div>
                  </div>

                  {(() => {
                    const sample =
                      result.diagnostics.explainability.local_explanations[
                        selectedSampleIndex
                      ] ??
                      result.diagnostics.explainability.local_explanations[0];
                    if (!sample) return null;

                    return (
                      <div className="flex-1 space-y-4">
                        {sample.class_explanations.map((classExp) => (
                          <div
                            key={classExp.class_index}
                            className="rounded-lg border border-slate-800/80 bg-slate-900/50 p-3 space-y-2"
                          >
                            <div className="flex items-center justify-between text-xs">
                              <span className="font-semibold text-slate-300">
                                Target Class {classExp.class_index} Decision Drivers
                              </span>
                              <span className="text-slate-500 text-[11px]">
                                Top {classExp.top_features.length} features
                              </span>
                            </div>

                            <div className="space-y-1.5">
                              {classExp.top_features.slice(0, 5).map((feat) => {
                                const isPos = feat.shap_value > 0;
                                return (
                                  <div
                                    key={feat.feature}
                                    className="flex items-center justify-between text-xs rounded bg-slate-950/60 px-2.5 py-1.5 border border-slate-800/60"
                                  >
                                    <span className="font-mono text-slate-300">
                                      {feat.feature}
                                    </span>
                                    <div className="flex items-center gap-2">
                                      <span
                                        className={`font-mono font-semibold ${
                                          isPos
                                            ? "text-emerald-400"
                                            : "text-rose-400"
                                        }`}
                                      >
                                        {isPos ? "+" : ""}
                                        {feat.shap_value.toFixed(4)}
                                      </span>
                                      <span
                                        className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
                                          isPos
                                            ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                                            : "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                                        }`}
                                      >
                                        {isPos ? "Supports" : "Contradicts"}
                                      </span>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    );
                  })()}
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: FINDINGS & RECOMMENDATIONS */}
          {activeTab === "findings" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-sm font-semibold text-white">
                    Automated Diagnostic Findings & Remediation Plan
                  </h4>
                  <p className="text-xs text-slate-400">
                    Rule-based heuristics and statistical boundary analysis for deep learning models
                  </p>
                </div>
                <span className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-300 font-medium border border-slate-700">
                  {allFindings.length} issue(s) identified
                </span>
              </div>

              {allFindings.length === 0 ? (
                <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-emerald-500/30 bg-emerald-500/5 p-8 text-center">
                  <CheckCircle2 size={32} className="text-emerald-400 mb-2" />
                  <h5 className="text-sm font-semibold text-white">
                    All Training Diagnostics Passed
                  </h5>
                  <p className="text-xs text-slate-400 mt-1 max-w-md">
                    No significant overfitting, gradient explosions, or high-confidence
                    misclassifications were detected for this experiment run.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {allFindings.map((finding, idx) => (
                    <div
                      key={idx}
                      className="rounded-xl border border-slate-800 bg-[#0b1120] p-4 space-y-2 hover:border-slate-700 transition-colors"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <SeverityBadge severity={finding.severity} />
                          <span className="font-semibold text-white text-sm">
                            {finding.type.replace(/_/g, " ").toUpperCase()}
                            {finding.class ? ` (Class ${finding.class})` : ""}
                          </span>
                        </div>
                      </div>

                      <p className="text-xs text-slate-300 leading-relaxed">
                        {finding.message}
                      </p>

                      <div className="rounded-lg bg-slate-900/80 p-3 border border-slate-800/80 flex items-start gap-2.5 text-xs text-slate-300">
                        <Info size={15} className="text-violet-400 shrink-0 mt-0.5" />
                        <div>
                          <strong className="text-violet-300">Recommendation: </strong>
                          {finding.recommendation}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Collapsible Raw JSON Block */}
          {showRawJson && (
            <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
              <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-300">Raw Diagnostic JSON Payload</span>
                <span>{JSON.stringify(result).length} bytes</span>
              </div>
              <pre className="max-h-[380px] overflow-auto rounded-lg bg-slate-900/90 p-4 text-xs font-mono text-emerald-300 leading-5">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function TabButton({
  active,
  onClick,
  icon,
  label,
  badge,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
  badge?: number;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-xs font-semibold transition-all cursor-pointer ${
        active
          ? "bg-violet-500/20 text-violet-300 border border-violet-500/40 shadow-sm"
          : "text-slate-400 hover:text-white hover:bg-slate-800/60"
      }`}
    >
      {icon}
      <span>{label}</span>
      {badge !== undefined && (
        <span className="ml-1 rounded-full bg-rose-500/20 px-1.5 py-0.2 text-[10px] text-rose-300 font-bold border border-rose-500/30">
          {badge}
        </span>
      )}
    </button>
  );
}

function SeverityBadge({ severity }: { severity: string }) {
  switch (severity.toLowerCase()) {
    case "high":
      return (
        <span className="rounded-full bg-rose-500/15 border border-rose-500/30 px-2 py-0.5 text-[10px] font-bold text-rose-400 uppercase">
          High Severity
        </span>
      );
    case "medium":
      return (
        <span className="rounded-full bg-amber-500/15 border border-amber-500/30 px-2 py-0.5 text-[10px] font-bold text-amber-300 uppercase">
          Medium Severity
        </span>
      );
    default:
      return (
        <span className="rounded-full bg-sky-500/15 border border-sky-500/30 px-2 py-0.5 text-[10px] font-bold text-sky-300 uppercase">
          Low Severity
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
        required
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-900/90 p-2 text-xs text-white outline-none focus:border-violet-400 transition-colors"
      />
    </label>
  );
}

interface TooltipPayloadItem {
  name: string;
  value: number;
  color: string;
}

function CustomTrajectoryTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: number;
}) {
  if (!active || !payload || !payload.length) return null;

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 p-2.5 text-xs shadow-xl space-y-1">
      <p className="font-semibold text-slate-200">Epoch {label}</p>
      {payload.map((item, index) => (
        <div key={index} className="flex items-center justify-between gap-4">
          <span style={{ color: item.color }}>{item.name}:</span>
          <span className="font-mono font-bold text-white">
            {Number(item.value).toFixed(4)}
          </span>
        </div>
      ))}
    </div>
  );
}

function CustomAccuracyTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: number;
}) {
  if (!active || !payload || !payload.length) return null;

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-950 p-2.5 text-xs shadow-xl space-y-1">
      <p className="font-semibold text-slate-200">Epoch {label}</p>
      {payload.map((item, index) => (
        <div key={index} className="flex items-center justify-between gap-4">
          <span style={{ color: item.color }}>{item.name}:</span>
          <span className="font-mono font-bold text-white">{item.value}%</span>
        </div>
      ))}
    </div>
  );
}
