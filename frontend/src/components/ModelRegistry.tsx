import { useEffect, useState, useCallback } from "react";
import {
  Boxes,
  Plus,
  ArrowRightLeft,
  Trash2,
  RefreshCw,
  Tag,
  Clock,
  Sparkles,
} from "lucide-react";
import {
  listRegisteredModels,
  registerModel,
  updateModelStage,
  deleteRegisteredModel,
  compareRegisteredModels,
} from "../api";
import type { RegisteredModel, ModelStage, ModelComparisonResult } from "../types";

const stageStyles: Record<ModelStage, { badge: string; dot: string; label: string }> = {
  production: {
    badge: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    dot: "bg-emerald-400",
    label: "Production",
  },
  staging: {
    badge: "border-indigo-500/30 bg-indigo-500/10 text-indigo-300",
    dot: "bg-indigo-400",
    label: "Staging",
  },
  development: {
    badge: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    dot: "bg-amber-400",
    label: "Development",
  },
  archived: {
    badge: "border-zinc-700 bg-zinc-800/60 text-zinc-400",
    dot: "bg-zinc-500",
    label: "Archived",
  },
};

export default function ModelRegistry() {
  const [models, setModels] = useState<RegisteredModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedStage, setSelectedStage] = useState<string>("all");
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showCompareModal, setShowCompareModal] = useState(false);

  // Form state
  const [name, setName] = useState("");
  const [version, setVersion] = useState("v1.0.0");
  const [stage, setStage] = useState<ModelStage>("development");
  const [framework, setFramework] = useState("sklearn");
  const [description, setDescription] = useState("");
  const [accuracy, setAccuracy] = useState("0.925");
  const [f1Score, setF1Score] = useState("0.918");
  const [latencyMs, setLatencyMs] = useState("45");
  const [tagsInput, setTagsInput] = useState("fraud-detection, production-ready");
  const [submitting, setSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Comparison state
  const [baselineId, setBaselineId] = useState<string>("");
  const [candidateId, setCandidateId] = useState<string>("");
  const [comparisonResult, setComparisonResult] = useState<ModelComparisonResult | null>(null);
  const [comparing, setComparing] = useState(false);

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listRegisteredModels();
      setModels(data);
    } catch (err: unknown) {
      console.error("Failed to load model registry", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    try {
      const metrics: Record<string, number> = {};
      if (accuracy) metrics.accuracy = parseFloat(accuracy);
      if (f1Score) metrics.f1_score = parseFloat(f1Score);
      if (latencyMs) metrics.latency_ms = parseFloat(latencyMs);

      const tags = tagsInput
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);

      await registerModel({
        name,
        version,
        stage,
        framework,
        description,
        metrics,
        tags,
      });

      setStatusMessage(`Model "${name}" (${version}) registered successfully.`);
      setShowCreateModal(false);
      setName("");
      setVersion("v1.0.1");
      setDescription("");
      void loadModels();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Registration failed";
      setStatusMessage(`Error: ${msg}`);
    } finally {
      setSubmitting(false);
      setTimeout(() => setStatusMessage(null), 5000);
    }
  };

  const handleStageChange = async (modelId: string, newStage: ModelStage) => {
    try {
      await updateModelStage(modelId, newStage);
      setStatusMessage(`Model transitioned to ${newStage.toUpperCase()}`);
      void loadModels();
    } catch (err: unknown) {
      console.error(err);
    }
  };

  const handleDelete = async (modelId: string, modelName: string, modelVer: string) => {
    if (!window.confirm(`Delete model ${modelName} (${modelVer})?`)) return;
    try {
      await deleteRegisteredModel(modelId);
      setStatusMessage(`Deleted ${modelName} (${modelVer})`);
      void loadModels();
    } catch (err: unknown) {
      console.error(err);
    }
  };

  const handleCompare = async () => {
    if (!baselineId || !candidateId) return;
    setComparing(true);
    try {
      const result = await compareRegisteredModels(baselineId, candidateId);
      setComparisonResult(result);
    } catch (err: unknown) {
      console.error(err);
    } finally {
      setComparing(false);
    }
  };

  const filteredModels = models.filter((m) => {
    if (selectedStage === "all") return true;
    return m.stage === selectedStage;
  });

  return (
    <div className="space-y-6">
      {/* Top Action Header */}
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h3 className="text-sm font-semibold text-white">Model Registry & Lifecycle Ledger</h3>
          <p className="text-xs text-zinc-400">
            Audit model lineage, promote champion models to production, and benchmark candidate versions.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowCompareModal(true)}
            className="flex items-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-white/20 hover:bg-white/[0.06] hover:text-white"
          >
            <ArrowRightLeft size={13} className="text-indigo-400" />
            <span>Diff Versions</span>
          </button>

          <button
            type="button"
            onClick={() => setShowCreateModal(true)}
            className="linear-button-primary flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold"
          >
            <Plus size={13} />
            <span>Register Model</span>
          </button>

          <button
            type="button"
            onClick={() => void loadModels()}
            disabled={loading}
            className="rounded-lg border border-white/[0.08] bg-white/[0.03] p-1.5 text-zinc-400 hover:text-white"
          >
            <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      {statusMessage && (
        <div className="rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-4 py-2 text-xs text-indigo-300">
          {statusMessage}
        </div>
      )}

      {/* Stage Filter Pills */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
        {["all", "production", "staging", "development", "archived"].map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setSelectedStage(s)}
            className={`rounded-full px-3 py-1 font-medium capitalize transition ${
              selectedStage === s
                ? "bg-zinc-800 text-white shadow-sm border border-zinc-700"
                : "text-zinc-500 hover:bg-white/[0.03] hover:text-zinc-300"
            }`}
          >
            {s} {s === "all" ? `(${models.length})` : `(${models.filter((m) => m.stage === s).length})`}
          </button>
        ))}
      </div>

      {/* Model Cards Grid */}
      {filteredModels.length === 0 ? (
        <div className="rounded-xl border border-dashed border-white/[0.08] p-12 text-center">
          <Boxes size={28} className="mx-auto text-zinc-600" />
          <p className="mt-3 text-xs font-medium text-zinc-300">No models found in registry</p>
          <p className="text-[11px] text-zinc-500">
            Register your first model version above or run a training pipeline to auto-register.
          </p>
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {filteredModels.map((m) => {
            const stageInfo = stageStyles[m.stage] || stageStyles.development;
            return (
              <div
                key={m.model_id}
                className="group relative flex flex-col justify-between rounded-xl border border-white/[0.07] bg-[#0c0e15] p-4 transition-all hover:border-white/20 hover:bg-[#10121b]"
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="text-sm font-semibold text-white group-hover:text-indigo-300">
                          {m.name}
                        </span>
                        <span className="rounded border border-white/[0.1] bg-white/[0.04] px-1.5 py-0.2 font-mono text-[10px] text-zinc-300">
                          {m.version}
                        </span>
                      </div>
                      <p className="mt-1 line-clamp-2 text-[11px] text-zinc-400">
                        {m.description || "No description provided."}
                      </p>
                    </div>

                    <span
                      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium ${stageInfo.badge}`}
                    >
                      <span className={`h-1.5 w-1.5 rounded-full ${stageInfo.dot}`} />
                      {stageInfo.label}
                    </span>
                  </div>

                  {/* Framework & Metrics */}
                  <div className="mt-3 flex items-center gap-2 text-[10px] font-mono text-zinc-400">
                    <span className="rounded bg-white/[0.04] px-1.5 py-0.5 text-zinc-300">
                      {m.framework.toUpperCase()}
                    </span>
                    {m.metrics && Object.entries(m.metrics).slice(0, 3).map(([k, v]) => (
                      <span key={k} className="border-l border-white/[0.08] pl-2">
                        {k}: <strong className="text-zinc-200">{typeof v === "number" ? v.toFixed(3) : v}</strong>
                      </span>
                    ))}
                  </div>

                  {/* Tags */}
                  {m.tags && m.tags.length > 0 && (
                    <div className="mt-2.5 flex flex-wrap items-center gap-1">
                      {m.tags.map((t) => (
                        <span
                          key={t}
                          className="inline-flex items-center gap-1 rounded bg-white/[0.03] px-1.5 py-0.2 text-[9px] text-zinc-400"
                        >
                          <Tag size={9} />
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Footer Controls */}
                <div className="mt-4 flex items-center justify-between border-t border-white/[0.05] pt-3 text-[11px]">
                  <span className="flex items-center gap-1 text-[10px] text-zinc-500">
                    <Clock size={11} />
                    {new Date(m.created_at).toLocaleDateString()}
                  </span>

                  <div className="flex items-center gap-1">
                    {/* Stage transition dropdown */}
                    <select
                      value={m.stage}
                      onChange={(e) => void handleStageChange(m.model_id, e.target.value as ModelStage)}
                      className="rounded border border-white/[0.08] bg-[#12141c] px-2 py-0.5 text-[10px] text-zinc-300 focus:outline-none"
                    >
                      <option value="development">Dev</option>
                      <option value="staging">Staging</option>
                      <option value="production">Production</option>
                      <option value="archived">Archive</option>
                    </select>

                    <button
                      type="button"
                      onClick={() => void handleDelete(m.model_id, m.name, m.version)}
                      title="Delete model version"
                      className="rounded p-1 text-zinc-500 hover:text-rose-400"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Register Model Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-xl border border-white/[0.1] bg-[#0e1017] p-6 shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
              <h4 className="text-sm font-semibold text-white">Register New Model Version</h4>
              <button
                type="button"
                onClick={() => setShowCreateModal(false)}
                className="text-zinc-500 hover:text-white"
              >
                ✕
              </button>
            </div>

            <form onSubmit={(e) => void handleRegister(e)} className="mt-4 space-y-3 text-xs">
              <div>
                <label className="block font-medium text-zinc-300">Model Name</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. credit-risk-xgboost"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="linear-input mt-1 w-full px-3 py-1.5"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-medium text-zinc-300">Version</label>
                  <input
                    type="text"
                    required
                    placeholder="v1.0.0"
                    value={version}
                    onChange={(e) => setVersion(e.target.value)}
                    className="linear-input mt-1 w-full px-3 py-1.5"
                  />
                </div>
                <div>
                  <label className="block font-medium text-zinc-300">Stage</label>
                  <select
                    value={stage}
                    onChange={(e) => setStage(e.target.value as ModelStage)}
                    className="linear-input mt-1 w-full px-3 py-1.5"
                  >
                    <option value="development">Development</option>
                    <option value="staging">Staging</option>
                    <option value="production">Production</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-medium text-zinc-300">Framework</label>
                  <select
                    value={framework}
                    onChange={(e) => setFramework(e.target.value)}
                    className="linear-input mt-1 w-full px-3 py-1.5"
                  >
                    <option value="sklearn">Scikit-Learn</option>
                    <option value="pytorch">PyTorch</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="transformers">HuggingFace</option>
                  </select>
                </div>
                <div>
                  <label className="block font-medium text-zinc-300">Accuracy</label>
                  <input
                    type="number"
                    step="0.001"
                    value={accuracy}
                    onChange={(e) => setAccuracy(e.target.value)}
                    className="linear-input mt-1 w-full px-3 py-1.5"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-medium text-zinc-300">F1 Score</label>
                  <input
                    type="number"
                    step="0.001"
                    value={f1Score}
                    onChange={(e) => setF1Score(e.target.value)}
                    className="linear-input mt-1 w-full px-3 py-1.5"
                  />
                </div>
                <div>
                  <label className="block font-medium text-zinc-300">Latency (ms)</label>
                  <input
                    type="number"
                    value={latencyMs}
                    onChange={(e) => setLatencyMs(e.target.value)}
                    className="linear-input mt-1 w-full px-3 py-1.5"
                  />
                </div>
              </div>

              <div>
                <label className="block font-medium text-zinc-300">Description</label>
                <textarea
                  rows={2}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Primary objective, training dataset date, notes..."
                  className="linear-input mt-1 w-full px-3 py-1.5"
                />
              </div>

              <div>
                <label className="block font-medium text-zinc-300">Tags (comma-separated)</label>
                <input
                  type="text"
                  value={tagsInput}
                  onChange={(e) => setTagsInput(e.target.value)}
                  className="linear-input mt-1 w-full px-3 py-1.5"
                />
              </div>

              <div className="flex justify-end gap-2 pt-3">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="linear-button-secondary rounded-lg px-3 py-1.5"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={submitting}
                  className="linear-button-primary rounded-lg px-4 py-1.5 font-semibold"
                >
                  {submitting ? "Registering..." : "Save to Registry"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Model Diff / Comparison Modal */}
      {showCompareModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm">
          <div className="w-full max-w-2xl rounded-xl border border-white/[0.1] bg-[#0e1017] p-6 shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/[0.08] pb-3">
              <div className="flex items-center gap-2">
                <Sparkles size={16} className="text-indigo-400" />
                <h4 className="text-sm font-semibold text-white">Compare Registered Model Versions</h4>
              </div>
              <button
                type="button"
                onClick={() => {
                  setShowCompareModal(false);
                  setComparisonResult(null);
                }}
                className="text-zinc-500 hover:text-white"
              >
                ✕
              </button>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-4 text-xs">
              <div>
                <label className="block font-medium text-zinc-300">Baseline (Champion)</label>
                <select
                  value={baselineId}
                  onChange={(e) => setBaselineId(e.target.value)}
                  className="linear-input mt-1 w-full px-3 py-1.5"
                >
                  <option value="">Select baseline model...</option>
                  {models.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.name} ({m.version}) [{m.stage}]
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block font-medium text-zinc-300">Candidate (Challenger)</label>
                <select
                  value={candidateId}
                  onChange={(e) => setCandidateId(e.target.value)}
                  className="linear-input mt-1 w-full px-3 py-1.5"
                >
                  <option value="">Select candidate model...</option>
                  {models.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.name} ({m.version}) [{m.stage}]
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="mt-4 flex justify-end">
              <button
                type="button"
                onClick={() => void handleCompare()}
                disabled={!baselineId || !candidateId || comparing}
                className="linear-button-primary rounded-lg px-4 py-1.5 text-xs font-semibold disabled:opacity-50"
              >
                {comparing ? "Comparing..." : "Run Diff Comparison"}
              </button>
            </div>

            {/* Comparison Results */}
            {comparisonResult && (
              <div className="mt-5 space-y-4 rounded-lg border border-white/[0.08] bg-[#0c0e15] p-4 text-xs">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-2">
                  <span className="font-semibold text-zinc-200">
                    {comparisonResult.baseline.name} ({comparisonResult.baseline.version}) vs.{" "}
                    {comparisonResult.candidate.name} ({comparisonResult.candidate.version})
                  </span>
                  <span className="font-mono text-[10px] text-zinc-500">
                    Δ Metrics Summary
                  </span>
                </div>

                <div className="grid gap-2 sm:grid-cols-3">
                  {Object.entries(comparisonResult.metric_deltas).map(([metricKey, d]) => (
                    <div
                      key={metricKey}
                      className="rounded border border-white/[0.06] bg-white/[0.02] p-2.5"
                    >
                      <span className="text-[10px] uppercase text-zinc-400">{metricKey}</span>
                      <div className="mt-1 flex items-baseline justify-between font-mono">
                        <span className="text-zinc-300">
                          {d.baseline ?? "—"} ➔ <strong className="text-white">{d.candidate ?? "—"}</strong>
                        </span>
                        <span
                          className={`text-[11px] font-semibold ${
                            d.improved ? "text-emerald-400" : d.improved === false ? "text-rose-400" : "text-zinc-400"
                          }`}
                        >
                          {d.percent_change != null
                            ? `${d.percent_change >= 0 ? "+" : ""}${d.percent_change.toFixed(1)}%`
                            : "—"}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>

                <p className="text-[11px] leading-relaxed text-zinc-400">
                  {comparisonResult.summary}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
