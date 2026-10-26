import { useEffect, useState } from "react";
import axios from "axios";
import {
  Zap,
  Loader2,
  Play,
  AlertTriangle,
  CheckCircle2,
  RotateCcw,
  Sliders,
  ShieldAlert,
  BarChart2,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Info,
} from "lucide-react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

interface PresetConfig {
  mean_shift: number;
  variance_scale: number;
  gaussian_noise: number;
  outlier_rate: number;
  outlier_scale?: number;
  label_flip_rate: number;
  target_features?: number[];
}

interface PresetItem {
  id: string;
  name: string;
  description: string;
  config: PresetConfig;
}

interface FeatureDrift {
  feature_index: number;
  feature_name: string;
  is_targeted_in_simulation: boolean;
  ks_statistic: number;
  raw_p_value: number;
  adjusted_p_value: number;
  drift_detected: boolean;
  psi_score: number;
  psi_risk: "stable" | "moderate" | "critical" | string;
  wasserstein_distance: number;
  baseline_mean: number;
  baseline_std: number;
  perturbed_mean: number;
  perturbed_std: number;
  mean_shift_magnitude: number;
}

interface DensityBin {
  bin_center: number;
  baseline_density: number;
  perturbed_density: number;
}

interface DistributionVisual {
  feature_name: string;
  feature_index: number;
  is_drifted: boolean;
  psi_score: number;
  density_bins: DensityBin[];
}

interface SimulationReport {
  status: string;
  dataset_metadata: {
    total_samples: number;
    train_size: number;
    test_size: number;
    feature_count: number;
    target_features_perturbed: number[];
  };
  applied_perturbations: {
    mean_shift: number;
    variance_scale: number;
    gaussian_noise: number;
    outlier_rate: number;
    outlier_scale: number;
    label_flip_rate: number;
  };
  model_stress_test: {
    model_type: string;
    baseline: {
      accuracy: number;
      f1_score: number;
      precision: number;
      recall: number;
    };
    perturbed: {
      accuracy: number;
      f1_score: number;
      precision: number;
      recall: number;
    };
    accuracy_drop: number;
    f1_drop: number;
    resilience_rating: "highly_resilient" | "minor_impact" | "moderate_degradation" | "critical_vulnerability" | string;
  };
  drift_summary: {
    total_features: number;
    drifted_features_count: number;
    critical_psi_features_count: number;
    drift_percentage: number;
    max_psi_score: number;
    max_wasserstein: number;
  };
  feature_drifts: FeatureDrift[];
  distribution_visuals: DistributionVisual[];
  recommendations: string[];
}

export default function DriftSimulator() {
  const [presets, setPresets] = useState<PresetItem[]>([]);
  const [activePresetId, setActivePresetId] = useState<string>("sensor_drift");

  // Simulation Parameters
  const [samples, setSamples] = useState(1500);
  const [features, setFeatures] = useState(8);
  const [meanShift, setMeanShift] = useState(1.5);
  const [varianceScale, setVarianceScale] = useState(1.2);
  const [gaussianNoise, setGaussianNoise] = useState(0.4);
  const [outlierRate, setOutlierRate] = useState(0.0);
  const [outlierScale, setOutlierScale] = useState(5.0);
  const [labelFlipRate, setLabelFlipRate] = useState(0.0);
  const [targetFeatures, setTargetFeatures] = useState<number[]>([0, 1]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [report, setReport] = useState<SimulationReport | null>(null);
  const [selectedFeaturePlot, setSelectedFeaturePlot] = useState<number>(0);
  const [copied, setCopied] = useState(false);
  const [showRawJson, setShowRawJson] = useState(false);

  // Load presets on mount
  useEffect(() => {
    async function fetchPresets() {
      try {
        const res = await axios.get<{ presets: PresetItem[] }>(
          `${API_BASE}/api/v1/drift/presets`,
        );
        setPresets(res.data.presets);
      } catch {
        // Fallback local presets
      }
    }
    void fetchPresets();
  }, []);

  function applyPreset(preset: PresetItem) {
    setActivePresetId(preset.id);
    const cfg = preset.config;
    setMeanShift(cfg.mean_shift);
    setVarianceScale(cfg.variance_scale);
    setGaussianNoise(cfg.gaussian_noise);
    setOutlierRate(cfg.outlier_rate);
    if (cfg.outlier_scale !== undefined) setOutlierScale(cfg.outlier_scale);
    setLabelFlipRate(cfg.label_flip_rate);
    if (cfg.target_features) setTargetFeatures(cfg.target_features);
  }

  function toggleTargetFeature(idx: number) {
    setTargetFeatures((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx].sort(),
    );
  }

  async function handleSimulate(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setReport(null);
    setLoading(true);

    try {
      const res = await axios.post<SimulationReport>(
        `${API_BASE}/api/v1/drift/simulate`,
        {
          n_samples: samples,
          n_features: features,
          mean_shift: meanShift,
          variance_scale: varianceScale,
          gaussian_noise: gaussianNoise,
          outlier_rate: outlierRate,
          outlier_scale: outlierScale,
          label_flip_rate: labelFlipRate,
          target_features: targetFeatures,
          random_state: 42,
        },
      );
      setReport(res.data);
      if (res.data.distribution_visuals.length > 0) {
        setSelectedFeaturePlot(res.data.distribution_visuals[0].feature_index);
      }
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(
          typeof detail === "string"
            ? detail
            : `Simulation failed (${err.response?.status ?? "network error"}).`,
        );
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  }

  function copyJson() {
    if (!report) return;
    navigator.clipboard.writeText(JSON.stringify(report, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const activeVisual = report?.distribution_visuals.find(
    (v) => v.feature_index === selectedFeaturePlot,
  ) ?? report?.distribution_visuals[0];

  return (
    <section className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5 sm:p-6 shadow-xl">
      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-5">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-amber-500/10 p-3 text-amber-400 border border-amber-500/20 shadow-inner">
            <Zap size={24} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-white tracking-tight">
                Interactive Synthetic Drift & Adversarial Perturbation Sandbox
              </h2>
              <span className="rounded-full bg-amber-400/10 px-2 py-0.5 text-xs font-semibold text-amber-300 border border-amber-400/20">
                Live Stress-Testing
              </span>
            </div>
            <p className="mt-1 text-sm text-slate-400">
              Inject controlled covariate shifts, variance expansion, noise bursts, outlier spikes, and concept label flips to benchmark model resilience.
            </p>
          </div>
        </div>

        {/* Preset Selector */}
        {presets.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="text-slate-500">Preset Scenarios:</span>
            {presets.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => applyPreset(p)}
                className={`rounded-lg px-2.5 py-1 transition-all cursor-pointer ${
                  activePresetId === p.id
                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 font-semibold"
                    : "border border-slate-700 bg-slate-900/80 text-slate-300 hover:border-slate-600 hover:text-white"
                }`}
                title={p.description}
              >
                {p.name}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Perturbation Parameter Controls */}
      <form onSubmit={handleSimulate} className="space-y-5">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {/* Dataset sizing */}
          <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-3.5 space-y-3">
            <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
              <Sliders size={14} className="text-cyan-400" />
              Dataset Dimensions
            </span>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <label>
                <span className="text-slate-400">Total Samples</span>
                <input
                  type="number"
                  min={300}
                  max={10000}
                  step={100}
                  value={samples}
                  onChange={(e) => setSamples(Number(e.target.value))}
                  className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-1.5 text-white outline-none focus:border-cyan-400"
                />
              </label>
              <label>
                <span className="text-slate-400">Features</span>
                <input
                  type="number"
                  min={4}
                  max={30}
                  value={features}
                  onChange={(e) => {
                    const f = Number(e.target.value);
                    setFeatures(f);
                    setTargetFeatures((prev) => prev.filter((i) => i < f));
                  }}
                  className="mt-1 w-full rounded border border-slate-700 bg-slate-900 p-1.5 text-white outline-none focus:border-cyan-400"
                />
              </label>
            </div>
          </div>

          {/* Covariate Shift */}
          <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-3.5 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300">Mean Shift (Δμ)</span>
              <span className="font-mono text-cyan-300 font-bold">{meanShift > 0 ? `+${meanShift}` : meanShift}</span>
            </div>
            <input
              type="range"
              min={-5.0}
              max={5.0}
              step={0.1}
              value={meanShift}
              onChange={(e) => setMeanShift(Number(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <div className="flex items-center justify-between text-[11px] text-slate-500">
              <span>-5.0 (Negative)</span>
              <span>0 (None)</span>
              <span>+5.0 (Positive)</span>
            </div>
          </div>

          {/* Variance Spread */}
          <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-3.5 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300">Variance Scaling</span>
              <span className="font-mono text-indigo-300 font-bold">{varianceScale}x</span>
            </div>
            <input
              type="range"
              min={0.2}
              max={4.0}
              step={0.1}
              value={varianceScale}
              onChange={(e) => setVarianceScale(Number(e.target.value))}
              className="w-full accent-indigo-400 cursor-pointer"
            />
            <div className="flex items-center justify-between text-[11px] text-slate-500">
              <span>0.2x (Compressed)</span>
              <span>1.0x (Normal)</span>
              <span>4.0x (Expanded)</span>
            </div>
          </div>

          {/* Gaussian Noise Injection */}
          <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-3.5 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300">Gaussian Noise (σ)</span>
              <span className="font-mono text-emerald-300 font-bold">{gaussianNoise}</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={2.5}
              step={0.05}
              value={gaussianNoise}
              onChange={(e) => setGaussianNoise(Number(e.target.value))}
              className="w-full accent-emerald-400 cursor-pointer"
            />
            <div className="flex items-center justify-between text-[11px] text-slate-500">
              <span>0 (Clean)</span>
              <span>1.0 (Moderate)</span>
              <span>2.5 (Severe)</span>
            </div>
          </div>
        </div>

        {/* Secondary Perturbations: Outliers & Label Flips */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {/* Outliers */}
          <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-3.5 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300">Outlier Contamination</span>
              <span className="font-mono text-amber-300 font-bold">{(outlierRate * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={0.25}
              step={0.01}
              value={outlierRate}
              onChange={(e) => setOutlierRate(Number(e.target.value))}
              className="w-full accent-amber-400 cursor-pointer"
            />
            <div className="flex items-center justify-between text-[11px] text-slate-500">
              <span>0%</span>
              <span>Scale: {outlierScale}x std</span>
              <span>25%</span>
            </div>
          </div>

          {/* Label Flips (Concept Drift) */}
          <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-3.5 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300">Label Inversion (Concept Shift)</span>
              <span className="font-mono text-rose-300 font-bold">{(labelFlipRate * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={0.40}
              step={0.02}
              value={labelFlipRate}
              onChange={(e) => setLabelFlipRate(Number(e.target.value))}
              className="w-full accent-rose-400 cursor-pointer"
            />
            <div className="flex items-center justify-between text-[11px] text-slate-500">
              <span>0% (Static)</span>
              <span>20% (Substantial)</span>
              <span>40% (Degraded)</span>
            </div>
          </div>

          {/* Targeted Features Selection */}
          <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-3.5 space-y-2">
            <span className="text-xs font-semibold text-slate-300 block">
              Perturbed Feature Targets:
            </span>
            <div className="flex flex-wrap gap-1.5 max-h-16 overflow-y-auto pt-0.5">
              {Array.from({ length: features }).map((_, idx) => {
                const isSelected = targetFeatures.includes(idx);
                return (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => toggleTargetFeature(idx)}
                    className={`rounded px-2 py-0.5 text-[11px] font-mono transition-colors cursor-pointer ${
                      isSelected
                        ? "bg-amber-500/25 text-amber-200 border border-amber-500/40 font-bold"
                        : "bg-slate-900 text-slate-500 border border-slate-800 hover:text-slate-300"
                    }`}
                  >
                    f_{idx}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="flex items-center justify-between pt-1">
          <button
            type="button"
            onClick={() => {
              setMeanShift(0.0);
              setVarianceScale(1.0);
              setGaussianNoise(0.0);
              setOutlierRate(0.0);
              setLabelFlipRate(0.0);
            }}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-400 hover:text-white transition-colors cursor-pointer"
          >
            <RotateCcw size={13} />
            Reset to Neutral
          </button>

          <button
            type="submit"
            disabled={loading}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-amber-500 to-orange-600 px-5 py-2.5 text-sm font-semibold text-slate-950 shadow-lg shadow-amber-500/20 hover:from-amber-400 hover:to-orange-500 disabled:opacity-50 transition-all cursor-pointer font-sans"
          >
            {loading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Play size={16} fill="currentColor" />
            )}
            {loading ? "Simulating Perturbations..." : "Inject Perturbations & Stress-Test"}
          </button>
        </div>
      </form>

      {/* Error Callout */}
      {error && (
        <div className="mt-5 flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">
          <AlertTriangle size={18} className="shrink-0 mt-0.5 text-rose-400" />
          <div>
            <p className="font-semibold text-rose-300">Simulation Error</p>
            <p className="text-xs text-rose-200/90 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Simulation Results Dashboard */}
      {report && (
        <div className="mt-8 space-y-6 border-t border-slate-800/80 pt-6">
          {/* Header Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-slate-900/60 p-4 border border-slate-800">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">
                  Drift Simulation Run
                </span>
                <span className="rounded-full bg-amber-500/15 border border-amber-500/30 px-2.5 py-0.5 text-[11px] font-semibold text-amber-300">
                  {report.drift_summary.drifted_features_count} of {report.drift_summary.total_features} Features Shifted
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Tested with {report.model_stress_test.model_type} across {report.dataset_metadata.test_size} evaluation samples
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={copyJson}
                className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs text-slate-300 hover:text-white transition-colors cursor-pointer"
              >
                {copied ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
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

          {/* Model Resilience & Stress Test KPI Strip */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {/* KPI 1: Resilience Rating */}
            <div className="rounded-xl border border-slate-800 bg-[#111a2e] p-4 shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Model Resilience Rating</span>
                <ShieldAlert size={16} className="text-amber-400" />
              </div>
              <div className="mt-2">
                <ResilienceBadge rating={report.model_stress_test.resilience_rating} />
              </div>
              <div className="mt-2 text-xs text-slate-400 pt-2 border-t border-slate-800">
                Stress Model: {report.model_stress_test.model_type}
              </div>
            </div>

            {/* KPI 2: Accuracy Degradation */}
            <div className="rounded-xl border border-slate-800 bg-[#111a2e] p-4 shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Accuracy Impact</span>
                <span className="font-mono text-xs text-slate-400">
                  {(report.model_stress_test.baseline.accuracy * 100).toFixed(1)}% → {(report.model_stress_test.perturbed.accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="mt-2 text-2xl font-bold font-mono text-white flex items-center gap-2">
                {(report.model_stress_test.perturbed.accuracy * 100).toFixed(1)}%
                <span className={`text-xs font-normal ${report.model_stress_test.accuracy_drop > 0.05 ? "text-rose-400" : "text-emerald-400"}`}>
                  (Δ -{(report.model_stress_test.accuracy_drop * 100).toFixed(1)}%)
                </span>
              </div>
              <div className="mt-2 text-xs text-slate-400 pt-2 border-t border-slate-800">
                Clean Baseline: {(report.model_stress_test.baseline.accuracy * 100).toFixed(1)}%
              </div>
            </div>

            {/* KPI 3: F1-Score Impact */}
            <div className="rounded-xl border border-slate-800 bg-[#111a2e] p-4 shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>F1-Score Impact</span>
                <span className="font-mono text-xs text-slate-400">
                  {(report.model_stress_test.baseline.f1_score * 100).toFixed(1)}% → {(report.model_stress_test.perturbed.f1_score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="mt-2 text-2xl font-bold font-mono text-white flex items-center gap-2">
                {(report.model_stress_test.perturbed.f1_score * 100).toFixed(1)}%
                <span className={`text-xs font-normal ${report.model_stress_test.f1_drop > 0.05 ? "text-rose-400" : "text-emerald-400"}`}>
                  (Δ -{(report.model_stress_test.f1_drop * 100).toFixed(1)}%)
                </span>
              </div>
              <div className="mt-2 text-xs text-slate-400 pt-2 border-t border-slate-800">
                Precision: {(report.model_stress_test.perturbed.precision * 100).toFixed(1)}% • Recall: {(report.model_stress_test.perturbed.recall * 100).toFixed(1)}%
              </div>
            </div>

            {/* KPI 4: Max PSI & Wasserstein */}
            <div className="rounded-xl border border-slate-800 bg-[#111a2e] p-4 shadow-sm">
              <div className="flex items-center justify-between text-slate-400 text-xs font-medium">
                <span>Maximum PSI Score</span>
                <BarChart2 size={16} className="text-cyan-400" />
              </div>
              <div className="mt-2 text-2xl font-bold font-mono text-cyan-300">
                {report.drift_summary.max_psi_score.toFixed(4)}
              </div>
              <div className="mt-2 flex items-center justify-between text-xs text-slate-400 pt-2 border-t border-slate-800">
                <span>Max Earth Mover: {report.drift_summary.max_wasserstein.toFixed(4)}</span>
                <span>Critical: {report.drift_summary.critical_psi_features_count}</span>
              </div>
            </div>
          </div>

          {/* Interactive Density Distribution Shift Curves (Recharts) */}
          {report.distribution_visuals.length > 0 && activeVisual && (
            <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h4 className="text-sm font-semibold text-white">
                    Probability Density Shift (Before vs After Perturbation)
                  </h4>
                  <p className="text-xs text-slate-400">
                    Comparing continuous distribution curves between clean baseline and perturbed validation sets
                  </p>
                </div>

                {/* Feature Selector Tabs */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-xs text-slate-500">Feature:</span>
                  {report.distribution_visuals.map((v) => (
                    <button
                      key={v.feature_index}
                      type="button"
                      onClick={() => setSelectedFeaturePlot(v.feature_index)}
                      className={`rounded px-2.5 py-1 text-xs font-mono transition-colors cursor-pointer ${
                        selectedFeaturePlot === v.feature_index
                          ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold"
                          : "bg-slate-900 text-slate-400 hover:text-white border border-slate-800"
                      }`}
                    >
                      {v.feature_name} (PSI: {v.psi_score.toFixed(2)})
                    </button>
                  ))}
                </div>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={activeVisual.density_bins}
                    margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="baselineDensity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.0} />
                      </linearGradient>
                      <linearGradient id="perturbedDensity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis
                      dataKey="bin_center"
                      stroke="#64748b"
                      fontSize={11}
                      tickLine={false}
                    />
                    <YAxis stroke="#64748b" fontSize={11} tickLine={false} />
                    <Tooltip
                      formatter={(val) => [Number(val).toFixed(4), ""]}
                      contentStyle={{
                        backgroundColor: "#0f172a",
                        border: "1px solid #334155",
                        borderRadius: "8px",
                        fontSize: "12px",
                      }}
                    />
                    <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: "12px" }} />
                    <Area
                      type="monotone"
                      dataKey="baseline_density"
                      name="Clean Baseline Density"
                      stroke="#38bdf8"
                      strokeWidth={2}
                      fillOpacity={1}
                      fill="url(#baselineDensity)"
                    />
                    <Area
                      type="monotone"
                      dataKey="perturbed_density"
                      name="Perturbed / Drifted Density"
                      stroke="#f43f5e"
                      strokeWidth={2}
                      fillOpacity={1}
                      fill="url(#perturbedDensity)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Per-Feature Drift Breakdown Table */}
          <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4">
            <h4 className="text-sm font-semibold text-white mb-1">
              Multi-Metric Drift Diagnostic Breakdown
            </h4>
            <p className="text-xs text-slate-400 mb-3">
              Evaluating KS tests (FDR-corrected p-values), PSI risk buckets, and Wasserstein distance per feature
            </p>

            <div className="overflow-x-auto rounded-lg border border-slate-800">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-900/90 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3">Feature</th>
                    <th className="py-2.5 px-3">Status</th>
                    <th className="py-2.5 px-3">PSI Risk</th>
                    <th className="py-2.5 px-3">KS Stat</th>
                    <th className="py-2.5 px-3">Adj. p-value</th>
                    <th className="py-2.5 px-3">Wasserstein</th>
                    <th className="py-2.5 px-3">Mean Shift</th>
                    <th className="py-2.5 px-3">Perturbed Target</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {report.feature_drifts.map((f) => (
                    <tr key={f.feature_name} className="hover:bg-slate-900/40 text-slate-300">
                      <td className="py-2 px-3 font-semibold text-white">
                        {f.feature_name}
                      </td>
                      <td className="py-2 px-3 font-sans">
                        {f.drift_detected ? (
                          <span className="inline-flex items-center gap-1 text-rose-400 font-semibold text-[11px]">
                            <AlertTriangle size={12} /> Drifted
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-emerald-400 text-[11px]">
                            <CheckCircle2 size={12} /> Stable
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-3">
                        <PsiRiskBadge risk={f.psi_risk} score={f.psi_score} />
                      </td>
                      <td className="py-2 px-3 text-cyan-300">{f.ks_statistic.toFixed(4)}</td>
                      <td className="py-2 px-3 text-slate-400">{f.adjusted_p_value.toExponential(2)}</td>
                      <td className="py-2 px-3 text-amber-300">{f.wasserstein_distance.toFixed(4)}</td>
                      <td className="py-2 px-3">
                        {f.baseline_mean.toFixed(2)} → {f.perturbed_mean.toFixed(2)}
                      </td>
                      <td className="py-2 px-3 font-sans text-[11px]">
                        {f.is_targeted_in_simulation ? (
                          <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-amber-300 font-semibold border border-amber-500/30">
                            Active Target
                          </span>
                        ) : (
                          <span className="text-slate-500">Unperturbed</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Remediation & Engineering Actions */}
          <div className="rounded-xl border border-slate-800 bg-[#0b1120] p-4 space-y-3">
            <div className="flex items-center gap-2">
              <Info size={16} className="text-amber-400" />
              <h4 className="text-sm font-semibold text-white">
                Prescriptive Remediation & Safety Recommendations
              </h4>
            </div>
            <div className="space-y-2">
              {report.recommendations.map((rec, idx) => (
                <div
                  key={idx}
                  className="rounded-lg bg-slate-900/80 p-3 border border-slate-800/80 text-xs text-slate-300 leading-relaxed flex items-start gap-2.5"
                >
                  <span className="text-amber-400 font-bold">•</span>
                  <span>{rec}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Raw JSON Accordion */}
          {showRawJson && (
            <div className="rounded-xl border border-slate-800 bg-slate-950 p-4">
              <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-300">Raw Simulation Report JSON</span>
                <span>{JSON.stringify(report).length} bytes</span>
              </div>
              <pre className="max-h-[350px] overflow-auto rounded-lg bg-slate-900/90 p-4 text-xs font-mono text-amber-200 leading-5">
                {JSON.stringify(report, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function ResilienceBadge({ rating }: { rating: string }) {
  switch (rating) {
    case "highly_resilient":
      return (
        <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-500/20 border border-emerald-500/30 px-2.5 py-1 text-xs font-bold text-emerald-300">
          <CheckCircle2 size={14} /> Highly Resilient
        </span>
      );
    case "minor_impact":
      return (
        <span className="inline-flex items-center gap-1.5 rounded-md bg-sky-500/20 border border-sky-500/30 px-2.5 py-1 text-xs font-bold text-sky-300">
          Minor Degradation
        </span>
      );
    case "moderate_degradation":
      return (
        <span className="inline-flex items-center gap-1.5 rounded-md bg-amber-500/20 border border-amber-500/30 px-2.5 py-1 text-xs font-bold text-amber-300">
          <AlertTriangle size={14} /> Moderate Vulnerability
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1.5 rounded-md bg-rose-500/20 border border-rose-500/30 px-2.5 py-1 text-xs font-bold text-rose-300">
          <ShieldAlert size={14} /> Critical Vulnerability
        </span>
      );
  }
}

function PsiRiskBadge({ risk, score }: { risk: string; score: number }) {
  if (risk === "critical" || score >= 0.25) {
    return (
      <span className="rounded bg-rose-500/20 px-2 py-0.5 text-[11px] font-bold text-rose-300 border border-rose-500/30">
        {score.toFixed(3)} (Critical)
      </span>
    );
  }
  if (risk === "moderate" || score >= 0.10) {
    return (
      <span className="rounded bg-amber-500/20 px-2 py-0.5 text-[11px] font-bold text-amber-300 border border-amber-500/30">
        {score.toFixed(3)} (Moderate)
      </span>
    );
  }
  return (
    <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[11px] font-semibold text-emerald-300 border border-emerald-500/30">
      {score.toFixed(3)} (Stable)
    </span>
  );
}
