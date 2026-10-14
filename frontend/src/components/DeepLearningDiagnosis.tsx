
import { useState } from "react";
import axios from "axios";
import { BrainCircuit, Loader2, Play, AlertTriangle } from "lucide-react";

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

const initialConfig: DiagnosisConfig = {
  epochs: 20,
  batch_size: 64,
  learning_rate: 0.001,
  n_samples: 3000,
  n_features: 20,
  hidden_units: 64,
  dropout: 0.2,
  random_state: 42,
  explain: true,
};

export default function DeepLearningDiagnosis() {
  const [config, setConfig] = useState(initialConfig);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function updateConfig<K extends keyof DiagnosisConfig>(
    key: K,
    value: DiagnosisConfig[K],
  ) {
    setConfig((current) => ({ ...current, [key]: value }));
  }

  async function runDiagnosis(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const response = await axios.post(
        `${API_BASE}/api/v1/deep-learning/diagnose`,
        config,
      );

      setResult(response.data);
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

  return (
    <section className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5 sm:p-6">
      <div className="mb-6 flex items-center gap-3">
        <div className="rounded-xl bg-violet-400/10 p-3 text-violet-300">
          <BrainCircuit size={23} />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-white">
            Deep Learning Diagnosis
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Run a neural-network experiment using the existing backend.
          </p>
        </div>
      </div>

      <form onSubmit={runDiagnosis} className="space-y-5">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <NumberField
            label="Epochs"
            value={config.epochs}
            min={2}
            max={100}
            onChange={(value) => updateConfig("epochs", value)}
          />
          <NumberField
            label="Batch size"
            value={config.batch_size}
            min={1}
            max={512}
            onChange={(value) => updateConfig("batch_size", value)}
          />
          <NumberField
            label="Learning rate"
            value={config.learning_rate}
            min={0.00001}
            max={1}
            step={0.0001}
            onChange={(value) => updateConfig("learning_rate", value)}
          />
          <NumberField
            label="Samples"
            value={config.n_samples}
            min={100}
            max={20000}
            onChange={(value) => updateConfig("n_samples", value)}
          />
          <NumberField
            label="Features"
            value={config.n_features}
            min={4}
            max={200}
            onChange={(value) => updateConfig("n_features", value)}
          />
          <NumberField
            label="Hidden units"
            value={config.hidden_units}
            min={2}
            max={512}
            onChange={(value) => updateConfig("hidden_units", value)}
          />
          <NumberField
            label="Dropout"
            value={config.dropout}
            min={0}
            max={0.99}
            step={0.05}
            onChange={(value) => updateConfig("dropout", value)}
          />
        </div>

        <label className="flex items-center gap-3 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={config.explain}
            onChange={(event) =>
              updateConfig("explain", event.target.checked)
            }
            className="h-4 w-4 accent-violet-400"
          />
          Generate feature explanations
        </label>

        <button
          type="submit"
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-violet-400 px-5 py-3 text-sm font-semibold text-slate-950 hover:bg-violet-300 disabled:opacity-50"
        >
          {loading ? (
            <Loader2 size={17} className="animate-spin" />
          ) : (
            <Play size={17} />
          )}
          {loading ? "Running diagnosis..." : "Run diagnosis"}
        </button>
      </form>

      {error && (
        <div className="mt-5 flex gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">
          <AlertTriangle size={18} className="shrink-0" />
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 border-t border-slate-800 pt-5">
          <h3 className="mb-3 font-semibold text-white">
            Diagnosis response
          </h3>
          <pre className="max-h-[500px] overflow-auto rounded-xl bg-slate-950 p-4 text-xs leading-6 text-emerald-200">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </section>
  );
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
    <label className="block text-sm text-slate-300">
      {label}
      <input
        type="number"
        required
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 outline-none focus:border-violet-400"
      />
    </label>
  );
}
