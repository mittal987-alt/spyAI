
import { useState } from "react";
import axios from "axios";
import { GitCompare, Loader2 } from "lucide-react";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export default function DeepLearningComparison() {
  const [epochs, setEpochs] = useState(15);
  const [samples, setSamples] = useState(1500);
  const [features, setFeatures] = useState(10);
  const [randomState, setRandomState] = useState(42);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function compareModels(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setResult(null);
    setLoading(true);

    try {
      const response = await axios.post(
        `${API_BASE}/api/v1/deep-learning/compare`,
        {
          epochs,
          n_samples: samples,
          n_features: features,
          random_state: randomState,
        },
      );

      setResult(response.data);
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

  return (
    <section className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5 sm:p-6">
      <div className="mb-5 flex items-center gap-3">
        <div className="rounded-xl bg-cyan-400/10 p-3 text-cyan-300">
          <GitCompare size={22} />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-white">
            Model Comparison
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Compare supported model variants using the backend comparison service.
          </p>
        </div>
      </div>

      <form onSubmit={compareModels} className="space-y-5">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <NumberField
            label="Epochs"
            value={epochs}
            min={2}
            max={100}
            onChange={setEpochs}
          />
          <NumberField
            label="Samples"
            value={samples}
            min={100}
            max={20000}
            onChange={setSamples}
          />
          <NumberField
            label="Features"
            value={features}
            min={4}
            max={200}
            onChange={setFeatures}
          />
          <NumberField
            label="Random state"
            value={randomState}
            min={0}
            max={2147483647}
            onChange={setRandomState}
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-5 py-3 text-sm font-medium text-cyan-200 hover:bg-cyan-400/20 disabled:opacity-50"
        >
          {loading ? (
            <Loader2 size={17} className="animate-spin" />
          ) : (
            <GitCompare size={17} />
          )}
          {loading ? "Comparing models..." : "Compare models"}
        </button>
      </form>

      {error && (
        <p className="mt-5 rounded-lg bg-rose-500/10 p-3 text-sm text-rose-200">
          {error}
        </p>
      )}

      {result && (
        <div className="mt-6 border-t border-slate-800 pt-5">
          <h3 className="mb-3 font-semibold text-white">
            Comparison response
          </h3>
          <pre className="max-h-[500px] overflow-auto rounded-xl bg-slate-950 p-4 text-xs leading-6 text-cyan-100">
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
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block text-sm text-slate-300">
      {label}
      <input
        type="number"
        min={min}
        max={max}
        required
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 outline-none focus:border-cyan-400"
      />
    </label>
  );
}
