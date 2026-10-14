
import { useState } from "react";
import axios from "axios";
import {
  Activity,
  AlertTriangle,
  Loader2,
  Play,
} from "lucide-react";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

interface SemanticRequest {
  reference_texts: string[];
  current_texts: string[];
  n_clusters: number;
  n_permutations: number;
  alpha: number;
  random_state: number;
}

export default function SemanticMonitoring() {
  const [referenceText, setReferenceText] = useState("");
  const [currentText, setCurrentText] = useState("");
  const [nClusters, setNClusters] = useState(5);
  const [nPermutations, setNPermutations] = useState(199);
  const [alpha, setAlpha] = useState(0.05);
  const [randomState, setRandomState] = useState(42);

  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function runMonitoring(
    event: React.FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();
    setError("");
    setResult(null);

    const reference_texts = referenceText
      .split("\n")
      .map((text) => text.trim())
      .filter(Boolean);

    const current_texts = currentText
      .split("\n")
      .map((text) => text.trim())
      .filter(Boolean);

    if (reference_texts.length < 5 || current_texts.length < 5) {
      setError(
        "Provide at least five non-empty reference texts and five current texts.",
      );
      return;
    }

    const payload: SemanticRequest = {
      reference_texts,
      current_texts,
      n_clusters: nClusters,
      n_permutations: nPermutations,
      alpha,
      random_state: randomState,
    };

    setLoading(true);

    try {
      const response = await axios.post(
        `${API_BASE}/api/v1/semantic/monitor`,
        payload,
      );

      setResult(response.data);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;

        setError(
          typeof detail === "string"
            ? detail
            : `Semantic monitoring failed (${err.response?.status ?? "network error"}).`,
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
        <div className="rounded-xl bg-emerald-400/10 p-3 text-emerald-300">
          <Activity size={23} />
        </div>

        <div>
          <h2 className="text-xl font-semibold text-white">
            Semantic Drift Monitoring
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Compare text distributions to investigate semantic changes.
          </p>
        </div>
      </div>

      <form onSubmit={runMonitoring} className="space-y-5">
        <div>
          <label className="mb-2 block text-sm text-slate-300">
            Reference texts
          </label>

          <textarea
            required
            value={referenceText}
            onChange={(event) => setReferenceText(event.target.value)}
            placeholder={
              "Enter one reference text per line...\nExample: Customers usually ask about account balances."
            }
            className="min-h-40 w-full rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm leading-6 outline-none focus:border-emerald-400"
          />

          <p className="mt-1 text-xs text-slate-500">
            At least five non-empty lines are required.
          </p>
        </div>

        <div>
          <label className="mb-2 block text-sm text-slate-300">
            Current texts
          </label>

          <textarea
            required
            value={currentText}
            onChange={(event) => setCurrentText(event.target.value)}
            placeholder={
              "Enter one current text per line...\nExample: Customers now ask about international transfers."
            }
            className="min-h-40 w-full rounded-xl border border-slate-700 bg-slate-950 p-3 text-sm leading-6 outline-none focus:border-emerald-400"
          />

          <p className="mt-1 text-xs text-slate-500">
            Use samples from the period or system you want to investigate.
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <NumberField
            label="Clusters"
            value={nClusters}
            min={2}
            max={20}
            onChange={setNClusters}
          />

          <NumberField
            label="Permutations"
            value={nPermutations}
            min={19}
            max={999}
            onChange={setNPermutations}
          />

          <NumberField
            label="Significance alpha"
            value={alpha}
            min={0.001}
            max={0.999}
            step={0.01}
            onChange={setAlpha}
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
          className="flex items-center gap-2 rounded-lg bg-emerald-400 px-5 py-3 text-sm font-semibold text-slate-950 hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? (
            <Loader2 size={17} className="animate-spin" />
          ) : (
            <Play size={17} />
          )}

          {loading ? "Analyzing semantic drift..." : "Run semantic analysis"}
        </button>
      </form>

      {error && (
        <div className="mt-5 flex items-start gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">
          <AlertTriangle size={18} className="mt-0.5 shrink-0" />
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 border-t border-slate-800 pt-5">
          <h3 className="mb-3 font-semibold text-white">
            Semantic monitoring results
          </h3>

          <pre className="max-h-[600px] overflow-auto rounded-xl bg-slate-950 p-4 text-xs leading-6 text-emerald-200">
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
        className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 outline-none focus:border-emerald-400"
      />
    </label>
  );
}
