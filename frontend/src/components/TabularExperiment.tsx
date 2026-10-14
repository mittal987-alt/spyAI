
import { useState } from "react";
import axios from "axios";
import { Activity, AlertCircle, LoaderCircle, Play } from "lucide-react";
import DriftVisualization from "./DriftVisualization";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

type ExperimentResult = Record<string, unknown>;

export default function TabularExperiment() {
  const [result, setResult] = useState<ExperimentResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runExperiment() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post<ExperimentResult>(
        `${API_BASE_URL}/api/v1/experiments/drift`,
        undefined,
        { timeout: 120000 },
      );

      setResult(response.data);
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;

        setError(
          typeof detail === "string"
            ? detail
            : err.response
              ? `Request failed with status ${err.response.status}. Check the backend response.`
              : "Could not connect to the backend. Make sure the FastAPI server is running.",
        );
      } else {
        setError("An unexpected error occurred while running the experiment.");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <section id="tabular-experiments" className="space-y-6">
      <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-6">
        <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-4">
            <div className="rounded-xl bg-cyan-400/10 p-3 text-cyan-300">
              <Activity size={25} />
            </div>

            <div>
              <h2 className="text-xl font-semibold text-white">
                Tabular Drift Experiment
              </h2>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
                Run the backend's tabular drift experiment and inspect its
                returned metrics, feature-level results, and raw JSON output.
              </p>

              <p className="mt-2 text-xs text-slate-500">
                Endpoint: POST /api/v1/experiments/drift
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={runExperiment}
            disabled={loading}
            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl bg-cyan-400 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? (
              <LoaderCircle size={18} className="animate-spin" />
            ) : (
              <Play size={18} />
            )}

            {loading ? "Running experiment..." : "Run experiment"}
          </button>
        </div>

        {loading && (
          <div
            role="status"
            className="mt-5 rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-4 text-sm text-cyan-200"
          >
            The experiment is running. Please wait for the backend response.
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="mt-5 flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4"
          >
            <AlertCircle
              size={20}
              className="mt-0.5 shrink-0 text-rose-300"
            />

            <div>
              <p className="font-medium text-rose-200">
                Experiment failed
              </p>
              <p className="mt-1 break-words text-sm text-rose-300/90">
                {error}
              </p>
            </div>
          </div>
        )}
      </div>

      <DriftVisualization result={result} />
    </section>
  );
}
