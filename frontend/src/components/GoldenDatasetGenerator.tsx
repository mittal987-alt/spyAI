import { useEffect, useState } from "react";
import axios from "axios";
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Download,
  Layers,
  Loader2,
  Play,
  RotateCcw,
  Sparkles,
} from "lucide-react";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

interface RetrievedDoc {
  id: string;
  text: string;
}

interface GoldenCase {
  test_id: string;
  test_type: "factual_benchmark" | "adversarial_hallucination";
  query: string;
  answer: string;
  ground_truth_context: string;
  relevant_document_ids: string[];
  retrieved_documents: RetrievedDoc[];
  k: number;
  relevance_threshold: number;
  support_threshold: number;
  notes: string;
}

interface PresetItem {
  title: string;
  text: string;
}

interface GoldenDatasetGeneratorProps {
  onLoadCaseIntoEvaluator?: (testCase: GoldenCase) => void;
}

export default function GoldenDatasetGenerator({
  onLoadCaseIntoEvaluator,
}: GoldenDatasetGeneratorProps) {
  const [presets, setPresets] = useState<Record<string, PresetItem>>({});
  const [selectedPresetKey, setSelectedPresetKey] = useState<string>("cloud_infrastructure");
  const [docText, setDocText] = useState("");
  const [numPairs, setNumPairs] = useState(4);
  const [includeHallucinations, setIncludeHallucinations] = useState(true);

  const [generating, setGenerating] = useState(false);
  const [cases, setCases] = useState<GoldenCase[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Fetch presets on mount
  useEffect(() => {
    async function loadPresets() {
      try {
        const res = await axios.get<{ presets: Record<string, PresetItem> }>(
          `${API_BASE}/api/v1/rag/dataset-presets`,
        );
        setPresets(res.data.presets);
        if (res.data.presets["cloud_infrastructure"]) {
          setDocText(res.data.presets["cloud_infrastructure"].text);
        }
      } catch (err: unknown) {
        console.warn("Could not load dataset presets:", err);
      }
    }
    void loadPresets();
  }, []);

  const handleSelectPreset = (key: string) => {
    setSelectedPresetKey(key);
    if (presets[key]) {
      setDocText(presets[key].text);
    }
  };

  const handleGenerate = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!docText.trim()) return;

    setGenerating(true);
    setError(null);

    try {
      const res = await axios.post<{ cases: GoldenCase[] }>(
        `${API_BASE}/api/v1/rag/generate-dataset`,
        {
          document_text: docText,
          num_pairs: numPairs,
          include_hallucination_cases: includeHallucinations,
          k_retrieved: 4,
        },
      );
      setCases(res.data.cases);
    } catch (err: unknown) {
      const msg =
        axios.isAxiosError(err) && err.response?.data?.detail
          ? err.response.data.detail
          : "Dataset generation failed. Check document length and connectivity.";
      setError(String(msg));
    } finally {
      setGenerating(false);
    }
  };

  const handleDownloadJSON = () => {
    if (cases.length === 0) return;
    const blob = new Blob([JSON.stringify(cases, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `rag_golden_benchmark_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyPrompt = (c: GoldenCase) => {
    navigator.clipboard.writeText(
      JSON.stringify(
        {
          query: c.query,
          answer: c.answer,
          retrieved_documents: c.retrieved_documents,
          relevant_document_ids: c.relevant_document_ids,
        },
        null,
        2,
      ),
    );
    setCopiedId(c.test_id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleSendToEvaluator = (c: GoldenCase) => {
    if (onLoadCaseIntoEvaluator) {
      onLoadCaseIntoEvaluator(c);
    }
    // Scroll smoothly to evaluate section
    const elem = document.getElementById("evaluate");
    if (elem) {
      elem.scrollIntoView({ behavior: "smooth" });
    }
  };

  const factualCount = cases.filter((c) => c.test_type === "factual_benchmark").length;
  const hallucinatedCount = cases.filter(
    (c) => c.test_type === "adversarial_hallucination",
  ).length;

  return (
    <div className="space-y-6">
      {/* Console Input Card */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-6 shadow-xl">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="flex items-center gap-2 text-base font-semibold text-white">
              <Sparkles className="text-cyan-400" size={18} />
              Synthetic Golden Dataset & Benchmark Generator
            </h3>
            <p className="mt-1 text-xs text-slate-400">
              Transform raw domain documentation into ground-truth RAG evaluation test suites with query-passage attribution and adversarial hallucination test cases.
            </p>
          </div>

          {/* Presets */}
          <div className="flex flex-wrap gap-2">
            {Object.entries(presets).map(([k, p]) => (
              <button
                key={k}
                type="button"
                onClick={() => handleSelectPreset(k)}
                className={`rounded-lg border px-2.5 py-1.5 text-xs transition ${
                  selectedPresetKey === k
                    ? "border-cyan-400/60 bg-cyan-500/10 text-cyan-300 font-medium"
                    : "border-slate-700 bg-slate-900/60 text-slate-300 hover:border-cyan-400/40 hover:bg-slate-800 hover:text-white"
                }`}
              >
                {p.title}
              </button>
            ))}
          </div>
        </div>

        {/* Text Input */}
        <div className="mt-4">
          <label className="block text-xs font-medium text-slate-400 mb-1">
            Raw Knowledge Base Text / Documentation
          </label>
          <textarea
            value={docText}
            onChange={(e) => setDocText(e.target.value)}
            rows={5}
            placeholder="Paste raw documentation, architectural specifications, or medical guidelines..."
            className="w-full rounded-xl border border-slate-700 bg-slate-950/70 p-3.5 text-xs font-mono text-slate-200 placeholder-slate-500 focus:border-cyan-400 focus:outline-none"
          />
        </div>

        {/* Controls */}
        <div className="mt-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap items-center gap-5 text-xs text-slate-300">
            <div className="flex items-center gap-2">
              <label htmlFor="num-pairs" className="text-slate-400">
                Number of Test Cases:
              </label>
              <select
                id="num-pairs"
                value={numPairs}
                onChange={(e) => setNumPairs(Number(e.target.value))}
                className="rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1 text-xs text-slate-200 focus:border-cyan-400 focus:outline-none"
              >
                <option value={2}>2 Cases</option>
                <option value={3}>3 Cases</option>
                <option value={4}>4 Cases</option>
                <option value={6}>6 Cases</option>
                <option value={8}>8 Cases</option>
              </select>
            </div>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={includeHallucinations}
                onChange={(e) => setIncludeHallucinations(e.target.checked)}
                className="rounded border-slate-700 bg-slate-900 text-cyan-500 focus:ring-0"
              />
              <span>Include Adversarial Hallucination Tests</span>
            </label>
          </div>

          <button
            type="button"
            onClick={() => void handleGenerate()}
            disabled={generating || !docText.trim()}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-5 py-2.5 text-xs font-semibold text-slate-950 shadow-lg shadow-cyan-500/20 transition hover:from-cyan-400 hover:to-blue-500 disabled:opacity-50"
          >
            {generating ? (
              <>
                <Loader2 size={14} className="animate-spin text-slate-950" />
                Synthesizing Dataset...
              </>
            ) : (
              <>
                <Sparkles size={14} />
                Generate Golden Benchmark
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 flex items-center gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
            <AlertTriangle size={15} />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Results View */}
      {cases.length > 0 && (
        <div className="space-y-4">
          {/* Summary KPI header */}
          <div className="flex flex-col gap-3 rounded-2xl border border-slate-800 bg-[#0d1424] p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1.5 text-xs font-semibold text-white">
                <Layers size={15} className="text-cyan-400" />
                Generated {cases.length} Benchmark Test Pairs
              </span>

              <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 text-[10px] font-semibold text-emerald-400">
                {factualCount} Factual
              </span>

              {hallucinatedCount > 0 && (
                <span className="rounded-full bg-rose-500/10 border border-rose-500/30 px-2 py-0.5 text-[10px] font-semibold text-rose-400">
                  {hallucinatedCount} Adversarial
                </span>
              )}
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleDownloadJSON}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-white"
              >
                <Download size={13} />
                Export JSON
              </button>

              <button
                type="button"
                onClick={() => void handleGenerate()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs text-slate-300 transition hover:bg-slate-700 hover:text-white"
              >
                <RotateCcw size={13} /> Re-synthesize
              </button>
            </div>
          </div>

          {/* Test cases list */}
          <div className="space-y-4">
            {cases.map((c, idx) => (
              <div
                key={c.test_id}
                className="overflow-hidden rounded-2xl border border-slate-800 bg-[#0d1424] p-5 shadow-lg transition hover:border-cyan-500/30"
              >
                {/* Header tag */}
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-2">
                    <span className="rounded bg-slate-800 px-2 py-0.5 font-mono text-[11px] text-cyan-300">
                      Case #{idx + 1}
                    </span>

                    {c.test_type === "factual_benchmark" ? (
                      <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-[10px] font-semibold text-emerald-300">
                        <CheckCircle2 size={11} /> FACTUAL BENCHMARK
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 rounded-full border border-rose-500/30 bg-rose-500/10 px-2.5 py-0.5 text-[10px] font-semibold text-rose-300">
                        <AlertTriangle size={11} /> ADVERSARIAL HALLUCINATION TEST
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => handleCopyPrompt(c)}
                      className="inline-flex items-center gap-1 rounded border border-slate-700 bg-slate-900 px-2.5 py-1 text-[11px] text-slate-400 hover:text-white"
                      title="Copy JSON representation"
                    >
                      <Copy size={12} />
                      {copiedId === c.test_id ? "Copied" : "JSON"}
                    </button>

                    <button
                      type="button"
                      onClick={() => handleSendToEvaluator(c)}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-500/10 border border-cyan-400/40 px-3 py-1 text-[11px] font-medium text-cyan-300 transition hover:bg-cyan-500 hover:text-slate-950"
                      title="Load into RAG evaluation form"
                    >
                      <Play size={11} /> Load into RAG Evaluator
                    </button>
                  </div>
                </div>

                {/* Query & Answer */}
                <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-3.5">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                      User Query
                    </span>
                    <p className="mt-1 text-xs font-medium text-white">{c.query}</p>
                  </div>

                  <div className="rounded-xl border border-slate-800/80 bg-slate-950/60 p-3.5">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                      Model Answer (Synthesized)
                    </span>
                    <p className="mt-1 text-xs text-slate-300 leading-relaxed">
                      {c.answer}
                    </p>
                  </div>
                </div>

                {/* Context & Retrieval sets */}
                <div className="mt-3 rounded-xl border border-slate-800/50 bg-slate-950/40 p-3">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                    Retrieved Passages ({c.retrieved_documents.length})
                  </span>

                  <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {c.retrieved_documents.map((doc, dIdx) => {
                      const isTarget = c.relevant_document_ids.includes(doc.id);
                      return (
                        <div
                          key={doc.id}
                          className={`rounded-lg border p-2.5 text-xs ${
                            isTarget
                              ? "border-emerald-500/30 bg-emerald-950/20 text-emerald-200"
                              : "border-slate-800 bg-slate-900/40 text-slate-400"
                          }`}
                        >
                          <div className="flex items-center justify-between text-[10px] mb-1">
                            <span className="font-mono font-bold">{doc.id}</span>
                            <span
                              className={`rounded px-1.5 py-0.2 uppercase font-semibold text-[9px] ${
                                isTarget
                                  ? "bg-emerald-500/20 text-emerald-300"
                                  : "bg-slate-800 text-slate-500"
                              }`}
                            >
                              {isTarget ? "Relevant Ground Truth" : `Distractor #${dIdx}`}
                            </span>
                          </div>
                          <p className="line-clamp-2 text-[11px] font-mono leading-relaxed">
                            {doc.text}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
