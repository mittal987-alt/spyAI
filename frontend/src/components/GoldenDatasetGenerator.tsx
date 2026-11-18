import { useEffect, useState, useCallback } from "react";
import axios from "axios";
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Download,
  FolderOpen,
  Layers,
  Loader2,
  Play,
  RotateCcw,
  Save,
  Sparkles,
  Trash2,
} from "lucide-react";
import {
  saveGoldenDataset,
  listGoldenDatasets,
  getGoldenDataset,
  deleteGoldenDataset,
} from "../api";
import type { GoldenDatasetRecord } from "../types";

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

  // Saved library state
  const [showSavedDrawer, setShowSavedDrawer] = useState(false);
  const [savedDatasets, setSavedDatasets] = useState<GoldenDatasetRecord[]>([]);
  const [loadingSaved, setLoadingSaved] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [savingDataset, setSavingDataset] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

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

  const loadSavedLibrary = useCallback(async () => {
    setLoadingSaved(true);
    try {
      const data = await listGoldenDatasets();
      setSavedDatasets(data);
    } catch (err: unknown) {
      console.error("Failed to load saved datasets", err);
    } finally {
      setLoadingSaved(false);
    }
  }, []);

  useEffect(() => {
    if (showSavedDrawer) {
      void loadSavedLibrary();
    }
  }, [showSavedDrawer, loadSavedLibrary]);

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

  const handleSaveToRegistry = async () => {
    if (cases.length === 0) return;
    const name = saveName.trim() || `benchmark-${selectedPresetKey}-${new Date().toISOString().slice(0, 10)}`;
    setSavingDataset(true);
    try {
      await saveGoldenDataset({
        name,
        description: `Synthetic benchmark with ${cases.length} ground-truth pairs`,
        cases: cases as any[],
      });
      setStatusMessage(`Saved "${name}" to Persistent Benchmark Library.`);
      setSaveName("");
      void loadSavedLibrary();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to save dataset");
    } finally {
      setSavingDataset(false);
      setTimeout(() => setStatusMessage(null), 5000);
    }
  };

  const handleLoadSavedDataset = async (datasetId: string) => {
    try {
      const full = await getGoldenDataset(datasetId);
      if (full.test_cases && full.test_cases.length > 0) {
        // Map to GoldenCase structure
        const mappedCases: GoldenCase[] = full.test_cases.map((tc, idx) => ({
          test_id: `saved-${idx + 1}`,
          test_type: tc.is_adversarial ? "adversarial_hallucination" : "factual_benchmark",
          query: tc.query,
          answer: tc.ground_truth_answer,
          ground_truth_context: tc.expected_context,
          relevant_document_ids: ["doc-1"],
          retrieved_documents: [
            { id: "doc-1", text: tc.expected_context },
          ],
          k: 4,
          relevance_threshold: 0.7,
          support_threshold: 0.6,
          notes: tc.topic || "Loaded from persistent benchmark registry",
        }));
        setCases(mappedCases);
        setShowSavedDrawer(false);
        setStatusMessage(`Loaded ${mappedCases.length} test cases from "${full.name}".`);
        setTimeout(() => setStatusMessage(null), 4000);
      }
    } catch (err: unknown) {
      console.error(err);
    }
  };

  const handleDeleteSaved = async (datasetId: string, name: string) => {
    if (!window.confirm(`Delete saved benchmark "${name}"?`)) return;
    try {
      await deleteGoldenDataset(datasetId);
      void loadSavedLibrary();
    } catch (err: unknown) {
      console.error(err);
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
    void navigator.clipboard.writeText(
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
    const elem = document.getElementById("evaluate");
    elem?.scrollIntoView({ behavior: "smooth" });
  };

  const factualCount = cases.filter((c) => c.test_type === "factual_benchmark").length;
  const hallucinatedCount = cases.filter(
    (c) => c.test_type === "adversarial_hallucination",
  ).length;

  return (
    <div className="space-y-6">
      {/* Console Input Card */}
      <div className="rounded-xl border border-white/[0.08] bg-[#0c0e15] p-5 shadow-sm">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
              <Sparkles className="text-indigo-400" size={16} />
              Synthetic Golden Dataset & Benchmark Generator
            </h3>
            <p className="mt-1 text-xs text-zinc-400">
              Transform raw domain documentation into ground-truth RAG evaluation test suites with query-passage attribution and adversarial hallucination test cases.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setShowSavedDrawer(!showSavedDrawer)}
              className="flex items-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.03] px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-white/20 hover:bg-white/[0.06] hover:text-white"
            >
              <FolderOpen size={13} className="text-indigo-400" />
              <span>Saved Benchmarks</span>
            </button>
          </div>
        </div>

        {/* Presets */}
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] text-zinc-500 mr-1">Presets:</span>
          {Object.entries(presets).map(([k, p]) => (
            <button
              key={k}
              type="button"
              onClick={() => handleSelectPreset(k)}
              className={`rounded-md border px-2.5 py-1 text-[11px] transition ${
                selectedPresetKey === k
                  ? "border-indigo-500/40 bg-indigo-500/10 text-indigo-300 font-medium"
                  : "border-white/[0.08] bg-white/[0.03] text-zinc-400 hover:text-white"
              }`}
            >
              {p.title}
            </button>
          ))}
        </div>

        {/* Text Input */}
        <div className="mt-4">
          <label className="block text-xs font-medium text-zinc-400 mb-1">
            Raw Knowledge Base Text / Documentation
          </label>
          <textarea
            value={docText}
            onChange={(e) => setDocText(e.target.value)}
            rows={5}
            placeholder="Paste raw documentation, architectural specifications, or medical guidelines..."
            className="linear-input w-full p-3 text-xs font-mono"
          />
        </div>

        {/* Controls */}
        <div className="mt-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex flex-wrap items-center gap-4 text-xs text-zinc-400">
            <div className="flex items-center gap-2">
              <span>Test Cases:</span>
              <select
                value={numPairs}
                onChange={(e) => setNumPairs(Number(e.target.value))}
                className="linear-input px-2 py-1 text-xs"
              >
                <option value={2}>2 Cases</option>
                <option value={4}>4 Cases</option>
                <option value={6}>6 Cases</option>
                <option value={8}>8 Cases</option>
              </select>
            </div>

            <label className="flex items-center gap-2 cursor-pointer text-zinc-300">
              <input
                type="checkbox"
                checked={includeHallucinations}
                onChange={(e) => setIncludeHallucinations(e.target.checked)}
                className="rounded border-zinc-700 bg-zinc-900 text-indigo-500"
              />
              <span>Include Adversarial Hallucination Tests</span>
            </label>
          </div>

          <button
            type="button"
            onClick={() => void handleGenerate()}
            disabled={generating || !docText.trim()}
            className="linear-button-primary inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold disabled:opacity-50"
          >
            {generating ? (
              <>
                <Loader2 size={13} className="animate-spin text-white" />
                Synthesizing Dataset...
              </>
            ) : (
              <>
                <Sparkles size={13} />
                Generate Golden Benchmark
              </>
            )}
          </button>
        </div>

        {statusMessage && (
          <div className="mt-3 rounded-lg border border-indigo-500/30 bg-indigo-500/10 p-2.5 text-xs text-indigo-300">
            {statusMessage}
          </div>
        )}

        {error && (
          <div className="mt-3 flex items-center gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-2.5 text-xs text-rose-300">
            <AlertTriangle size={14} />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Saved Benchmarks Drawer */}
      {showSavedDrawer && (
        <div className="rounded-xl border border-indigo-500/20 bg-[#0e1017] p-5 shadow-lg">
          <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
            <div className="flex items-center gap-2">
              <FolderOpen size={16} className="text-indigo-400" />
              <h4 className="text-xs font-semibold text-white">Persistent Golden Benchmark Library</h4>
            </div>
            <button
              type="button"
              onClick={() => setShowSavedDrawer(false)}
              className="text-zinc-500 hover:text-white text-xs"
            >
              ✕ Close
            </button>
          </div>

          {loadingSaved ? (
            <p className="py-6 text-center text-xs text-zinc-500">Loading saved benchmarks...</p>
          ) : savedDatasets.length === 0 ? (
            <p className="py-6 text-center text-xs text-zinc-500">
              No benchmark suites saved yet. Generate a test set and click &ldquo;Save to Library&rdquo;.
            </p>
          ) : (
            <div className="mt-3 divide-y divide-white/[0.04]">
              {savedDatasets.map((ds) => (
                <div
                  key={ds.dataset_id}
                  className="flex items-center justify-between py-2.5 text-xs"
                >
                  <div>
                    <span className="font-semibold text-zinc-200">{ds.name}</span>
                    <span className="ml-2 font-mono text-[10px] text-zinc-500">
                      ({ds.test_case_count} cases) · {new Date(ds.created_at).toLocaleDateString()}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => void handleLoadSavedDataset(ds.dataset_id)}
                      className="rounded border border-indigo-500/30 bg-indigo-500/10 px-2.5 py-1 text-[11px] font-medium text-indigo-300 hover:bg-indigo-500/20"
                    >
                      Load Cases
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleDeleteSaved(ds.dataset_id, ds.name)}
                      className="rounded p-1 text-zinc-500 hover:text-rose-400"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Results View */}
      {cases.length > 0 && (
        <div className="space-y-4">
          {/* Summary KPI header */}
          <div className="flex flex-col gap-3 rounded-xl border border-white/[0.08] bg-[#0c0e15] p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1.5 text-xs font-semibold text-white">
                <Layers size={14} className="text-indigo-400" />
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
              <input
                type="text"
                placeholder="Dataset name..."
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
                className="linear-input w-36 px-2.5 py-1 text-[11px]"
              />

              <button
                type="button"
                onClick={() => void handleSaveToRegistry()}
                disabled={savingDataset}
                className="flex items-center gap-1 rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-2.5 py-1 text-xs font-medium text-indigo-300 hover:bg-indigo-500/20 disabled:opacity-50"
              >
                <Save size={12} />
                <span>Save</span>
              </button>

              <button
                type="button"
                onClick={handleDownloadJSON}
                className="flex items-center gap-1 rounded-lg border border-white/[0.08] bg-white/[0.03] px-2.5 py-1 text-xs text-zinc-300 hover:text-white"
              >
                <Download size={12} />
                <span>Export JSON</span>
              </button>

              <button
                type="button"
                onClick={() => void handleGenerate()}
                className="rounded-lg border border-white/[0.08] bg-white/[0.03] p-1.5 text-zinc-400 hover:text-white"
                title="Re-synthesize"
              >
                <RotateCcw size={12} />
              </button>
            </div>
          </div>

          {/* Test cases list */}
          <div className="space-y-3">
            {cases.map((c, idx) => (
              <div
                key={c.test_id}
                className="rounded-xl border border-white/[0.07] bg-[#0c0e15] p-4 transition hover:border-white/20"
              >
                {/* Header tag */}
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-2">
                    <span className="rounded bg-zinc-800 px-2 py-0.5 font-mono text-[10px] text-zinc-300">
                      Case #{idx + 1}
                    </span>

                    {c.test_type === "factual_benchmark" ? (
                      <span className="inline-flex items-center gap-1 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-semibold text-emerald-300">
                        <CheckCircle2 size={10} /> FACTUAL BENCHMARK
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 rounded-full border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 text-[10px] font-semibold text-rose-300">
                        <AlertTriangle size={10} /> ADVERSARIAL HALLUCINATION
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => handleCopyPrompt(c)}
                      className="rounded border border-white/[0.08] bg-white/[0.03] px-2 py-0.5 text-[10px] text-zinc-400 hover:text-white"
                    >
                      <Copy size={11} className="inline mr-1" />
                      {copiedId === c.test_id ? "Copied" : "JSON"}
                    </button>

                    <button
                      type="button"
                      onClick={() => handleSendToEvaluator(c)}
                      className="flex items-center gap-1 rounded border border-indigo-500/30 bg-indigo-500/10 px-2.5 py-0.5 text-[10px] font-medium text-indigo-300 hover:bg-indigo-500/20"
                    >
                      <Play size={10} /> Load in Evaluator
                    </button>
                  </div>
                </div>

                {/* Query & Answer */}
                <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2 text-xs">
                  <div className="rounded-lg border border-white/[0.04] bg-[#0e1017] p-3">
                    <span className="font-mono text-[9px] uppercase tracking-wider text-zinc-500">
                      User Query
                    </span>
                    <p className="mt-1 font-medium text-white">{c.query}</p>
                  </div>

                  <div className="rounded-lg border border-white/[0.04] bg-[#0e1017] p-3">
                    <span className="font-mono text-[9px] uppercase tracking-wider text-zinc-500">
                      Model Answer (Ground Truth)
                    </span>
                    <p className="mt-1 text-zinc-300 leading-relaxed">
                      {c.answer}
                    </p>
                  </div>
                </div>

                {/* Context passages */}
                <div className="mt-3 rounded-lg border border-white/[0.04] bg-[#0e1017] p-3 text-xs">
                  <span className="font-mono text-[9px] uppercase tracking-wider text-zinc-500">
                    Retrieved Passages ({c.retrieved_documents.length})
                  </span>
                  <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {c.retrieved_documents.map((doc, dIdx) => {
                      const isTarget = c.relevant_document_ids.includes(doc.id);
                      return (
                        <div
                          key={doc.id}
                          className={`rounded border p-2 text-[11px] ${
                            isTarget
                              ? "border-emerald-500/30 bg-emerald-500/5 text-emerald-200"
                              : "border-white/[0.06] bg-white/[0.02] text-zinc-400"
                          }`}
                        >
                          <div className="flex items-center justify-between text-[10px] mb-1 font-mono">
                            <span className="font-bold">{doc.id}</span>
                            <span
                              className={`rounded px-1.5 py-0.2 text-[9px] ${
                                isTarget
                                  ? "bg-emerald-500/20 text-emerald-300"
                                  : "bg-zinc-800 text-zinc-500"
                              }`}
                            >
                              {isTarget ? "Relevant Ground Truth" : `Distractor #${dIdx}`}
                            </span>
                          </div>
                          <p className="line-clamp-2 leading-relaxed">
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
