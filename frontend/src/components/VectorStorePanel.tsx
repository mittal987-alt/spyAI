import { useState, useEffect } from "react";
import {
  Database,
  Search,
  Plus,
  Trash2,
  RefreshCw,
  Sparkles,
  Layers,
  CheckCircle2,
} from "lucide-react";
import {
  getVectorStoreStatus,
  indexVectorDocuments,
  searchVectorStore,
  clearVectorStore,
} from "../api";
import type { VectorStoreStatus, VectorDocumentResult } from "../types";

const SAMPLE_KNOWLEDGE_DOCS = [
  {
    text: "Population Stability Index (PSI) measures the shift in distribution between a baseline dataset and production inference data over time. A PSI above 0.25 indicates critical feature drift.",
    tag: "drift-detection",
  },
  {
    text: "Kolmogorov-Smirnov (KS) two-sample non-parametric test determines whether two continuous distributions differ significantly, producing a p-value below 0.05 on drifted columns.",
    tag: "statistics",
  },
  {
    text: "Retrieval-Augmented Generation (RAG) grounds LLM responses by retrieving relevant passages from a vector database before text generation, reducing hallucinations.",
    tag: "rag-architecture",
  },
  {
    text: "Regularization techniques like Dropout, L2 weight decay, and early stopping prevent neural networks from overfitting to noisy training datasets.",
    tag: "deep-learning",
  },
  {
    text: "Cosine similarity measures the cosine of the angle between two multi-dimensional dense vectors, scoring lexical and conceptual orientation from 0 to 1.",
    tag: "vector-search",
  },
];

export function VectorStorePanel() {
  const [status, setStatus] = useState<VectorStoreStatus | null>(null);
  const [newText, setNewText] = useState<string>("");
  const [newTag, setNewTag] = useState<string>("custom");
  const [searchQuery, setSearchQuery] = useState<string>("How does PSI detect feature drift?");
  const [useRerank, setUseRerank] = useState<boolean>(true);
  const [searchResults, setSearchResults] = useState<VectorDocumentResult[]>([]);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [isIndexing, setIsIndexing] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [notice, setNotice] = useState<string | null>(null);

  const loadStatus = async () => {
    try {
      setIsLoading(true);
      const res = await getVectorStoreStatus();
      setStatus(res);
    } catch (err) {
      console.error("Failed to load vector store status", err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadStatus();
  }, []);

  const handleIndexSingle = async () => {
    if (!newText.trim()) return;
    setIsIndexing(true);
    try {
      await indexVectorDocuments({
        texts: [newText.trim()],
        metadatas: [{ tag: newTag.trim() || "general" }],
      });
      setNewText("");
      setNotice("Document indexed successfully.");
      setTimeout(() => setNotice(null), 3000);
      await loadStatus();
    } catch (err) {
      console.error("Indexing failed", err);
    } finally {
      setIsIndexing(false);
    }
  };

  const handleLoadSamples = async () => {
    setIsIndexing(true);
    try {
      await indexVectorDocuments({
        texts: SAMPLE_KNOWLEDGE_DOCS.map((d) => d.text),
        metadatas: SAMPLE_KNOWLEDGE_DOCS.map((d) => ({ tag: d.tag })),
      });
      setNotice("Loaded 5 standard machine learning benchmark passages.");
      setTimeout(() => setNotice(null), 3000);
      await loadStatus();
    } catch (err) {
      console.error("Loading samples failed", err);
    } finally {
      setIsIndexing(false);
    }
  };

  const handleClear = async () => {
    if (!confirm("Are you sure you want to clear all indexed passages from the vector store?")) {
      return;
    }
    try {
      await clearVectorStore();
      setSearchResults([]);
      setNotice("Vector store cleared.");
      setTimeout(() => setNotice(null), 3000);
      await loadStatus();
    } catch (err) {
      console.error("Clear failed", err);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    try {
      const res = await searchVectorStore({
        query: searchQuery,
        top_k: 5,
        rerank: useRerank,
      });
      setSearchResults(res.results);
    } catch (err) {
      console.error("Search failed", err);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 p-4 rounded-xl bg-slate-900 border border-slate-800">
        <div>
          <h2 className="text-xl font-semibold text-slate-100 flex items-center gap-2">
            <Database className="w-5 h-5 text-emerald-400" />
            Vector Store & Dense Semantic Retrieval
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Embed knowledge passages into a persistent vector store, execute cosine nearest-neighbor search, and inspect cross-encoder neural reranking.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadStatus}
            disabled={isLoading}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
            Refresh
          </button>
          <button
            onClick={handleClear}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-rose-950/40 hover:bg-rose-900/60 text-rose-300 border border-rose-800/60 transition"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear
          </button>
        </div>
      </div>

      {notice && (
        <div className="p-3 rounded-lg bg-emerald-950/30 border border-emerald-800/80 text-emerald-300 text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>{notice}</span>
        </div>
      )}

      {/* Status Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400">Indexed Passages</span>
          <p className="text-2xl font-bold text-slate-100 mt-1">
            {status ? status.document_count.toLocaleString() : "--"}
          </p>
          <span className="text-[11px] text-slate-500">Persistent SQLite Ledger</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400">Embedding Dimensions</span>
          <p className="text-2xl font-bold text-indigo-400 mt-1">
            {status ? `${status.embedding_dimension}-dim` : "64-dim"}
          </p>
          <span className="text-[11px] text-slate-500">Normalized Dense Vector</span>
        </div>

        <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
          <span className="text-xs text-slate-400">Retrieval & Rerank Engine</span>
          <p className="text-sm font-semibold text-emerald-400 mt-2 truncate">
            {status ? status.backend : "Local Dense Embeddings"}
          </p>
          <span className="text-[11px] text-slate-500">Cosine Similarity + Cross-Encoder</span>
        </div>
      </div>

      {/* Main Action Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Panel 1: Document Indexer */}
        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Plus className="w-4 h-4 text-emerald-400" />
              Document Knowledge Ingestion
            </h3>
            <button
              onClick={handleLoadSamples}
              disabled={isIndexing}
              className="text-xs text-emerald-400 hover:text-emerald-300 font-medium flex items-center gap-1"
            >
              <Sparkles className="w-3.5 h-3.5" />
              Load Sample Knowledge Base
            </button>
          </div>

          <div>
            <label className="text-xs font-medium text-slate-400 block mb-1">
              Passage Text
            </label>
            <textarea
              rows={4}
              value={newText}
              onChange={(e) => setNewText(e.target.value)}
              placeholder="Paste custom document passage to embed..."
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-100 focus:outline-none focus:border-emerald-500"
            />
          </div>

          <div className="flex gap-2">
            <input
              type="text"
              value={newTag}
              onChange={(e) => setNewTag(e.target.value)}
              placeholder="Metadata Tag (e.g., drift, policy, api)"
              className="w-1/2 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 font-mono"
            />
            <button
              onClick={handleIndexSingle}
              disabled={isIndexing || !newText.trim()}
              className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white transition"
            >
              {isIndexing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
              {isIndexing ? "Embedding..." : "Index Passage"}
            </button>
          </div>
        </div>

        {/* Panel 2: Semantic Search & Reranking */}
        <div className="p-5 rounded-xl bg-slate-900 border border-slate-800 space-y-4">
          <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <Search className="w-4 h-4 text-blue-400" />
            Semantic Search & Neural Reranker
          </h3>

          <div className="space-y-2">
            <label className="text-xs font-medium text-slate-400 block">
              Search Query
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Ask semantic question..."
                className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-blue-500"
              />
              <button
                onClick={handleSearch}
                disabled={isSearching || !searchQuery.trim()}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white transition"
              >
                {isSearching ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
                Search
              </button>
            </div>
          </div>

          <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={useRerank}
              onChange={(e) => setUseRerank(e.target.checked)}
              className="rounded bg-slate-950 border-slate-800 text-blue-600 focus:ring-0"
            />
            <span className="flex items-center gap-1">
              <Layers className="w-3.5 h-3.5 text-indigo-400" />
              Enable Cross-Encoder Neural Reranking
            </span>
          </label>

          {/* Results list */}
          <div className="space-y-2 mt-3">
            {searchResults.length === 0 ? (
              <p className="text-xs text-slate-500 p-4 rounded-lg bg-slate-950/60 border border-slate-800/80 text-center">
                Enter a query above to see ranked semantic document matches.
              </p>
            ) : (
              searchResults.map((r, idx) => (
                <div
                  key={r.doc_id}
                  className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5"
                >
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-5 rounded-full bg-blue-950/80 text-blue-400 font-mono text-[10px] flex items-center justify-center font-bold">
                        #{idx + 1}
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300">
                        tag: {String(r.metadata?.tag || "general")}
                      </span>
                    </div>

                    <div className="flex items-center gap-3 font-mono text-[11px]">
                      <span className="text-slate-400">
                        Dense: <strong className="text-slate-200">{(r.similarity_score * 100).toFixed(1)}%</strong>
                      </span>
                      {r.rerank_score != null && (
                        <span className="text-indigo-400">
                          Rerank: <strong>{(r.rerank_score * 100).toFixed(1)}%</strong>
                        </span>
                      )}
                    </div>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed font-sans">
                    {r.text}
                  </p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
