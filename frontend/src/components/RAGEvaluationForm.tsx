
import { useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Play,
  Plus,
  Trash2,
  XCircle,
} from "lucide-react";
import axios from "axios";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

interface DocumentInput {
  id: string;
  text: string;
}

interface SentenceAnalysis {
  sentence: string;
  support_score: number;
  best_context_index: number;
  supported_by_similarity_threshold: boolean;
}

interface EvaluationResult {
  query: string;
  summary: Record<string, number>;
  findings: Array<{
    type?: string;
    severity?: string;
    message?: string;
  }>;
  history?: {
    run_id?: string;
    created_at?: string;
  };
  evaluation?: {
    answer?: {
      sentence_analysis?: SentenceAnalysis[];
      answer_relevance_score?: number;
      context_support_coverage?: number;
    };
    retrieval?: {
      precision_at_k?: number;
      recall_at_k?: number;
      ndcg_at_k?: number;
    };
  };
}

// ── helpers ──────────────────────────────────────────────────────────────────

function scoreColor(score: number, supported: boolean): string {
  if (supported) return "supported";
  if (score >= 0.35) return "partial";
  return "unsupported";
}

function ScoreBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const cls =
    score >= 0.5 ? "sb-green" : score >= 0.35 ? "sb-amber" : "sb-red";
  return (
    <span
      style={{
        display: "inline-block",
        fontSize: "0.64rem",
        fontWeight: 700,
        padding: "0.08rem 0.45rem",
        borderRadius: "999px",
        border: `1px solid ${
          cls === "sb-green"
            ? "rgba(52,211,153,0.35)"
            : cls === "sb-amber"
              ? "rgba(251,191,36,0.35)"
              : "rgba(248,113,113,0.35)"
        }`,
        background:
          cls === "sb-green"
            ? "rgba(52,211,153,0.1)"
            : cls === "sb-amber"
              ? "rgba(251,191,36,0.1)"
              : "rgba(248,113,113,0.1)",
        color:
          cls === "sb-green"
            ? "#34d399"
            : cls === "sb-amber"
              ? "#fbbf24"
              : "#f87171",
      }}
    >
      {pct}%
    </span>
  );
}

function MetricPill({
  label,
  value,
}: {
  label: string;
  value: number | undefined;
}) {
  if (value == null || !Number.isFinite(value)) return null;
  const display = value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
  const good = value >= 0.5;
  return (
    <div
      style={{
        background: "rgba(0,0,0,0.3)",
        border: "1px solid rgba(255,255,255,0.07)",
        borderRadius: "0.6rem",
        padding: "0.6rem 0.75rem",
      }}
    >
      <p style={{ fontSize: "0.67rem", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "0.25rem" }}>
        {label}
      </p>
      <p style={{ fontSize: "1.1rem", fontWeight: 700, color: good ? "#34d399" : "#fbbf24", letterSpacing: "-0.02em" }}>
        {display}
      </p>
    </div>
  );
}

function SentenceAttributionPanel({
  sentences,
  documents,
}: {
  sentences: SentenceAnalysis[];
  documents: DocumentInput[];
}) {
  const [openIdx, setOpenIdx] = useState<number | null>(null);
  if (!sentences.length) return null;

  const DOTS: Record<string, string> = {
    supported: "#34d399",
    partial: "#fbbf24",
    unsupported: "#f87171",
  };

  return (
    <div
      style={{
        marginTop: "1.5rem",
        borderRadius: "0.9rem",
        border: "1px solid rgba(139,92,246,0.2)",
        background: "linear-gradient(135deg,rgba(139,92,246,0.05) 0%,rgba(13,20,36,0.95) 100%)",
        padding: "1.1rem 1.15rem",
      }}
    >
      <h4 style={{ fontSize: "0.9rem", fontWeight: 600, color: "#c4b5fd", marginBottom: "0.3rem" }}>
        Sentence-Level Context Attribution
      </h4>
      <p style={{ fontSize: "0.77rem", color: "#94a3b8", lineHeight: 1.5, marginBottom: "0.85rem" }}>
        Each answer sentence is scored by how well it is supported by the retrieved documents.
        Click a sentence to reveal the best-matching context chunk.
      </p>

      {/* Legend */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.7rem", marginBottom: "0.9rem" }}>
        {(["supported", "partial", "unsupported"] as const).map((tier) => (
          <span key={tier} style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.71rem", color: "#94a3b8" }}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: DOTS[tier], display: "inline-block" }} />
            {tier === "supported" ? "Supported (≥50%)" : tier === "partial" ? "Partial (35–49%)" : "Unsupported (<35%)"}
          </span>
        ))}
      </div>

      {/* Sentence cards */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.45rem" }}>
        {sentences.map((item, i) => {
          const tier = scoreColor(item.support_score, item.supported_by_similarity_threshold);
          const isOpen = openIdx === i;
          const matchedDoc = documents[item.best_context_index];
          const borderCol =
            tier === "supported"
              ? "rgba(52,211,153,0.25)"
              : tier === "partial"
                ? "rgba(251,191,36,0.25)"
                : "rgba(248,113,113,0.3)";
          const bgCol =
            tier === "supported"
              ? "rgba(52,211,153,0.04)"
              : tier === "partial"
                ? "rgba(251,191,36,0.04)"
                : "rgba(248,113,113,0.05)";
          const accentCol = DOTS[tier];

          return (
            <div
              key={i}
              style={{ borderRadius: "0.55rem", border: `1px solid ${borderCol}`, background: bgCol, overflow: "hidden" }}
            >
              <button
                type="button"
                onClick={() => setOpenIdx(isOpen ? null : i)}
                aria-expanded={isOpen}
                style={{
                  width: "100%",
                  display: "flex",
                  alignItems: "flex-start",
                  gap: "0.55rem",
                  padding: "0.6rem 0.7rem",
                  textAlign: "left",
                  background: "transparent",
                  cursor: "pointer",
                  border: "none",
                }}
              >
                <span style={{ flexShrink: 0, fontSize: "0.63rem", fontWeight: 700, fontFamily: "monospace", color: "#64748b", paddingTop: "0.1rem" }}>
                  S{i + 1}
                </span>
                <span style={{ flex: 1, fontSize: "0.79rem", color: "#e2e8f0", lineHeight: 1.55 }}>
                  {item.sentence}
                </span>
                <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", flexShrink: 0 }}>
                  <ScoreBadge score={item.support_score} />
                  {isOpen
                    ? <ChevronUp size={13} style={{ color: "#64748b" }} />
                    : <ChevronDown size={13} style={{ color: "#64748b" }} />}
                </div>
              </button>

              {isOpen && (
                <div
                  style={{
                    padding: "0.7rem 0.8rem 0.8rem",
                    borderTop: "1px solid rgba(255,255,255,0.06)",
                  }}
                >
                  <p style={{ fontSize: "0.7rem", fontWeight: 600, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "0.45rem" }}>
                    Best match — {matchedDoc ? `"${matchedDoc.id}"` : `index ${item.best_context_index}`}
                  </p>
                  <blockquote
                    style={{
                      fontSize: "0.77rem",
                      color: "#cbd5e1",
                      lineHeight: 1.6,
                      background: "rgba(0,0,0,0.25)",
                      borderLeft: `3px solid ${accentCol}`,
                      padding: "0.55rem 0.7rem",
                      borderRadius: "0 0.35rem 0.35rem 0",
                      margin: "0 0 0.55rem",
                      maxHeight: "110px",
                      overflowY: "auto",
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                    }}
                  >
                    {matchedDoc?.text ?? "Context not available in current session."}
                  </blockquote>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.45rem", fontSize: "0.74rem", color: "#64748b" }}>
                    <span>Cosine similarity:</span>
                    <span style={{ fontWeight: 700, color: accentCol }}>
                      {(item.support_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  {!item.supported_by_similarity_threshold && (
                    <p
                      style={{
                        fontSize: "0.72rem",
                        color: "#fca5a5",
                        marginTop: "0.45rem",
                        padding: "0.38rem 0.55rem",
                        borderRadius: "0.35rem",
                        background: "rgba(248,113,113,0.1)",
                      }}
                    >
                      ⚠ Sentence did not meet the support threshold — review for potential hallucination.
                    </p>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function RAGEvaluationForm({
  onComplete,
}: {
  onComplete: () => void;
}) {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [documents, setDocuments] = useState<DocumentInput[]>([
    { id: "doc-1", text: "" },
  ]);
  const [relevantIds, setRelevantIds] = useState("doc-1");
  const [k, setK] = useState(5);
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  function updateDocument(
    index: number,
    field: keyof DocumentInput,
    value: string,
  ) {
    setDocuments((current) =>
      current.map((doc, i) =>
        i === index ? { ...doc, [field]: value } : doc,
      ),
    );
  }

  function addDocument() {
    setDocuments((current) => [
      ...current,
      { id: `doc-${current.length + 1}`, text: "" },
    ]);
  }

  function removeDocument(index: number) {
    setDocuments((current) => {
      const remaining = current.filter((_, i) => i !== index);
      setRelevantIds((ids) =>
        ids
          .split(",")
          .map((id) => id.trim())
          .filter((id) => remaining.some((doc) => doc.id === id))
          .join(", "),
      );
      return remaining;
    });
  }

  async function submitEvaluation(
    event: React.FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();
    setError("");
    setResult(null);

    const cleanDocuments = documents.map((doc) => ({
      id: doc.id.trim(),
      text: doc.text.trim(),
    }));

    if (
      cleanDocuments.length === 0 ||
      cleanDocuments.some((doc) => !doc.id || !doc.text)
    ) {
      setError("Every document needs a unique ID and non-empty text.");
      return;
    }

    if (new Set(cleanDocuments.map((doc) => doc.id)).size !== cleanDocuments.length) {
      setError("Document IDs must be unique.");
      return;
    }

    const relevantDocumentIds = relevantIds
      .split(",")
      .map((id) => id.trim())
      .filter(Boolean);

    if (
      relevantDocumentIds.some(
        (id) => !cleanDocuments.some((doc) => doc.id === id),
      )
    ) {
      setError("Relevant document IDs must match IDs in the document list.");
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post<EvaluationResult>(
        `${API_BASE}/api/v1/rag/evaluate`,
        {
          query,
          answer,
          retrieved_documents: cleanDocuments,
          relevant_document_ids: relevantDocumentIds,
          k,
          relevance_threshold: 0.45,
          support_threshold: 0.5,
        },
      );

      setResult(response.data);
      onComplete();
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(
          typeof detail === "string"
            ? detail
            : `Evaluation request failed (${err.response?.status ?? "network error"}).`,
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
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white">
          Run RAG Evaluation
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Evaluate retrieval quality and how well an answer is supported by its context.
        </p>
      </div>

      <form onSubmit={submitEvaluation} className="space-y-5">
        <div>
          <label className="mb-2 block text-sm text-slate-300">
            User query
          </label>
          <textarea
            required
            maxLength={2000}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="min-h-20 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-cyan-400"
            placeholder="What question did the RAG system receive?"
          />
        </div>

        <div>
          <label className="mb-2 block text-sm text-slate-300">
            Generated answer
          </label>
          <textarea
            required
            maxLength={10000}
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            className="min-h-28 w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-cyan-400"
            placeholder="Paste the answer generated by your RAG system..."
          />
        </div>

        <div>
          <div className="mb-3 flex items-center justify-between gap-3">
            <label className="text-sm text-slate-300">
              Retrieved documents
            </label>
            <button
              type="button"
              onClick={addDocument}
              disabled={documents.length >= 100}
              className="flex items-center gap-1 text-sm text-cyan-300 disabled:opacity-40"
            >
              <Plus size={15} /> Add document
            </button>
          </div>

          <div className="space-y-4">
            {documents.map((doc, index) => (
              <div
                key={`${index}-${doc.id}`}
                className="rounded-xl border border-slate-800 bg-slate-950/60 p-4"
              >
                <div className="mb-3 flex items-center justify-between gap-3">
                  <label className="text-xs text-slate-400">
                    Document {index + 1}
                  </label>
                  <button
                    type="button"
                    onClick={() => removeDocument(index)}
                    disabled={documents.length === 1}
                    aria-label={`Remove document ${index + 1}`}
                    className="text-slate-500 hover:text-rose-300 disabled:opacity-30"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>

                <input
                  required
                  value={doc.id}
                  onChange={(e) =>
                    updateDocument(index, "id", e.target.value)
                  }
                  placeholder="Document ID"
                  className="mb-3 w-full rounded-lg border border-slate-700 bg-slate-900 p-2.5 text-sm outline-none focus:border-cyan-400"
                />

                <textarea
                  required
                  maxLength={10000}
                  value={doc.text}
                  onChange={(e) =>
                    updateDocument(index, "text", e.target.value)
                  }
                  placeholder="Document content..."
                  className="min-h-24 w-full rounded-lg border border-slate-700 bg-slate-900 p-3 text-sm outline-none focus:border-cyan-400"
                />
              </div>
            ))}
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className="mb-2 block text-sm text-slate-300">
              Relevant document IDs
            </label>
            <input
              value={relevantIds}
              onChange={(e) => setRelevantIds(e.target.value)}
              placeholder="doc-1, doc-2"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-cyan-400"
            />
            <p className="mt-1 text-xs text-slate-500">
              Separate IDs with commas. Leave empty if none are known to be relevant.
            </p>
          </div>

          <div>
            <label className="mb-2 block text-sm text-slate-300">
              Evaluation K
            </label>
            <input
              type="number"
              min={1}
              max={100}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              required
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm outline-none focus:border-cyan-400"
            />
          </div>
        </div>

        {error && (
          <div className="flex gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">
            <AlertCircle size={18} className="shrink-0" />
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="flex items-center gap-2 rounded-lg bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 hover:bg-cyan-300 disabled:opacity-50"
        >
          <Play size={16} />
          {loading ? "Evaluating..." : "Evaluate and save"}
        </button>
      </form>

      {result && (
        <div
          style={{
            marginTop: "1.5rem",
            borderRadius: "0.85rem",
            border: "1px solid rgba(52,211,153,0.2)",
            background: "rgba(52,211,153,0.04)",
            padding: "1.1rem 1.2rem",
          }}
        >
          {/* Heading */}
          <div style={{ display: "flex", alignItems: "center", gap: "0.45rem", fontSize: "0.88rem", fontWeight: 600, color: "#6ee7b7", marginBottom: "0.7rem" }}>
            <CheckCircle2 size={15} />
            Evaluation completed
          </div>

          {result.history?.run_id && (
            <p style={{ fontSize: "0.77rem", color: "#64748b", marginBottom: "0.9rem" }}>
              Run ID: <span style={{ fontFamily: "monospace", color: "#e2e8f0" }}>{result.history.run_id}</span>
            </p>
          )}

          {/* Metric pills */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(130px,1fr))", gap: "0.55rem", marginBottom: "0.9rem" }}>
            <MetricPill label="Precision@K" value={result.evaluation?.retrieval?.precision_at_k} />
            <MetricPill label="Recall@K" value={result.evaluation?.retrieval?.recall_at_k} />
            <MetricPill label="NDCG@K" value={result.evaluation?.retrieval?.ndcg_at_k} />
            <MetricPill label="Answer Relevance" value={result.evaluation?.answer?.answer_relevance_score} />
            <MetricPill label="Context Support" value={result.evaluation?.answer?.context_support_coverage} />
          </div>

          {/* Findings */}
          {result.findings?.length > 0 && (
            <div style={{ marginBottom: "0.5rem" }}>
              <p style={{ fontSize: "0.68rem", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: "0.4rem" }}>Findings</p>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                {result.findings.map((f, i) => (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: "0.45rem",
                      fontSize: "0.77rem",
                      padding: "0.45rem 0.6rem",
                      borderRadius: "0.4rem",
                      background: "rgba(0,0,0,0.2)",
                      border: "1px solid rgba(255,255,255,0.05)",
                      color: "#cbd5e1",
                    }}
                  >
                    {f.severity === "high" ? (
                      <XCircle size={14} style={{ color: "#f87171", flexShrink: 0 }} />
                    ) : f.severity === "medium" || f.severity === "review" ? (
                      <AlertCircle size={14} style={{ color: "#fbbf24", flexShrink: 0 }} />
                    ) : (
                      <CheckCircle2 size={14} style={{ color: "#34d399", flexShrink: 0 }} />
                    )}
                    <span>{f.message}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Attribution highlighter */}
          <SentenceAttributionPanel
            sentences={result.evaluation?.answer?.sentence_analysis ?? []}
            documents={documents}
          />
        </div>
      )}
    </section>
  );
}
