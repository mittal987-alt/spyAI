
import { useState } from "react";
import { AlertCircle, Play, Plus, Trash2 } from "lucide-react";
import axios from "axios";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

interface DocumentInput {
  id: string;
  text: string;
}

interface EvaluationResult {
  query: string;
  summary: Record<string, number>;
  findings: Array<{
    title?: string;
    severity?: string;
    description?: string;
  }>;
  history?: {
    run_id?: string;
    created_at?: string;
  };
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
        <div className="mt-6 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
          <h3 className="font-medium text-emerald-300">
            Evaluation completed
          </h3>
          <p className="mt-2 text-sm text-slate-400">
            Run ID:{" "}
            <span className="font-mono text-slate-200">
              {result.history?.run_id ?? "Saved; run ID not returned"}
            </span>
          </p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {Object.entries(result.summary ?? {}).map(([key, value]) => (
              <div key={key} className="rounded-lg bg-slate-900 p-3">
                <p className="text-xs text-slate-500">{key.replaceAll("_", " ")}</p>
                <p className="mt-1 text-lg text-white">
                  {Number.isFinite(value) && Math.abs(value) <= 1
                    ? `${(value * 100).toFixed(1)}%`
                    : value}
                </p>
              </div>
            ))}
          </div>
          {result.findings?.length > 0 && (
            <p className="mt-4 text-sm text-amber-300">
              {result.findings.length} finding(s) reported. Review the evaluation details.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
