import { useCallback, useEffect, useState, useMemo } from "react";
import {
  Activity,
  AlertTriangle,
  Boxes,
  CheckCircle2,
  FileSpreadsheet,
  FlaskConical,
  LayoutGrid,
  RefreshCw,
  Shield,
  X,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { getRagDashboard, getRagHistory } from "./api";
import type {
  RagDashboardResponse,
  RagHistoryItem,
} from "./types";
import { useSSE } from "./hooks/useSSE";

import SystemHealth from "./components/SystemHealth";
import DiagnosticsOverview from "./components/DiagnosticsOverview";
import RAGEvaluationForm from "./components/RAGEvaluationForm";
import DatasetExplorer from "./components/DatasetExplorer";
import RAGComparison from "./components/RAGComparison";
import DeepLearningDiagnosis from "./components/DeepLearningDiagnosis";
import DeepLearningComparison from "./components/DeepLearningComparison";
import SemanticMonitoring from "./components/SemanticMonitoring";
import DriftAlerts from "./components/DriftAlerts";
import TabularExperiment from "./components/TabularExperiment";
import CSVDriftUpload from "./components/CSVDriftUpload";
import ExportReport from "./components/ExportReport";
import PromptGuardrails from "./components/PromptGuardrails";
import RedTeamingSuite from "./components/RedTeamingSuite";
import WebhookAlerts from "./components/WebhookAlerts";
import GoldenDatasetGenerator from "./components/GoldenDatasetGenerator";
import DriftSimulator from "./components/DriftSimulator";
import ModelCardExport from "./components/ModelCardExport";
import HallucinationBenchmark from "./components/HallucinationBenchmark";
import AlertRulesEngine from "./components/AlertRulesEngine";
import ModelRegistry from "./components/ModelRegistry";
import DriftExplanationDrawer from "./components/DriftExplanationDrawer";
import { LLMTracingPanel } from "./components/LLMTracingPanel";
import { AutonomousAgentsPanel } from "./components/AutonomousAgentsPanel";
import { LLMJudgePanel } from "./components/LLMJudgePanel";
import { VectorStorePanel } from "./components/VectorStorePanel";

type MainTab = "overview" | "rag" | "drift" | "safety" | "models";

const humanTabs: { id: MainTab; label: string; icon: typeof LayoutGrid; subtitle: string }[] = [
  {
    id: "overview",
    label: "Overview",
    icon: LayoutGrid,
    subtitle: "Health summary and quick launch shortcuts",
  },
  {
    id: "rag",
    label: "AI Accuracy (RAG)",
    icon: FlaskConical,
    subtitle: "Test answers, check hallucinations and generate benchmarks",
  },
  {
    id: "drift",
    label: "Data Health & Drift",
    icon: Activity,
    subtitle: "Detect when real-world data changes from baseline training",
  },
  {
    id: "safety",
    label: "Safety & Security",
    icon: Shield,
    subtitle: "Scan for private data leaks (PII) and malicious prompts",
  },
  {
    id: "models",
    label: "Models & Audit",
    icon: Boxes,
    subtitle: "Manage model versions, compliance reports and alert notifications",
  },
];

const ragSubTabs = [
  { id: "judge-panel", label: "LLM-as-a-Judge (3 Jury)" },
  { id: "vector-store", label: "Vector Search & Reranker" },
  { id: "eval-form", label: "Test AI Answer" },
  { id: "hallucination", label: "Hallucination Check" },
  { id: "benchmark-gen", label: "Auto-Generate Benchmarks" },
  { id: "trends", label: "Accuracy Trends & History" },
  { id: "compare", label: "Compare Two Answers" },
  { id: "semantic", label: "Text Vocabulary Shift" },
] as const;

const driftSubTabs = [
  { id: "csv-upload", label: "Upload CSV to Check" },
  { id: "ai-explain", label: "AI Explains Drift" },
  { id: "alerts", label: "Active Shift Alerts" },
  { id: "tabular", label: "Tabular Drift Lab" },
  { id: "explorer", label: "Dataset Column Explorer" },
  { id: "simulator", label: "Data Stress Simulator" },
] as const;

const safetySubTabs = [
  { id: "autonomous-agents", label: "Self-Healing & Auto-Immunize" },
  { id: "guardrails", label: "Prompt Privacy & Scanner" },
  { id: "red-team", label: "Security Threat Test" },
] as const;

const modelsSubTabs = [
  { id: "tracing", label: "LLM Tracing & Cost" },
  { id: "registry", label: "Model Versions & Promotion" },
  { id: "deep-learning", label: "Neural Network Insights" },
  { id: "compliance", label: "Model Card (EU AI Act)" },
  { id: "rules", label: "Smart Alert Rules" },
  { id: "webhooks", label: "Slack & Discord Alerts" },
  { id: "export", label: "Download Audit Report" },
] as const;

function formatScore(val: number | null | undefined): string {
  if (val == null || !Number.isFinite(val)) return "—";
  return (val * 100).toFixed(1) + "%";
}

function formatDate(val: string): string {
  const d = new Date(val);
  if (Number.isNaN(d.getTime())) return val;
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function App() {
  const [activeTab, setActiveTab] = useState<MainTab>("overview");
  const [ragSubTab, setRagSubTab] = useState<string>("eval-form");
  const [driftSubTab, setDriftSubTab] = useState<string>("csv-upload");
  const [safetySubTab, setSafetySubTab] = useState<string>("guardrails");
  const [modelsSubTab, setModelsSubTab] = useState<string>("registry");

  const [dashboard, setDashboard] = useState<RagDashboardResponse | null>(null);
  const [history, setHistory] = useState<RagHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notification, setNotification] = useState<string | null>(null);

  const loadDashboard = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      const [dash, hist] = await Promise.all([
        getRagDashboard(20),
        getRagHistory(20),
      ]);
      setDashboard(dash);
      setHistory(hist);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unable to reach diagnostics backend.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  const handleSSEEvent = useCallback(
    (event: { type: string; [key: string]: any }) => {
      if (event.type === "rag_evaluated") {
        setNotification("New evaluation completed and saved.");
        void loadDashboard(true);
        setTimeout(() => setNotification(null), 4000);
      } else if (event.type === "drift_completed") {
        setNotification("Data drift analysis completed.");
        void loadDashboard(true);
        setTimeout(() => setNotification(null), 4000);
      }
    },
    [loadDashboard],
  );

  const { isConnected: sseConnected } = useSSE({ onEvent: handleSSEEvent });

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard]);

  // Navigate directly from overview cards
  const handleSelectModule = (moduleId: string) => {
    switch (moduleId) {
      case "judge-panel":
        setActiveTab("rag");
        setRagSubTab("judge-panel");
        break;
      case "vector-store":
        setActiveTab("rag");
        setRagSubTab("vector-store");
        break;
      case "evaluate":
        setActiveTab("rag");
        setRagSubTab("eval-form");
        break;
      case "hallucination":
        setActiveTab("rag");
        setRagSubTab("hallucination");
        break;
      case "golden-dataset":
        setActiveTab("rag");
        setRagSubTab("benchmark-gen");
        break;
      case "rag-dashboard":
        setActiveTab("rag");
        setRagSubTab("trends");
        break;
      case "rag-comparison":
        setActiveTab("rag");
        setRagSubTab("compare");
        break;
      case "csv-drift":
        setActiveTab("drift");
        setDriftSubTab("csv-upload");
        break;
      case "drift-explanation":
        setActiveTab("drift");
        setDriftSubTab("ai-explain");
        break;
      case "drift-alerts":
        setActiveTab("drift");
        setDriftSubTab("alerts");
        break;
      case "dataset-explorer":
        setActiveTab("drift");
        setDriftSubTab("explorer");
        break;
      case "drift-simulator":
        setActiveTab("drift");
        setDriftSubTab("simulator");
        break;
      case "guardrails":
        setActiveTab("safety");
        setSafetySubTab("guardrails");
        break;
      case "red-teaming":
        setActiveTab("safety");
        setSafetySubTab("red-team");
        break;
      case "autonomous-agents":
        setActiveTab("safety");
        setSafetySubTab("autonomous-agents");
        break;
      case "llm-tracing":
        setActiveTab("models");
        setModelsSubTab("tracing");
        break;
      case "model-registry":
        setActiveTab("models");
        setModelsSubTab("registry");
        break;
      case "model-card":
        setActiveTab("models");
        setModelsSubTab("compliance");
        break;
      case "webhooks":
        setActiveTab("models");
        setModelsSubTab("webhooks");
        break;
      case "alert-rules":
        setActiveTab("models");
        setModelsSubTab("rules");
        break;
      case "export-report":
        setActiveTab("models");
        setModelsSubTab("export");
        break;
      default:
        setActiveTab("overview");
    }
  };

  const chartData = useMemo(() => {
    return [...history]
      .reverse()
      .map((run, i) => ({
        run: `Test ${i + 1}`,
        precision: run.summary?.precision_at_k ?? null,
        recall: run.summary?.recall_at_k ?? null,
        relevance: run.summary?.answer_relevance_score ?? null,
      }));
  }, [history]);

  const avgPrecision = dashboard?.metrics?.precision_at_k?.average;
  const avgRelevance = dashboard?.metrics?.answer_relevance_score?.average;

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 font-sans">
      {/* Friendly Top Header */}
      <header className="border-b border-slate-800 bg-[#0d121f] px-4 py-3 sm:px-6">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-600 text-white shadow-md shadow-blue-500/20 font-bold">
              M
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-base text-white tracking-tight">
                  ModelAutopsy AI
                </span>
                <span className="rounded-full bg-emerald-500/10 border border-emerald-500/30 px-2 py-0.5 text-[10px] font-medium text-emerald-400">
                  {sseConnected ? "● Live Connected" : "● Offline"}
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Simple, transparent AI accuracy & data health monitoring
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => void loadDashboard(true)}
              disabled={refreshing || loading}
              className="btn-secondary text-xs"
            >
              <RefreshCw size={13} className={refreshing ? "animate-spin text-blue-400" : ""} />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        {/* Human Navigation Tabs */}
        <div className="mx-auto max-w-7xl mt-4 flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
          {humanTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 rounded-lg px-3.5 py-2 font-medium transition ${
                  isActive
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-400 hover:bg-slate-800/80 hover:text-slate-200"
                }`}
              >
                <Icon size={15} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </header>

      {/* Live Toast */}
      {notification && (
        <div className="fixed bottom-4 right-4 z-50 flex items-center gap-2 rounded-lg border border-blue-500/40 bg-slate-900 px-4 py-2.5 text-xs text-blue-200 shadow-xl">
          <CheckCircle2 size={15} className="text-emerald-400" />
          <span>{notification}</span>
          <button
            type="button"
            onClick={() => setNotification(null)}
            className="ml-2 text-slate-400 hover:text-white"
          >
            <X size={12} />
          </button>
        </div>
      )}

      {/* Main Body */}
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {error && (
          <div className="mb-6 flex items-center justify-between rounded-xl border border-rose-500/30 bg-rose-500/10 p-3.5 text-xs text-rose-300">
            <div className="flex items-center gap-2">
              <AlertTriangle size={16} />
              <span>{error}</span>
            </div>
            <button
              type="button"
              onClick={() => void loadDashboard(true)}
              className="rounded bg-rose-500/20 px-2 py-1 font-medium hover:bg-rose-500/30"
            >
              Retry
            </button>
          </div>
        )}

        {/* ============================================================== */}
        {/* TAB 1: OVERVIEW (Friendly Home Dashboard)                      */}
        {/* ============================================================== */}
        {activeTab === "overview" && (
          <div className="space-y-6">
            {/* Friendly Greeting Card */}
            <div className="rounded-2xl border border-slate-800 bg-gradient-to-r from-slate-900 via-[#101729] to-slate-900 p-6 shadow-sm">
              <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                <div>
                  <h1 className="text-xl font-bold text-white sm:text-2xl">
                    Welcome to ModelAutopsy
                  </h1>
                  <p className="mt-1 text-xs text-slate-300 sm:text-sm max-w-2xl leading-relaxed">
                    Check if your AI answers questions truthfully, detect if real-world data changed, scan prompts for sensitive private info, and manage your production models.
                  </p>
                </div>

                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      setActiveTab("rag");
                      setRagSubTab("eval-form");
                    }}
                    className="btn-primary text-xs"
                  >
                    <FlaskConical size={14} />
                    <span>Test an AI Answer</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setActiveTab("drift");
                      setDriftSubTab("csv-upload");
                    }}
                    className="btn-secondary text-xs"
                  >
                    <FileSpreadsheet size={14} />
                    <span>Check Data Shift</span>
                  </button>
                </div>
              </div>

              {/* 4 Friendly KPI Health Cards */}
              <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div className="rounded-xl border border-slate-800/80 bg-slate-950/40 p-4">
                  <span className="text-xs text-slate-400">Answer Accuracy</span>
                  <p className="mt-1 text-xl font-bold text-white">
                    {formatScore(avgPrecision)}
                  </p>
                  <span className="text-[10px] text-slate-500">Retrieval precision</span>
                </div>

                <div className="rounded-xl border border-slate-800/80 bg-slate-950/40 p-4">
                  <span className="text-xs text-slate-400">Relevance Score</span>
                  <p className="mt-1 text-xl font-bold text-emerald-400">
                    {formatScore(avgRelevance)}
                  </p>
                  <span className="text-[10px] text-slate-500">Answer matches question</span>
                </div>

                <div className="rounded-xl border border-slate-800/80 bg-slate-950/40 p-4">
                  <span className="text-xs text-slate-400">Total Evaluations</span>
                  <p className="mt-1 text-xl font-bold text-white">
                    {dashboard?.runs_included ?? history.length}
                  </p>
                  <span className="text-[10px] text-slate-500">Benchmark runs saved</span>
                </div>

                <div className="rounded-xl border border-slate-800/80 bg-slate-950/40 p-4">
                  <span className="text-xs text-slate-400">Data Shift Status</span>
                  <p className="mt-1 text-xl font-bold text-blue-400">
                    Healthy
                  </p>
                  <span className="text-[10px] text-slate-500">Zero critical anomalies</span>
                </div>
              </div>
            </div>

            {/* Live System Health Component */}
            <div className="simple-card p-5">
              <SystemHealth />
            </div>

            {/* Quick-Launch Tools Directory */}
            <div className="simple-card p-5">
              <DiagnosticsOverview onSelectModule={handleSelectModule} />
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* TAB 2: AI ACCURACY (RAG & HALLUCINATIONS)                      */}
        {/* ============================================================== */}
        {activeTab === "rag" && (
          <div className="space-y-6">
            {/* Sub-navigation pills */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-base font-semibold text-white">AI Answer Quality & Accuracy</h2>
                <p className="text-xs text-slate-400">
                  Ensure answers are backed by real documents and free of made-up information.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-1.5">
                {ragSubTabs.map((st) => (
                  <button
                    key={st.id}
                    type="button"
                    onClick={() => setRagSubTab(st.id)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      ragSubTab === st.id
                        ? "bg-slate-800 text-white border border-slate-700"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {st.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Sub-view rendering */}
            {ragSubTab === "judge-panel" && (
              <section id="llm-judge">
                <LLMJudgePanel />
              </section>
            )}

            {ragSubTab === "vector-store" && (
              <section id="vector-store">
                <VectorStorePanel />
              </section>
            )}

            {ragSubTab === "eval-form" && (
              <section id="evaluate">
                <RAGEvaluationForm onComplete={() => void loadDashboard(true)} />
              </section>
            )}

            {ragSubTab === "hallucination" && (
              <section id="hallucination">
                <HallucinationBenchmark />
              </section>
            )}

            {ragSubTab === "benchmark-gen" && (
              <section id="golden-dataset">
                <GoldenDatasetGenerator />
              </section>
            )}

            {ragSubTab === "trends" && (
              <div className="space-y-6">
                <div className="simple-card p-5">
                  <h3 className="text-sm font-semibold text-white">Accuracy & Quality Over Time</h3>
                  <p className="text-xs text-slate-400 mb-4">
                    Precision (cyan), Recall (indigo), and Relevance (emerald) across recent tests.
                  </p>
                  {chartData.length === 0 ? (
                    <p className="text-xs text-slate-500 py-10 text-center">No evaluations recorded yet.</p>
                  ) : (
                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
                          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
                          <XAxis dataKey="run" tick={{ fill: "#64748b", fontSize: 11 }} />
                          <YAxis domain={[0, 1]} tick={{ fill: "#64748b", fontSize: 11 }} />
                          <Tooltip
                            contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
                            formatter={(v) => (typeof v === "number" ? (v * 100).toFixed(1) + "%" : "—")}
                          />
                          <Line type="monotone" dataKey="precision" name="Precision" stroke="#06b6d4" strokeWidth={2} dot={{ r: 3 }} />
                          <Line type="monotone" dataKey="recall" name="Recall" stroke="#6366f1" strokeWidth={2} dot={{ r: 3 }} />
                          <Line type="monotone" dataKey="relevance" name="Relevance" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                {/* History table */}
                <div className="simple-card overflow-hidden">
                  <div className="px-5 py-3 border-b border-slate-800 text-xs font-semibold text-white">
                    Recent Evaluation Runs
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead className="bg-slate-900/50 text-[10px] uppercase text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="px-4 py-2.5">Query</th>
                          <th className="px-4 py-2.5">Precision</th>
                          <th className="px-4 py-2.5">Recall</th>
                          <th className="px-4 py-2.5">Relevance</th>
                          <th className="px-4 py-2.5">Date</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {history.map((r) => (
                          <tr key={r.run_id} className="hover:bg-slate-800/30">
                            <td className="px-4 py-3 max-w-xs truncate text-slate-200">{r.query}</td>
                            <td className="px-4 py-3 font-mono text-cyan-400">{formatScore(r.summary?.precision_at_k)}</td>
                            <td className="px-4 py-3 font-mono text-indigo-400">{formatScore(r.summary?.recall_at_k)}</td>
                            <td className="px-4 py-3 font-mono text-emerald-400">{formatScore(r.summary?.answer_relevance_score)}</td>
                            <td className="px-4 py-3 text-slate-500 text-[11px]">{formatDate(r.created_at)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {ragSubTab === "compare" && (
              <section id="rag-comparison">
                <RAGComparison history={history} />
              </section>
            )}

            {ragSubTab === "semantic" && (
              <section id="semantic-monitoring">
                <SemanticMonitoring />
              </section>
            )}
          </div>
        )}

        {/* ============================================================== */}
        {/* TAB 3: DATA HEALTH & DRIFT                                     */}
        {/* ============================================================== */}
        {activeTab === "drift" && (
          <div className="space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-base font-semibold text-white">Data Quality & Drift Detection</h2>
                <p className="text-xs text-slate-400">
                  Catch data changes before they cause incorrect model predictions.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-1.5">
                {driftSubTabs.map((st) => (
                  <button
                    key={st.id}
                    type="button"
                    onClick={() => setDriftSubTab(st.id)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      driftSubTab === st.id
                        ? "bg-slate-800 text-white border border-slate-700"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {st.label}
                  </button>
                ))}
              </div>
            </div>

            {driftSubTab === "csv-upload" && (
              <section id="csv-drift">
                <CSVDriftUpload />
              </section>
            )}

            {driftSubTab === "ai-explain" && (
              <section id="drift-explanation">
                <DriftExplanationDrawer />
              </section>
            )}

            {driftSubTab === "alerts" && (
              <section id="drift-alerts">
                <DriftAlerts />
              </section>
            )}

            {driftSubTab === "tabular" && (
              <section id="tabular-experiments">
                <TabularExperiment />
              </section>
            )}

            {driftSubTab === "explorer" && (
              <section id="dataset-explorer">
                <DatasetExplorer />
              </section>
            )}

            {driftSubTab === "simulator" && (
              <section id="drift-simulator">
                <DriftSimulator />
              </section>
            )}
          </div>
        )}

        {/* ============================================================== */}
        {/* TAB 4: SAFETY & SECURITY                                       */}
        {/* ============================================================== */}
        {activeTab === "safety" && (
          <div className="space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-base font-semibold text-white">AI Safety & Guardrails</h2>
                <p className="text-xs text-slate-400">
                  Prevent leaks of private user information and block jailbreak attacks.
                </p>
              </div>

              <div className="flex items-center gap-1.5">
                {safetySubTabs.map((st) => (
                  <button
                    key={st.id}
                    type="button"
                    onClick={() => setSafetySubTab(st.id)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      safetySubTab === st.id
                        ? "bg-slate-800 text-white border border-slate-700"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {st.label}
                  </button>
                ))}
              </div>
            </div>

            {safetySubTab === "autonomous-agents" && (
              <section id="autonomous-agents">
                <AutonomousAgentsPanel />
              </section>
            )}

            {safetySubTab === "guardrails" && (
              <section id="guardrails">
                <PromptGuardrails />
              </section>
            )}

            {safetySubTab === "red-team" && (
              <section id="red-teaming">
                <RedTeamingSuite />
              </section>
            )}
          </div>
        )}

        {/* ============================================================== */}
        {/* TAB 5: MODELS & AUDIT                                          */}
        {/* ============================================================== */}
        {activeTab === "models" && (
          <div className="space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h2 className="text-base font-semibold text-white">Model Management & Compliance</h2>
                <p className="text-xs text-slate-400">
                  Track model versions, promote stages, generate audit model cards, and manage alerts.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-1.5">
                {modelsSubTabs.map((st) => (
                  <button
                    key={st.id}
                    type="button"
                    onClick={() => setModelsSubTab(st.id)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      modelsSubTab === st.id
                        ? "bg-slate-800 text-white border border-slate-700"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {st.label}
                  </button>
                ))}
              </div>
            </div>

            {modelsSubTab === "tracing" && (
              <section id="llm-tracing">
                <LLMTracingPanel />
              </section>
            )}

            {modelsSubTab === "registry" && (
              <section id="model-registry">
                <ModelRegistry />
              </section>
            )}

            {modelsSubTab === "deep-learning" && (
              <section id="deep-learning" className="space-y-6">
                <DeepLearningDiagnosis />
                <DeepLearningComparison />
              </section>
            )}

            {modelsSubTab === "compliance" && (
              <section id="model-card">
                <ModelCardExport />
              </section>
            )}

            {modelsSubTab === "rules" && (
              <section id="alert-rules">
                <AlertRulesEngine />
              </section>
            )}

            {modelsSubTab === "webhooks" && (
              <section id="webhooks">
                <WebhookAlerts />
              </section>
            )}

            {modelsSubTab === "export" && (
              <section id="export-report">
                <ExportReport />
              </section>
            )}
          </div>
        )}
      </main>

      {/* Clean, Simple Footer */}
      <footer className="mt-16 border-t border-slate-800 py-6 text-xs text-slate-500">
        <div className="mx-auto max-w-7xl px-4 flex flex-col items-center justify-between gap-3 sm:flex-row sm:px-6">
          <p>ModelAutopsy AI · Clear & Simple Machine Learning Observability</p>
          <div className="flex items-center gap-4 text-slate-400 text-[11px]">
            <span>FastAPI Backend (Port 8000)</span>
            <span>·</span>
            <span>React Frontend (Port 5173)</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
