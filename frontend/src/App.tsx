
import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bell,
  Brain,
  ChartNoAxesCombined,
  CheckCircle2,
  CircleHelp,
  Database,
  FlaskConical,
  RefreshCw,
  ShieldCheck,
  Shield,
  Sparkles,
  TrendingUp,
  Radio,
  Zap,
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
import RAGComparison from "./components/RAGComparison";
import DeepLearningDiagnosis from "./components/DeepLearningDiagnosis";
import DeepLearningComparison from "./components/DeepLearningComparison";
import SemanticMonitoring from "./components/SemanticMonitoring";
import DriftAlerts from "./components/DriftAlerts";
import TabularExperiment from "./components/TabularExperiment";
import CSVDriftUpload from "./components/CSVDriftUpload";
import ExportReport from "./components/ExportReport";
import PromptGuardrails from "./components/PromptGuardrails";
import WebhookAlerts from "./components/WebhookAlerts";
import GoldenDatasetGenerator from "./components/GoldenDatasetGenerator";
import DriftSimulator from "./components/DriftSimulator";

const metricDefinitions = [
  {
    key: "precision_at_k",
    label: "Precision@K",
    description: "Relevant documents among retrieved results",
  },
  {
    key: "recall_at_k",
    label: "Recall@K",
    description: "Relevant documents successfully retrieved",
  },
  {
    key: "ndcg_at_k",
    label: "NDCG@K",
    description: "Ranking quality",
  },
  {
    key: "answer_relevance_score",
    label: "Answer relevance",
    description: "How relevant the answer is to the query",
  },
  {
    key: "context_support_coverage",
    label: "Context support",
    description: "Answer sentences supported by retrieved context",
  },
] as const;

type MetricKey = (typeof metricDefinitions)[number]["key"];

function formatMetric(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(3);
}

function formatDate(value: string) {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString();
}

function getMetricValue(
  run: RagHistoryItem,
  key: MetricKey,
): number | null {
  const value = run.summary?.[key];

  return typeof value === "number" && Number.isFinite(value)
    ? value
    : null;
}

function SectionHeading({
  icon: Icon,
  title,
  description,
}: {
  icon: typeof Activity;
  title: string;
  description: string;
}) {
  return (
    <div className="mb-5 flex items-start gap-3">
      <div className="rounded-xl border border-cyan-400/20 bg-cyan-400/10 p-2.5 text-cyan-300">
        <Icon size={21} />
      </div>

      <div>
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        <p className="mt-1 text-sm leading-5 text-slate-400">
          {description}
        </p>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  description,
  availableRuns,
}: {
  label: string;
  value: number | null;
  description: string;
  availableRuns: number;
}) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5 transition hover:border-cyan-400/30">
      <p className="text-sm text-slate-400">{label}</p>

      <p className="mt-3 text-3xl font-semibold tracking-tight text-white">
        {formatMetric(value)}
      </p>

      <p className="mt-2 text-xs leading-5 text-slate-500">
        {description}
      </p>

      <div className="mt-4 border-t border-slate-800 pt-3">
        <span className="text-xs text-slate-500">
          {availableRuns} run{availableRuns === 1 ? "" : "s"} with data
        </span>
      </div>
    </div>
  );
}

function StatusPill({
  children,
  variant = "neutral",
}: {
  children: React.ReactNode;
  variant?: "success" | "warning" | "neutral";
}) {
  const styles = {
    success: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    warning: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    neutral: "border-slate-700 bg-slate-800/60 text-slate-300",
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs ${styles[variant]}`}
    >
      {children}
    </span>
  );
}

export default function App() {
  const [dashboard, setDashboard] =
    useState<RagDashboardResponse | null>(null);

  const [history, setHistory] = useState<RagHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [liveNotification, setLiveNotification] = useState<string | null>(null);

  const loadDashboard = useCallback(
    async (showRefreshIndicator = false) => {
      if (showRefreshIndicator) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setError(null);

      try {
        const [dashboardResult, historyResult] = await Promise.all([
          getRagDashboard(20),
          getRagHistory(20),
        ]);

        setDashboard(dashboardResult);
        setHistory(historyResult);
        setLastUpdated(new Date().toLocaleTimeString());
      } catch (err: unknown) {
        const message =
          err instanceof Error
            ? err.message
            : "Unable to load the RAG dashboard.";

        setError(message);
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [],
  );

  const handleSSEEvent = useCallback(
    (event: { type: string; [key: string]: any }) => {
      if (event.type === "rag_evaluated") {
        const preview = event.query_preview
          ? `"${event.query_preview}"`
          : "new run";
        setLiveNotification(
          `⚡ Live: New RAG evaluation saved (${preview}) — dashboard auto-refreshed`,
        );
        void loadDashboard(true);
        setTimeout(() => setLiveNotification(null), 5000);
      } else if (event.type === "drift_completed") {
        setLiveNotification(
          `⚡ Live: Drift experiment finished (${event.alert_count ?? 0} alerts) — dashboard auto-refreshed`,
        );
        void loadDashboard(true);
        setTimeout(() => setLiveNotification(null), 5000);
      }
    },
    [loadDashboard],
  );

  const { status: sseStatus, isConnected: sseConnected } = useSSE({
    onEvent: handleSSEEvent,
  });

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard]);

  const chartData = [...history]
    .reverse()
    .map((run, index) => ({
      run: `Run ${index + 1}`,
      date: formatDate(run.created_at),
      precision: getMetricValue(run, "precision_at_k"),
      recall: getMetricValue(run, "recall_at_k"),
      relevance: getMetricValue(run, "answer_relevance_score"),
    }));

  const metricCards = metricDefinitions.map((definition) => {
    const metric = dashboard?.metrics?.[definition.key];

    return {
      ...definition,
      average:
        typeof metric?.average === "number" &&
        Number.isFinite(metric.average)
          ? metric.average
          : null,
      availableRuns: metric?.available_runs ?? 0,
    };
  });

  return (
    <main className="min-h-screen bg-[#080d18] text-slate-200">
      <div className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6 lg:px-8 lg:py-10">
        {/* Header */}
        <header className="mb-8 overflow-hidden rounded-3xl border border-slate-800 bg-gradient-to-br from-[#111d31] via-[#0d1424] to-[#0b1020] p-6 sm:p-8">
          <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-center">
            <div>
              <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1.5 text-xs font-medium text-cyan-300">
                <Sparkles size={14} />
                AI Model Diagnostics Platform
              </div>

              <h1 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
                ModelAutopsy{" "}
                <span className="text-cyan-300">AI</span>
              </h1>

              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400 sm:text-base">
                Inspect model drift, investigate deep-learning behavior,
                monitor semantic changes, and evaluate retrieval-augmented
                generation quality from one dashboard.
              </p>

              <div className="mt-5 flex flex-wrap items-center gap-2">
                <StatusPill variant="success">
                  <Activity size={13} />
                  Diagnostics workspace
                </StatusPill>

                <StatusPill>
                  <Database size={13} />
                  RAG history
                </StatusPill>

                <StatusPill>
                  <ShieldCheck size={13} />
                  Model monitoring
                </StatusPill>

                <span
                  className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium transition ${
                    sseConnected
                      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                      : sseStatus === "connecting"
                      ? "border-amber-500/30 bg-amber-500/10 text-amber-300"
                      : "border-slate-700 bg-slate-800/60 text-slate-400"
                  }`}
                >
                  <span className="relative flex h-2 w-2">
                    {sseConnected && (
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
                    )}
                    <span
                      className={`relative inline-flex h-2 w-2 rounded-full ${
                        sseConnected
                          ? "bg-emerald-400"
                          : sseStatus === "connecting"
                          ? "bg-amber-400"
                          : "bg-slate-500"
                      }`}
                    ></span>
                  </span>
                  <Radio size={12} />
                  {sseConnected
                    ? "Live SSE Active"
                    : sseStatus === "connecting"
                    ? "Connecting stream..."
                    : "Stream Offline"}
                </span>
              </div>
            </div>

            <button
              type="button"
              onClick={() => void loadDashboard(true)}
              disabled={refreshing || loading}
              className="inline-flex items-center justify-center gap-2 self-start rounded-xl border border-slate-700 bg-slate-900/70 px-4 py-3 text-sm font-medium text-white transition hover:border-cyan-400/40 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 lg:self-center"
            >
              <RefreshCw
                size={16}
                className={refreshing ? "animate-spin" : ""}
              />
              Refresh dashboard
            </button>
          </div>

          {liveNotification && (
            <div className="mt-4 flex items-center justify-between rounded-xl border border-cyan-500/40 bg-cyan-950/60 px-4 py-2.5 text-xs font-medium text-cyan-200 shadow-lg shadow-cyan-950/40">
              <span className="flex items-center gap-2">
                <Sparkles size={14} className="text-cyan-400 animate-pulse" />
                {liveNotification}
              </span>
              <button
                type="button"
                onClick={() => setLiveNotification(null)}
                className="text-cyan-400 hover:text-white"
              >
                ✕
              </button>
            </div>
          )}

          {lastUpdated && (
            <p className="mt-4 text-xs text-slate-500">
              Last successful refresh: {lastUpdated}
            </p>
          )}
        </header>

        {/* System health */}
        <div className="mb-8">
          <SystemHealth />
        </div>

        {/* Module navigation */}
        <div className="mb-8">
          <DiagnosticsOverview />
        </div>

        {/* RAG dashboard */}
        <section id="rag-dashboard" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={ChartNoAxesCombined}
            title="RAG Evaluation Dashboard"
            description="Review retrieval quality, answer relevance, and context support across saved evaluation runs."
          />

          {error && (
            <div
              role="alert"
              className="mb-5 flex items-start gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4"
            >
              <AlertTriangle
                size={19}
                className="mt-0.5 shrink-0 text-rose-300"
              />

              <div className="min-w-0 flex-1">
                <p className="font-medium text-rose-200">
                  Could not load the RAG dashboard
                </p>
                <p className="mt-1 break-words text-sm text-rose-300/80">
                  {error}
                </p>
                <p className="mt-2 text-xs text-slate-400">
                  Check that FastAPI is running and the dashboard and history
                  endpoints are available.
                </p>
              </div>

              <button
                type="button"
                onClick={() => void loadDashboard(true)}
                className="shrink-0 rounded-lg border border-rose-500/30 px-3 py-1.5 text-xs text-rose-200 hover:bg-rose-500/10"
              >
                Retry
              </button>
            </div>
          )}

          {loading ? (
            <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-10 text-center">
              <RefreshCw
                size={25}
                className="mx-auto animate-spin text-cyan-300"
              />
              <p className="mt-4 text-sm text-slate-400">
                Loading RAG evaluation metrics...
              </p>
            </div>
          ) : (
            <>
              <div className="mb-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
                {metricCards.map((metric) => (
                  <MetricCard
                    key={metric.key}
                    label={metric.label}
                    value={metric.average}
                    description={metric.description}
                    availableRuns={metric.availableRuns}
                  />
                ))}
              </div>

              <div className="mb-5 grid gap-4 sm:grid-cols-3">
                <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5">
                  <p className="text-sm text-slate-400">
                    Evaluations included
                  </p>
                  <p className="mt-3 text-3xl font-semibold text-white">
                    {dashboard?.runs_included ?? history.length}
                  </p>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5">
                  <p className="text-sm text-slate-400">
                    Runs with findings
                  </p>
                  <p className="mt-3 text-3xl font-semibold text-amber-300">
                    {dashboard?.finding_summary?.runs_with_findings ?? "—"}
                  </p>
                </div>

                <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5">
                  <p className="text-sm text-slate-400">
                    Runs without findings
                  </p>
                  <p className="mt-3 text-3xl font-semibold text-emerald-300">
                    {dashboard?.finding_summary?.runs_without_findings ?? "—"}
                  </p>
                </div>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5 sm:p-6">
                <div className="mb-6">
                  <h3 className="font-semibold text-white">
                    Evaluation quality trends
                  </h3>
                  <p className="mt-1 text-sm text-slate-400">
                    Historical Precision@K, Recall@K, and answer relevance.
                    Missing metric values are left as gaps.
                  </p>
                </div>

                {chartData.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-slate-700 px-4 py-12 text-center">
                    <TrendingUp
                      size={27}
                      className="mx-auto text-slate-500"
                    />
                    <p className="mt-3 text-sm text-slate-300">
                      No saved RAG evaluation runs yet.
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      Complete an evaluation below to start building history.
                    </p>
                  </div>
                ) : (
                  <div className="h-[320px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={chartData}
                        margin={{
                          top: 10,
                          right: 12,
                          left: 0,
                          bottom: 5,
                        }}
                      >
                        <CartesianGrid
                          stroke="#263247"
                          strokeDasharray="3 3"
                        />

                        <XAxis
                          dataKey="run"
                          tick={{ fill: "#94a3b8", fontSize: 12 }}
                          axisLine={{ stroke: "#334155" }}
                          tickLine={false}
                        />

                        <YAxis
                          domain={[0, 1]}
                          tick={{ fill: "#94a3b8", fontSize: 12 }}
                          axisLine={false}
                          tickLine={false}
                          width={40}
                        />

                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#0b1220",
                            border: "1px solid #334155",
                            borderRadius: 12,
                            color: "#e2e8f0",
                          }}
                          labelStyle={{ color: "#cbd5e1" }}
                          formatter={(value) =>
                            typeof value === "number"
                              ? value.toFixed(3)
                              : "—"
                          }
                          labelFormatter={(label, payload) => {
                            const date = payload?.[0]?.payload?.date;
                            return `${label}${date ? ` · ${date}` : ""}`;
                          }}
                        />

                        <Line
                          type="monotone"
                          dataKey="precision"
                          name="Precision@K"
                          stroke="#22d3ee"
                          strokeWidth={2}
                          dot={{ r: 3 }}
                          connectNulls={false}
                        />

                        <Line
                          type="monotone"
                          dataKey="recall"
                          name="Recall@K"
                          stroke="#a78bfa"
                          strokeWidth={2}
                          dot={{ r: 3 }}
                          connectNulls={false}
                        />

                        <Line
                          type="monotone"
                          dataKey="relevance"
                          name="Answer relevance"
                          stroke="#34d399"
                          strokeWidth={2}
                          dot={{ r: 3 }}
                          connectNulls={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              {dashboard?.limitations &&
                dashboard.limitations.length > 0 && (
                  <div className="mt-5 rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
                    <div className="flex items-center gap-2 text-sm font-medium text-amber-200">
                      <CircleHelp size={16} />
                      Dashboard limitations
                    </div>

                    <ul className="mt-3 list-inside list-disc space-y-1 text-xs leading-5 text-slate-400">
                      {dashboard.limitations.map((limitation, index) => (
                        <li key={`${index}-${limitation}`}>
                          {limitation}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
            </>
          )}
        </section>

        {/* RAG history */}
        <section id="rag-history" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={Database}
            title="Recent RAG Evaluations"
            description="Browse stored evaluation runs and their headline metrics."
          />

          <div className="overflow-hidden rounded-2xl border border-slate-800 bg-[#0d1424]">
            {history.length === 0 ? (
              <div className="p-8 text-center">
                <Database
                  size={26}
                  className="mx-auto text-slate-500"
                />
                <p className="mt-3 text-sm text-slate-300">
                  No evaluation history available.
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  Run a RAG evaluation to save your first record.
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[800px] text-left text-sm">
                  <thead className="border-b border-slate-800 bg-slate-950/50 text-xs uppercase tracking-wider text-slate-500">
                    <tr>
                      <th className="px-5 py-4">Run ID</th>
                      <th className="px-5 py-4">Query</th>
                      <th className="px-5 py-4">Precision@K</th>
                      <th className="px-5 py-4">Recall@K</th>
                      <th className="px-5 py-4">Answer relevance</th>
                      <th className="px-5 py-4">Created</th>
                    </tr>
                  </thead>

                  <tbody>
                    {history.map((run) => (
                      <tr
                        key={run.run_id}
                        className="border-b border-slate-800/80 last:border-0 hover:bg-slate-900/40"
                      >
                        <td className="px-5 py-4 font-mono text-xs text-cyan-300">
                          {run.run_id.length > 14
                            ? `${run.run_id.slice(0, 14)}…`
                            : run.run_id}
                        </td>

                        <td className="max-w-[280px] px-5 py-4">
                          <p className="truncate text-slate-200">
                            {run.query}
                          </p>
                        </td>

                        <td className="px-5 py-4 text-slate-300">
                          {formatMetric(
                            getMetricValue(run, "precision_at_k"),
                          )}
                        </td>

                        <td className="px-5 py-4 text-slate-300">
                          {formatMetric(
                            getMetricValue(run, "recall_at_k"),
                          )}
                        </td>

                        <td className="px-5 py-4 text-slate-300">
                          {formatMetric(
                            getMetricValue(run, "answer_relevance_score"),
                          )}
                        </td>

                        <td className="px-5 py-4 text-xs text-slate-400">
                          {formatDate(run.created_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </section>

        {/* Synthetic Golden Dataset Generator */}
        <section id="golden-dataset" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={Sparkles}
            title="Synthetic Golden Dataset Generator"
            description="Auto-synthesize realistic RAG benchmark test cases from documentation with ground-truth attribution and adversarial tests."
          />

          <GoldenDatasetGenerator />
        </section>

        {/* RAG evaluation */}
        <section id="evaluate" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={FlaskConical}
            title="Evaluate a RAG Pipeline"
            description="Evaluate retrieval quality, answer relevance, and potential unsupported claims."
          />

          <RAGEvaluationForm
            onComplete={() => void loadDashboard(true)}
          />
        </section>

        {/* Prompt guardrails */}
        <section id="guardrails" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={Shield}
            title="Prompt Guardrails & Security Scanner"
            description="Scan prompts for jailbreak exploits, injection attempts, and PII leakage before LLM ingestion."
          />

          <PromptGuardrails />
        </section>

        {/* RAG comparison */}
        <section id="rag-comparison" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={TrendingUp}
            title="RAG Regression Comparison"
            description="Compare two saved evaluation runs to identify metric regressions and improvements."
          />

          <RAGComparison history={history} />
        </section>

        {/* Deep learning */}
        <section id="deep-learning" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={Brain}
            title="Deep Learning Lab"
            description="Diagnose a neural network, inspect its training behavior, and compare model configurations."
          />

          <div className="space-y-6">
            <DeepLearningDiagnosis />
            <DeepLearningComparison />
          </div>
        </section>

        {/* Semantic monitoring */}
        <section
          id="semantic-monitoring"
          className="mb-10 scroll-mt-6"
        >
          <SectionHeading
            icon={Sparkles}
            title="Semantic Intelligence"
            description="Analyze semantic changes between reference and current text collections."
          />

          <SemanticMonitoring />
        </section>

        {/* Drift alerts */}
        <section id="drift-alerts" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={AlertTriangle}
            title="Drift Alerts"
            description="Review recorded alerts and inspect their details."
          />

          <DriftAlerts />
        </section>

        {/* Webhook Alerts */}
        <section id="webhooks" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={Bell}
            title="Automated Alert Webhooks"
            description="Configure Slack, Discord, and HTTP endpoints to automatically dispatch alerts when drift or regressions are detected."
          />

          <WebhookAlerts />
        </section>

        {/* Tabular experiments */}
        <section id="tabular-experiments" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={Activity}
            title="Tabular Drift Experiments"
            description="Run the tabular experiment and inspect the returned metrics and feature-level results."
          />

          <TabularExperiment />
        </section>

        {/* CSV Dataset Drift Upload */}
        <section id="csv-drift" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={Activity}
            title="Upload Your Own Dataset for Drift"
            description="Drag and drop baseline and current CSV files to run KS-test drift detection on your real production data."
          />

          <CSVDriftUpload />
        </section>

        {/* Drift & Adversarial Perturbation Simulator */}
        <section id="drift-simulator" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={Zap}
            title="Drift & Adversarial Perturbation Simulator"
            description="Inject mean shifts, variance expansion, Gaussian noise, outlier contamination, and label flips to stress-test your models live."
          />

          <DriftSimulator />
        </section>

        {/* Export report */}
        <section id="export-report" className="mb-10 scroll-mt-6">
          <SectionHeading
            icon={CircleHelp}
            title="Export Diagnostic Report"
            description="Generate and download a full audit report covering RAG metrics, evaluation history, and drift alerts."
          />

          <ExportReport />
        </section>

        {/* Footer */}
        <footer className="border-t border-slate-800 py-6">
          <div className="flex flex-col gap-3 text-xs text-slate-500 sm:flex-row sm:items-center sm:justify-between">
            <p>
              ModelAutopsy AI · Model diagnostics and evaluation workspace
            </p>

            <div className="flex items-center gap-2">
              <CheckCircle2 size={14} />
              <span>
                Metrics depend on available evaluation data and backend results.
              </span>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
}
