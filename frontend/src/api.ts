import axios from "axios";
import type {
  RagDashboardResponse,
  RagHistoryItem,
  RagComparisonResponse,
  RegisteredModel,
  ModelComparisonResult,
  DriftExplanation,
  GoldenDatasetRecord,
  LLMTrace,
  LLMTraceStats,
  LLMProviderInfo,
  HardenedPromptResult,
  RetrainingEvaluation,
  RetrainingRun,
  PanelConsensus,
  VectorDocumentResult,
  VectorStoreStatus,
} from "./types";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
  timeout: 30000,
});

// -------------------------------------------------------------------
// RAG Telemetry
// -------------------------------------------------------------------
export async function getRagDashboard(
  limit = 20,
): Promise<RagDashboardResponse> {
  const response = await api.get<RagDashboardResponse>(
    "/api/v1/rag/dashboard",
    { params: { limit } },
  );
  return response.data;
}

export async function getRagHistory(
  limit = 20,
): Promise<RagHistoryItem[]> {
  const response = await api.get<{ evaluations: RagHistoryItem[] }>(
    "/api/v1/rag/history",
    { params: { limit } },
  );
  return response.data.evaluations;
}

export async function compareRagRuns(
  baselineRunId: string,
  candidateRunId: string,
  regressionThreshold = 0.1,
): Promise<RagComparisonResponse> {
  const response = await api.post<RagComparisonResponse>(
    "/api/v1/rag/compare",
    {
      baseline_run_id: baselineRunId,
      candidate_run_id: candidateRunId,
      regression_threshold: regressionThreshold,
    },
  );
  return response.data;
}

// -------------------------------------------------------------------
// Model Registry
// -------------------------------------------------------------------
export async function listRegisteredModels(params?: {
  name?: string;
  stage?: string;
  limit?: number;
}): Promise<RegisteredModel[]> {
  const response = await api.get<{ models: RegisteredModel[] }>(
    "/api/v1/models/registry",
    { params },
  );
  return response.data.models;
}

export async function registerModel(payload: {
  name: string;
  version: string;
  stage?: string;
  framework?: string;
  description?: string;
  artifact_uri?: string;
  metrics?: Record<string, any>;
  parameters?: Record<string, any>;
  tags?: string[];
}): Promise<RegisteredModel> {
  const response = await api.post<RegisteredModel>(
    "/api/v1/models/registry",
    payload,
  );
  return response.data;
}

export async function updateModelStage(
  modelId: string,
  stage: string,
): Promise<RegisteredModel> {
  const response = await api.patch<RegisteredModel>(
    `/api/v1/models/registry/${modelId}/stage`,
    { stage },
  );
  return response.data;
}

export async function deleteRegisteredModel(modelId: string): Promise<boolean> {
  const response = await api.delete<{ deleted: boolean }>(
    `/api/v1/models/registry/${modelId}`,
  );
  return response.data.deleted;
}

export async function compareRegisteredModels(
  baselineId: string,
  candidateId: string,
): Promise<ModelComparisonResult> {
  const response = await api.post<ModelComparisonResult>(
    "/api/v1/models/registry/compare",
    { baseline_id: baselineId, candidate_id: candidateId },
  );
  return response.data;
}

// -------------------------------------------------------------------
// AI Diagnostic Explanations
// -------------------------------------------------------------------
export async function explainDrift(
  driftSummary: Record<string, any>,
  featureDrifts: Record<string, any>[],
  apiKey?: string,
): Promise<DriftExplanation> {
  const response = await api.post<DriftExplanation>(
    "/api/v1/explain/drift",
    {
      drift_summary: driftSummary,
      feature_drifts: featureDrifts,
      api_key: apiKey,
    },
  );
  return response.data;
}

// -------------------------------------------------------------------
// Golden Benchmarks
// -------------------------------------------------------------------
export async function saveGoldenDataset(payload: {
  name: string;
  description?: string;
  cases: any[];
}): Promise<GoldenDatasetRecord> {
  const response = await api.post<GoldenDatasetRecord>(
    "/api/v1/rag/datasets/save",
    payload,
  );
  return response.data;
}

export async function listGoldenDatasets(
  limit = 50,
): Promise<GoldenDatasetRecord[]> {
  const response = await api.get<{ datasets: GoldenDatasetRecord[] }>(
    "/api/v1/rag/datasets",
    { params: { limit } },
  );
  return response.data.datasets;
}

export async function getGoldenDataset(
  datasetId: string,
): Promise<GoldenDatasetRecord> {
  const response = await api.get<GoldenDatasetRecord>(
    `/api/v1/rag/datasets/${datasetId}`,
  );
  return response.data;
}

export async function deleteGoldenDataset(
  datasetId: string,
): Promise<boolean> {
  const response = await api.delete<{ deleted: boolean }>(
    `/api/v1/rag/datasets/${datasetId}`,
  );
  return response.data.deleted;
}

// -------------------------------------------------------------
// LLM Telemetry & Tracing
// -------------------------------------------------------------
export async function getLLMProviders(): Promise<{
  providers: LLMProviderInfo[];
  recommended_provider: string;
}> {
  const response = await api.get("/api/v1/llm/providers");
  return response.data;
}

export async function getLLMTraces(limit = 50): Promise<LLMTrace[]> {
  const response = await api.get<{ traces: LLMTrace[] }>("/api/v1/llm/traces", {
    params: { limit },
  });
  return response.data.traces;
}

export async function getLLMTraceStats(): Promise<LLMTraceStats> {
  const response = await api.get<LLMTraceStats>("/api/v1/llm/traces/stats");
  return response.data;
}

export async function generateWithLLM(payload: {
  prompt: string;
  system_prompt?: string;
  provider?: string;
  model?: string;
  api_key?: string;
}): Promise<{
  text: string;
  model: string;
  provider: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}> {
  const response = await api.post("/api/v1/llm/generate", payload);
  return response.data;
}

// -------------------------------------------------------------
// Autonomous Remediation & Healing Agents
// -------------------------------------------------------------
export async function hardenPrompt(payload: {
  original_prompt: string;
  attack_vector?: string;
  exploit_sample?: string;
  provider?: string;
  model?: string;
}): Promise<HardenedPromptResult> {
  const response = await api.post<HardenedPromptResult>(
    "/api/v1/agents/harden-prompt",
    payload,
  );
  return response.data;
}

export async function checkRetraining(payload: {
  psi?: number;
  ks_p_value?: number;
  accuracy_drop_pct?: number;
}): Promise<RetrainingEvaluation> {
  const response = await api.post<RetrainingEvaluation>(
    "/api/v1/agents/check-retraining",
    payload,
  );
  return response.data;
}

export async function triggerRetraining(payload: {
  reason: string;
  metrics?: Record<string, unknown>;
}): Promise<{
  run_id: string;
  status: string;
  triggered_at: string;
  reason: string;
  candidate_model_id: string;
  message: string;
}> {
  const response = await api.post("/api/v1/agents/trigger-retraining", payload);
  return response.data;
}

export async function getRetrainingHistory(limit = 50): Promise<RetrainingRun[]> {
  const response = await api.get<{ runs: RetrainingRun[] }>(
    "/api/v1/agents/retraining-history",
    { params: { limit } },
  );
  return response.data.runs;
}

// -------------------------------------------------------------
// LLM-as-a-Judge 3-Panel Jury
// -------------------------------------------------------------
export async function runJudgePanel(payload: {
  query: string;
  answer: string;
  context?: string;
  provider?: string;
  model?: string;
}): Promise<PanelConsensus> {
  const response = await api.post<PanelConsensus>(
    "/api/v1/rag/judge-panel",
    payload,
  );
  return response.data;
}

// -------------------------------------------------------------
// Dense Vector Store & Neural Reranker
// -------------------------------------------------------------
export async function getVectorStoreStatus(): Promise<VectorStoreStatus> {
  const response = await api.get<VectorStoreStatus>(
    "/api/v1/rag/vectorstore/status",
  );
  return response.data;
}

export async function indexVectorDocuments(payload: {
  texts: string[];
  metadatas?: Array<Record<string, unknown>>;
}): Promise<{ indexed_count: number; doc_ids: string[] }> {
  const response = await api.post("/api/v1/rag/vectorstore/index", payload);
  return response.data;
}

export async function searchVectorStore(payload: {
  query: string;
  top_k?: number;
  rerank?: boolean;
}): Promise<{ results: VectorDocumentResult[]; total_found: number }> {
  const response = await api.post("/api/v1/rag/vectorstore/search", payload);
  return response.data;
}

export async function clearVectorStore(): Promise<boolean> {
  const response = await api.post<{ cleared: boolean }>(
    "/api/v1/rag/vectorstore/clear",
  );
  return response.data.cleared;
}
