export interface RagMetric {
  available_runs: number;
  average: number | null;
  latest_value: number | null;
}

export interface RagEvaluationSummary {
  precision_at_k?: number;
  recall_at_k?: number;
  reciprocal_rank?: number;
  ndcg_at_k?: number;
  answer_relevance_score?: number;
  context_support_coverage?: number;
  finding_count?: number;
  potentially_unsupported_sentence_count?: number;
}

export interface RagHistoryItem {
  run_id: string;
  created_at: string;
  query: string;
  summary: RagEvaluationSummary;
}

export interface RagDashboardResponse {
  service: string;
  runs_included: number;
  finding_summary: {
    runs_with_findings: number;
    runs_without_findings: number;
    runs_with_missing_finding_data: number;
  };
  metrics: Record<string, RagMetric>;
  recent_runs: RagHistoryItem[];
  limitations: string[];
}

export interface RagComparisonResponse {
  baseline_run_id: string;
  candidate_run_id: string;
  regression_threshold: number;
  metrics: Record<
    string,
    {
      baseline: number | null;
      candidate: number | null;
      change: number | null;
      relative_change: number | null;
      status: string;
    }
  >;
  regressions: string[];
  improvements: string[];
  regression_detected: boolean;
  limitations: string[];
}

// -------------------------------------------------------------
// Model Registry Types
// -------------------------------------------------------------
export type ModelStage = "development" | "staging" | "production" | "archived";

export interface RegisteredModel {
  model_id: string;
  name: string;
  version: string;
  stage: ModelStage;
  framework: string;
  description: string;
  artifact_uri: string;
  metrics: Record<string, number | string>;
  parameters: Record<string, any>;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface ModelComparisonResult {
  baseline: {
    model_id: string;
    name: string;
    version: string;
    stage: string;
  };
  candidate: {
    model_id: string;
    name: string;
    version: string;
    stage: string;
  };
  metric_deltas: Record<
    string,
    {
      baseline: number | null;
      candidate: number | null;
      delta: number | null;
      percent_change: number | null;
      improved: boolean | null;
    }
  >;
  summary: string;
}

// -------------------------------------------------------------
// Drift Explanation Types
// -------------------------------------------------------------
export interface DriftExplanation {
  engine: string;
  health_status: "HEALTHY" | "MONITOR" | "WARNING" | "CRITICAL" | "ANALYZED";
  headline: string;
  summary: string;
  key_findings?: string[];
  recommended_actions: string[];
}

// -------------------------------------------------------------
// Golden Dataset Types
// -------------------------------------------------------------
export interface GoldenTestCase {
  query: string;
  expected_context: string;
  ground_truth_answer: string;
  topic?: string;
  is_adversarial?: boolean;
}

export interface GoldenDatasetRecord {
  dataset_id: string;
  name: string;
  domain: string;
  test_case_count: number;
  tags: string[];
  created_at: string;
  test_cases?: GoldenTestCase[];
}

// -------------------------------------------------------------
// LLM Tracing & Telemetry Types
// -------------------------------------------------------------
export interface LLMSpan {
  name: string;
  start_ms: number;
  end_ms: number;
  duration_ms?: number;
  metadata?: Record<string, unknown>;
}

export interface LLMTrace {
  trace_id: string;
  request_id: string;
  provider: string;
  model: string;
  prompt: string;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  latency_ms: number;
  estimated_cost_usd: number;
  spans: LLMSpan[];
  created_at: string;
  success: boolean;
  error?: string;
}

export interface LLMTraceStats {
  total_calls: number;
  total_tokens: number;
  total_cost_usd: number;
  avg_latency_ms: number;
  min_latency_ms: number;
  max_latency_ms: number;
  success_rate: number;
  by_provider: Array<{
    provider: string;
    calls: number;
    total_cost_usd: number;
  }>;
}

export interface LLMProviderInfo {
  id: string;
  name: string;
  type: string;
  available: boolean;
  url?: string;
  description: string;
  models: string[];
}

// -------------------------------------------------------------
// Autonomous Agent & Self-Healing Types
// -------------------------------------------------------------
export interface HardenedPromptResult {
  original_prompt: string;
  attack_vector: string;
  hardened_prompt: string;
  confidence_score: number;
  techniques_applied: string[];
  diff_summary: string;
  provider_used: string;
}

export interface RetrainingEvaluation {
  retraining_recommended: boolean;
  urgency: string;
  psi: number;
  ks_p_value: number;
  reasons: string[];
  action_recommended: string;
}

export interface RetrainingRun {
  run_id: string;
  triggered_at: string;
  reason: string;
  status: string;
  candidate_model_id: string;
  metrics_snapshot?: Record<string, unknown>;
  notes?: string;
}

// -------------------------------------------------------------
// LLM-as-a-Judge 3-Panel Jury Types
// -------------------------------------------------------------
export interface JudgeVerdict {
  persona: string;
  verdict: "PASS" | "FAIL" | "WARNING";
  score: number;
  rationale: string;
  flags: string[];
}

export interface PanelConsensus {
  consensus: "APPROVED" | "REJECTED" | "REQUIRES_REVIEW";
  overall_score: number;
  pass_count: number;
  fail_count: number;
  warning_count: number;
  verdicts: JudgeVerdict[];
  recommendation: string;
}

// -------------------------------------------------------------
// Vector Store & Semantic Retrieval Types
// -------------------------------------------------------------
export interface VectorDocumentResult {
  doc_id: string;
  text: string;
  metadata: Record<string, unknown>;
  similarity_score: number;
  rerank_score?: number;
  created_at: string;
}

export interface VectorStoreStatus {
  initialized: boolean;
  document_count: number;
  embedding_dimension: number;
  backend: string;
}
