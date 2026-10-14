
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
