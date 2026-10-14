
import axios from "axios";
import type {
  RagDashboardResponse,
  RagHistoryItem,
  RagComparisonResponse,
} from "./types";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
  timeout: 30000,
});

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
