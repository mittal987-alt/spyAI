
import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import {
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  ShieldAlert,
} from "lucide-react";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

interface DriftAlert {
  id?: string | number;
  alert_id?: string | number;
  severity?: string;
  status?: string;
  message?: string;
  title?: string;
  feature_name?: string;
  created_at?: string;
  timestamp?: string;
  details?: unknown;
  [key: string]: unknown;
}

interface AlertSummary {
  total_alerts?: number;
  critical?: number;
  high?: number;
  medium?: number;
  low?: number;
  by_severity?: Record<string, number>;
  [key: string]: unknown;
}

function severityClass(severity?: string) {
  switch (severity?.toLowerCase()) {
    case "critical":
      return "border-rose-500/30 bg-rose-500/10 text-rose-300";
    case "high":
      return "border-orange-500/30 bg-orange-500/10 text-orange-300";
    case "medium":
      return "border-amber-500/30 bg-amber-500/10 text-amber-300";
    case "low":
      return "border-cyan-500/30 bg-cyan-500/10 text-cyan-300";
    default:
      return "border-slate-700 bg-slate-800/50 text-slate-300";
  }
}

export default function DriftAlerts() {
  const [alerts, setAlerts] = useState<DriftAlert[]>([]);
  const [summary, setSummary] = useState<AlertSummary | null>(null);
  const [selected, setSelected] = useState<DriftAlert | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadAlerts = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const [summaryResponse, alertsResponse] = await Promise.all([
        axios.get(`${API_BASE}/api/v1/alerts/summary`),
        axios.get(`${API_BASE}/api/v1/alerts`, {
          params: { limit: 100 },
        }),
      ]);

      setSummary(summaryResponse.data);

      const payload = alertsResponse.data;
      const items = Array.isArray(payload)
        ? payload
        : Array.isArray(payload?.alerts)
          ? payload.alerts
          : Array.isArray(payload?.items)
            ? payload.items
            : [];

      setAlerts(items);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail;
        setError(
          typeof detail === "string"
            ? detail
            : `Could not load alerts (${err.response?.status ?? "network error"}).`,
        );
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadAlerts();
  }, [loadAlerts]);

  async function openAlert(alert: DriftAlert) {
    const id = alert.alert_id ?? alert.id;

    if (id == null) {
      setSelected(alert);
      return;
    }

    try {
      const response = await axios.get(
        `${API_BASE}/api/v1/alerts/${encodeURIComponent(String(id))}`,
      );
      setSelected(response.data?.alert ?? response.data);
    } catch {
      setSelected(alert);
    }
  }

  const severityCounts = summary?.by_severity;

  const countFor = (severity: string) => {
    const direct = summary?.[severity] ?? severityCounts?.[severity];

    if (typeof direct === "number") return direct;

    return alerts.filter(
      (alert) => alert.severity?.toLowerCase() === severity,
    ).length;
  };

  return (
    <section className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5 sm:p-6">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-amber-400/10 p-3 text-amber-300">
            <ShieldAlert size={22} />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-white">
              Drift Alerts
            </h2>
            <p className="mt-1 text-sm text-slate-400">
              Inspect persisted monitoring alerts and their details.
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={() => void loadAlerts()}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm hover:border-cyan-400/40 disabled:opacity-50"
        >
          <RefreshCw
            size={15}
            className={loading ? "animate-spin" : ""}
          />
          Refresh alerts
        </button>
      </div>

      {error && (
        <div className="mb-5 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">
          {error}
        </div>
      )}

      <div className="mb-6 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {(["critical", "high", "medium", "low"] as const).map(
          (severity) => (
            <div
              key={severity}
              className={`rounded-xl border p-4 ${severityClass(severity)}`}
            >
              <p className="text-xs uppercase tracking-wider">
                {severity}
              </p>
              <p className="mt-2 text-2xl font-semibold">
                {countFor(severity)}
              </p>
            </div>
          ),
        )}
      </div>

      {loading && alerts.length === 0 ? (
        <p className="py-8 text-center text-sm text-slate-500">
          Loading alerts...
        </p>
      ) : alerts.length === 0 ? (
        <div className="rounded-xl border border-slate-800 p-8 text-center">
          <CheckCircle2
            size={28}
            className="mx-auto mb-3 text-emerald-300"
          />
          <p className="font-medium text-white">No alerts returned</p>
          <p className="mt-2 text-sm text-slate-500">
            This means the API returned no alert records for this request;
            it does not prove that every model is healthy.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {alerts.map((alert, index) => {
            const id = alert.alert_id ?? alert.id ?? index;

            return (
              <button
                type="button"
                key={String(id)}
                onClick={() => void openAlert(alert)}
                className="flex w-full flex-col gap-3 rounded-xl border border-slate-800 bg-slate-950/50 p-4 text-left transition hover:border-slate-600 sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="flex items-start gap-3">
                  <AlertTriangle
                    size={18}
                    className="mt-1 shrink-0 text-amber-300"
                  />

                  <div>
                    <p className="font-medium text-white">
                      {alert.title ??
                        alert.message ??
                        alert.feature_name ??
                        `Alert ${String(id)}`}
                    </p>

                    <p className="mt-1 text-xs text-slate-500">
                      {alert.created_at ??
                        alert.timestamp ??
                        "Timestamp unavailable"}
                    </p>
                  </div>
                </div>

                <span
                  className={`w-fit rounded-full border px-3 py-1 text-xs capitalize ${severityClass(alert.severity)}`}
                >
                  {alert.severity ?? "unknown severity"}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {selected && (
        <div className="mt-6 rounded-xl border border-cyan-400/20 bg-slate-950 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="font-semibold text-white">Alert details</h3>
            <button
              type="button"
              onClick={() => setSelected(null)}
              className="text-sm text-slate-400 hover:text-white"
            >
              Close
            </button>
          </div>

          <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words text-xs leading-6 text-cyan-100">
            {JSON.stringify(selected, null, 2)}
          </pre>
        </div>
      )}
    </section>
  );
}
