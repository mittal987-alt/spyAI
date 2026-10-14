
import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock3,
  RefreshCw,
  Server,
} from "lucide-react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

type HealthState = "checking" | "online" | "offline";

interface HealthResponse {
  status?: string;
  service?: string;
  version?: string;
  [key: string]: unknown;
}

export default function SystemHealth() {
  const [health, setHealth] = useState<HealthState>("checking");
  const [details, setDetails] = useState<HealthResponse | null>(null);
  const [error, setError] = useState("");
  const [checkedAt, setCheckedAt] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  const checkHealth = useCallback(async () => {
    setChecking(true);
    setHealth("checking");
    setError("");

    try {
      const response = await axios.get<HealthResponse>(
        `${API_BASE_URL}/health`,
        { timeout: 10000 },
      );

      setDetails(response.data);
      setHealth("online");
      setCheckedAt(new Date().toLocaleTimeString());
    } catch (err: unknown) {
      setHealth("offline");
      setDetails(null);
      setCheckedAt(new Date().toLocaleTimeString());

      setError(
        axios.isAxiosError(err)
          ? err.response
            ? `Health endpoint returned HTTP ${err.response.status}.`
            : "Cannot reach the backend. Check whether FastAPI is running."
          : "An unexpected error occurred while checking system health.",
      );
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    void checkHealth();
  }, [checkHealth]);

  const statusStyles = {
    checking: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    online: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
    offline: "border-rose-500/30 bg-rose-500/10 text-rose-300",
  };

  const StatusIcon =
    health === "online"
      ? CheckCircle2
      : health === "offline"
        ? AlertCircle
        : Clock3;

  return (
    <section
      id="system-health"
      className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5 sm:p-6"
    >
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="rounded-xl bg-cyan-400/10 p-3 text-cyan-300">
            <Server size={23} />
          </div>

          <div>
            <h2 className="text-lg font-semibold text-white">
              System Health
            </h2>
            <p className="mt-1 text-sm text-slate-400">
              Backend connectivity and service status
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={() => void checkHealth()}
          disabled={checking}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 transition hover:border-cyan-400/50 hover:text-white disabled:opacity-50"
        >
          <RefreshCw
            size={15}
            className={checking ? "animate-spin" : ""}
          />
          Refresh
        </button>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <Activity size={16} />
            Backend status
          </div>

          <div className="mt-3">
            <span
              className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm ${statusStyles[health]}`}
            >
              <StatusIcon size={15} />
              {health === "online"
                ? "Online"
                : health === "offline"
                  ? "Offline"
                  : "Checking"}
            </span>
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-4">
          <p className="text-sm text-slate-400">Last checked</p>
          <p className="mt-3 text-sm font-medium text-white">
            {checkedAt ?? "Not checked yet"}
          </p>
        </div>
      </div>

      {health === "online" && details && (
        <div className="mt-4 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
          <p className="text-sm font-medium text-emerald-300">
            Backend responded successfully.
          </p>

          <pre className="mt-3 max-h-48 overflow-auto whitespace-pre-wrap break-words text-xs leading-5 text-slate-400">
            {JSON.stringify(details, null, 2)}
          </pre>
        </div>
      )}

      {health === "offline" && (
        <div
          role="alert"
          className="mt-4 rounded-xl border border-rose-500/20 bg-rose-500/5 p-4"
        >
          <p className="text-sm text-rose-300">{error}</p>
          <p className="mt-2 text-xs leading-5 text-slate-500">
            Check the FastAPI terminal and confirm VITE_API_BASE_URL in your
            frontend environment file.
          </p>
        </div>
      )}
    </section>
  );
}
