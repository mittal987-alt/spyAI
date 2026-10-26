import { useEffect, useState, useCallback } from "react";
import axios from "axios";
import {
  Bell,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Globe,
  History,
  Loader2,
  MessageSquare,
  Plus,
  Radio,
  RefreshCw,
  Send,
  Trash2,
  XCircle,
} from "lucide-react";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

interface WebhookItem {
  webhook_id: string;
  name: string;
  url: string;
  service_type: "slack" | "discord" | "generic";
  min_severity: "low" | "medium" | "high";
  is_enabled: boolean;
  created_at: string;
}

interface WebhookLog {
  log_id: string;
  webhook_id: string;
  webhook_name: string;
  event_type: string;
  severity: string;
  status_code: number;
  success: boolean;
  message: string;
  created_at: string;
}

const PRESET_TEMPLATES = [
  {
    name: "Slack #model-alerts",
    service: "slack" as const,
    minSeverity: "medium" as const,
    placeholder: "https://hooks.slack.com/services/T000/B000/XXXX",
  },
  {
    name: "Discord #drift-channel",
    service: "discord" as const,
    minSeverity: "high" as const,
    placeholder: "https://discord.com/api/webhooks/123456789/XXXX",
  },
  {
    name: "PagerDuty / Generic Webhook",
    service: "generic" as const,
    minSeverity: "high" as const,
    placeholder: "https://events.pagerduty.com/v2/enqueue",
  },
];

export default function WebhookAlerts() {
  const [webhooks, setWebhooks] = useState<WebhookItem[]>([]);
  const [logs, setLogs] = useState<WebhookLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [showLogs, setShowLogs] = useState(false);

  // Form state
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [serviceType, setServiceType] = useState<"slack" | "discord" | "generic">("slack");
  const [minSeverity, setMinSeverity] = useState<"low" | "medium" | "high">("medium");
  const [submitting, setSubmitting] = useState(false);

  // Action status state
  const [testingId, setTestingId] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    status_code: number;
    message: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchWebhooks = useCallback(async () => {
    try {
      const [resHooks, resLogs] = await Promise.all([
        axios.get<{ webhooks: WebhookItem[] }>(`${API_BASE}/api/v1/webhooks`),
        axios.get<{ logs: WebhookLog[] }>(`${API_BASE}/api/v1/webhooks/logs?limit=15`),
      ]);
      setWebhooks(resHooks.data.webhooks);
      setLogs(resLogs.data.logs);
      setError(null);
    } catch (err: unknown) {
      console.warn("Could not load webhooks:", err);
      setError("Unable to connect to webhooks service.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchWebhooks();
  }, [fetchWebhooks]);

  const handleAddWebhook = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !url.trim()) return;

    setSubmitting(true);
    setError(null);

    try {
      await axios.post(`${API_BASE}/api/v1/webhooks`, {
        name: name.trim(),
        url: url.trim(),
        service_type: serviceType,
        min_severity: minSeverity,
      });

      setName("");
      setUrl("");
      await fetchWebhooks();
    } catch (err: unknown) {
      const msg =
        axios.isAxiosError(err) && err.response?.data?.detail
          ? err.response.data.detail
          : "Failed to create webhook.";
      setError(String(msg));
    } finally {
      setSubmitting(false);
    }
  };

  const handleToggle = async (webhookId: string, currentStatus: boolean) => {
    try {
      await axios.patch(`${API_BASE}/api/v1/webhooks/${webhookId}`, {
        is_enabled: !currentStatus,
      });
      setWebhooks((prev) =>
        prev.map((w) =>
          w.webhook_id === webhookId ? { ...w, is_enabled: !currentStatus } : w,
        ),
      );
    } catch (err: unknown) {
      console.warn("Failed to toggle webhook:", err);
    }
  };

  const handleDelete = async (webhookId: string) => {
    try {
      await axios.delete(`${API_BASE}/api/v1/webhooks/${webhookId}`);
      setWebhooks((prev) => prev.filter((w) => w.webhook_id !== webhookId));
    } catch (err: unknown) {
      console.warn("Failed to delete webhook:", err);
    }
  };

  const handleTestWebhook = async (item: WebhookItem) => {
    setTestingId(item.webhook_id);
    setTestResult(null);

    try {
      const res = await axios.post<{
        success: boolean;
        status_code: number;
        message: string;
      }>(`${API_BASE}/api/v1/webhooks/test`, {
        url: item.url,
        service_type: item.service_type,
        name: item.name,
      });
      setTestResult(res.data);
      void fetchWebhooks();
    } catch (err: unknown) {
      const msg =
        axios.isAxiosError(err) && err.response?.data?.detail
          ? err.response.data.detail
          : "Failed to test webhook connection.";
      setTestResult({
        success: false,
        status_code: 0,
        message: String(msg),
      });
    } finally {
      setTestingId(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Configuration Card */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-6 shadow-xl">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="flex items-center gap-2 text-base font-semibold text-white">
              <Bell className="text-cyan-400" size={18} />
              Automated Alert Webhooks
            </h3>
            <p className="mt-1 text-xs text-slate-400">
              Deliver real-time notifications to Slack, Discord, or internal HTTP endpoints when feature drift or model regressions are detected.
            </p>
          </div>

          {/* Preset Buttons */}
          <div className="flex flex-wrap gap-2">
            {PRESET_TEMPLATES.map((tmpl, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => {
                  setName(tmpl.name);
                  setServiceType(tmpl.service);
                  setMinSeverity(tmpl.minSeverity);
                  setUrl(tmpl.placeholder);
                }}
                className="rounded-lg border border-slate-700 bg-slate-900/60 px-2.5 py-1.5 text-xs text-slate-300 transition hover:border-cyan-400/50 hover:bg-slate-800 hover:text-white"
              >
                + {tmpl.service.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Add Webhook Form */}
        <form onSubmit={(e) => void handleAddWebhook(e)} className="mt-5 space-y-4">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Webhook Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. #prod-alerts-channel"
                required
                className="w-full rounded-xl border border-slate-700 bg-slate-950/70 px-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:border-cyan-400 focus:outline-none"
              />
            </div>

            <div className="sm:col-span-2 lg:col-span-1">
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Target Service
              </label>
              <select
                value={serviceType}
                onChange={(e) =>
                  setServiceType(e.target.value as "slack" | "discord" | "generic")
                }
                className="w-full rounded-xl border border-slate-700 bg-slate-950/70 px-3 py-2 text-xs text-slate-200 focus:border-cyan-400 focus:outline-none"
              >
                <option value="slack">Slack Incoming Webhook</option>
                <option value="discord">Discord Webhook Embed</option>
                <option value="generic">Generic JSON HTTP POST</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Minimum Severity
              </label>
              <select
                value={minSeverity}
                onChange={(e) =>
                  setMinSeverity(e.target.value as "low" | "medium" | "high")
                }
                className="w-full rounded-xl border border-slate-700 bg-slate-950/70 px-3 py-2 text-xs text-slate-200 focus:border-cyan-400 focus:outline-none"
              >
                <option value="low">Low (All Drift & Regressions)</option>
                <option value="medium">Medium (Moderate & High Alerts)</option>
                <option value="high">High Only (Critical Drift / Outages)</option>
              </select>
            </div>

            <div className="flex items-end">
              <button
                type="submit"
                disabled={submitting || !name || !url}
                className="inline-flex w-full items-center justify-center gap-1.5 rounded-xl bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:opacity-50"
              >
                {submitting ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Plus size={14} />
                )}
                Register Endpoint
              </button>
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">
              Endpoint URL (HTTPS)
            </label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://hooks.slack.com/services/... or https://discord.com/api/webhooks/..."
              required
              className="w-full rounded-xl border border-slate-700 bg-slate-950/70 px-3 py-2 font-mono text-xs text-slate-200 placeholder-slate-500 focus:border-cyan-400 focus:outline-none"
            />
          </div>
        </form>

        {error && (
          <div className="mt-4 flex items-center gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
            <XCircle size={15} />
            <span>{error}</span>
          </div>
        )}

        {testResult && (
          <div
            className={`mt-4 flex items-center justify-between rounded-xl border p-3 text-xs ${
              testResult.success
                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                : "border-rose-500/30 bg-rose-500/10 text-rose-300"
            }`}
          >
            <span className="flex items-center gap-2">
              {testResult.success ? (
                <CheckCircle2 size={15} className="text-emerald-400" />
              ) : (
                <XCircle size={15} className="text-rose-400" />
              )}
              <span>
                {testResult.success
                  ? `Test notification delivered successfully (HTTP ${testResult.status_code || 200})`
                  : `Test notification failed: ${testResult.message}`}
              </span>
            </span>
            <button
              type="button"
              onClick={() => setTestResult(null)}
              className="text-slate-400 hover:text-white"
            >
              ✕
            </button>
          </div>
        )}
      </div>

      {/* Webhooks Table */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-6 shadow-xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Radio size={16} className="text-emerald-400" />
            <h4 className="text-sm font-semibold text-white">
              Active Endpoints ({webhooks.length})
            </h4>
          </div>

          <button
            type="button"
            onClick={() => void fetchWebhooks()}
            disabled={loading}
            className="inline-flex items-center gap-1 text-xs text-slate-400 hover:text-cyan-300"
          >
            <RefreshCw size={12} className={loading ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>

        {webhooks.length === 0 ? (
          <div className="mt-4 rounded-xl border border-slate-800/80 bg-slate-900/40 p-6 text-center text-xs text-slate-500">
            No webhook endpoints registered yet. Use the form above to add a Slack, Discord, or generic endpoint.
          </div>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-800 bg-slate-950/40 text-slate-400">
                <tr>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Name</th>
                  <th className="px-4 py-3">Type</th>
                  <th className="px-4 py-3">Min Severity</th>
                  <th className="px-4 py-3">Target URL</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {webhooks.map((item) => (
                  <tr key={item.webhook_id} className="hover:bg-slate-900/40">
                    <td className="px-4 py-3">
                      <button
                        type="button"
                        onClick={() => void handleToggle(item.webhook_id, item.is_enabled)}
                        className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 font-medium transition ${
                          item.is_enabled
                            ? "bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30"
                            : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                        }`}
                      >
                        <span
                          className={`h-1.5 w-1.5 rounded-full ${
                            item.is_enabled ? "bg-emerald-400 animate-pulse" : "bg-slate-500"
                          }`}
                        />
                        {item.is_enabled ? "Active" : "Paused"}
                      </button>
                    </td>

                    <td className="px-4 py-3 font-medium text-white">
                      {item.name}
                    </td>

                    <td className="px-4 py-3">
                      <span className="inline-flex items-center gap-1 rounded bg-slate-800 px-2 py-0.5 uppercase tracking-wider text-[10px] font-semibold text-cyan-300">
                        {item.service_type === "slack" ? (
                          <MessageSquare size={11} />
                        ) : (
                          <Globe size={11} />
                        )}
                        {item.service_type}
                      </span>
                    </td>

                    <td className="px-4 py-3">
                      <span
                        className={`rounded px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${
                          item.min_severity === "high"
                            ? "bg-rose-500/20 text-rose-300"
                            : item.min_severity === "medium"
                            ? "bg-amber-500/20 text-amber-300"
                            : "bg-blue-500/20 text-blue-300"
                        }`}
                      >
                        {item.min_severity}
                      </span>
                    </td>

                    <td className="px-4 py-3 font-mono text-[11px] text-slate-400 max-w-[200px] truncate">
                      {item.url}
                    </td>

                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          type="button"
                          onClick={() => void handleTestWebhook(item)}
                          disabled={testingId === item.webhook_id}
                          className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/80 px-2.5 py-1 text-slate-300 hover:bg-slate-700 hover:text-white disabled:opacity-50"
                          title="Send test alert ping"
                        >
                          {testingId === item.webhook_id ? (
                            <Loader2 size={12} className="animate-spin text-cyan-400" />
                          ) : (
                            <Send size={12} />
                          )}
                          Test
                        </button>

                        <button
                          type="button"
                          onClick={() => void handleDelete(item.webhook_id)}
                          className="rounded-lg p-1.5 text-slate-500 hover:bg-rose-950/40 hover:text-rose-400"
                          title="Delete webhook"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Delivery Logs Toggle */}
        <div className="mt-5 border-t border-slate-800 pt-4">
          <button
            type="button"
            onClick={() => setShowLogs(!showLogs)}
            className="flex items-center gap-2 text-xs font-medium text-slate-400 hover:text-cyan-300"
          >
            <History size={14} />
            <span>Recent Delivery History ({logs.length})</span>
            {showLogs ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {showLogs && (
            <div className="mt-3 space-y-2">
              {logs.length === 0 ? (
                <p className="text-xs text-slate-500">No delivery attempts recorded yet.</p>
              ) : (
                logs.map((log) => (
                  <div
                    key={log.log_id}
                    className="flex flex-col gap-2 rounded-xl border border-slate-800/80 bg-slate-950/60 p-3 text-xs sm:flex-row sm:items-center sm:justify-between"
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`h-2 w-2 rounded-full ${
                          log.success ? "bg-emerald-400" : "bg-rose-400"
                        }`}
                      />
                      <span className="font-semibold text-white">{log.webhook_name}</span>
                      <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] uppercase text-slate-400">
                        {log.event_type}
                      </span>
                      <span
                        className={`text-[10px] font-semibold uppercase ${
                          log.severity === "high" ? "text-rose-400" : "text-amber-400"
                        }`}
                      >
                        {log.severity}
                      </span>
                    </div>

                    <div className="flex items-center gap-3 text-slate-400 font-mono text-[11px]">
                      <span>HTTP {log.status_code || "ERR"}</span>
                      <span className="text-slate-500">
                        {new Date(log.created_at).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
