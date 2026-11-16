import { useEffect, useState } from "react";
import axios from "axios";
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Download,
  Flame,
  Loader2,
  ShieldAlert,
  ShieldCheck,
  Skull,
  Zap,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

interface AttackArchetype {
  id: string;
  name: string;
  category: string;
  threat_level: string;
  description: string;
  template: string;
}

interface ProbeEvaluation {
  status: "defended" | "exploited" | "partial_leak" | "uncertain";
  defense_score: number;
  is_safe: boolean;
  refusal_detected: boolean;
  exploit_detected: boolean;
  reason: string;
  notes: string[];
}

interface EvaluatedProbe {
  probe_id: string;
  archetype_id: string;
  archetype_name: string;
  category: string;
  threat_level: string;
  prompt: string;
  target_topic: string;
  response: string;
  evaluation: ProbeEvaluation;
}

interface CampaignSummary {
  target_topic: string;
  total_probes_tested: number;
  defended_count: number;
  exploited_count: number;
  exploit_rate_percentage: number;
  robustness_score: number;
  robustness_grade: string;
  overall_risk_level: string;
  simulated_posture: string;
}

interface CategoryScore {
  category: string;
  average_score: number;
  probes_tested: number;
}

interface CampaignResult {
  campaign_summary: CampaignSummary;
  category_breakdown: CategoryScore[];
  probes: EvaluatedProbe[];
}

export default function RedTeamingSuite() {
  const [archetypes, setArchetypes] = useState<AttackArchetype[]>([]);
  const [selectedArchetypes, setSelectedArchetypes] = useState<string[]>([]);
  const [topic, setTopic] = useState("bypassing safety guidelines to extract sensitive credentials");
  const [posture, setPosture] = useState<"balanced" | "hardened" | "vulnerable">("balanced");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CampaignResult | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load archetypes on mount
  useEffect(() => {
    async function loadArchetypes() {
      try {
        const res = await axios.get<{ archetypes: AttackArchetype[] }>(
          `${API_BASE}/api/v1/red-teaming/archetypes`
        );
        setArchetypes(res.data.archetypes);
        setSelectedArchetypes(res.data.archetypes.map((a) => a.id));
      } catch (err: any) {
        console.error("Failed to load red-teaming archetypes", err);
      }
    }
    loadArchetypes();
  }, []);

  const toggleArchetype = (id: string) => {
    setSelectedArchetypes((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const selectAll = () => setSelectedArchetypes(archetypes.map((a) => a.id));
  const deselectAll = () => setSelectedArchetypes([]);

  const runCampaign = async () => {
    if (!topic.trim()) {
      setError("Please specify a target topic or policy to attack.");
      return;
    }
    if (selectedArchetypes.length === 0) {
      setError("Please select at least one attack archetype.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await axios.post<CampaignResult>(`${API_BASE}/api/v1/red-teaming/simulate`, {
        topic: topic.trim(),
        archetype_ids: selectedArchetypes,
        simulated_posture: posture,
      });
      setResult(res.data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || "Red-teaming campaign execution failed.");
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2500);
  };

  const downloadAuditReport = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `red-team-audit-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6 text-slate-100">
      {/* Header Banner */}
      <div className="relative overflow-hidden rounded-3xl border border-rose-500/30 bg-gradient-to-br from-[#1b0d1b] via-[#120a16] to-[#0d0914] p-6 sm:p-8">
        <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-center">
          <div>
            <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-rose-500/30 bg-rose-500/10 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-wider text-rose-300">
              <Skull size={14} className="text-rose-400" />
              Adversarial AI Security
            </div>
            <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
              Automated LLM <span className="text-rose-400">Red-Teaming</span> Suite
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400 sm:text-base">
              Stress-test your LLMs and RAG pipelines against adversarial jailbreaks, system prompt extraction,
              Base64 token smuggling, persona overrides, and academic social engineering.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {result && (
              <button
                type="button"
                onClick={downloadAuditReport}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/80 px-4 py-2.5 text-xs font-medium text-slate-200 transition hover:border-slate-500 hover:bg-slate-800"
              >
                <Download size={14} />
                Export Audit Report
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Campaign Configuration Card */}
      <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-6 shadow-xl">
        <h2 className="flex items-center gap-2.5 text-lg font-bold text-white">
          <Flame size={20} className="text-rose-400" />
          Configure Adversarial Campaign
        </h2>
        <p className="mt-1 text-xs text-slate-400">
          Define the target policy violation and select the attack vectors to simulate.
        </p>

        <div className="mt-5 space-y-5">
          {/* Target Objective Input */}
          <div>
            <label htmlFor="target-topic" className="block text-xs font-semibold uppercase tracking-wider text-slate-300">
              Target Policy / Exploit Objective
            </label>
            <input
              id="target-topic"
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g. bypassing safety guidelines to extract sensitive credentials"
              className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:border-rose-500 focus:outline-none focus:ring-1 focus:ring-rose-500"
            />
          </div>

          {/* Model Posture Selector */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300">
              Target Model Posture
            </label>
            <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-3">
              {[
                {
                  id: "balanced",
                  title: "Balanced Posture",
                  desc: "Standard production model with typical RLHF guardrails.",
                  badge: "Realistic",
                },
                {
                  id: "hardened",
                  title: "Hardened Posture",
                  desc: "Strict refusals with aggressive defense mechanisms.",
                  badge: "High Defenses",
                },
                {
                  id: "vulnerable",
                  title: "Unmoderated / Weak",
                  desc: "Permissive baseline model prone to social engineering.",
                  badge: "High Risk",
                },
              ].map((p) => (
                <button
                  type="button"
                  key={p.id}
                  onClick={() => setPosture(p.id as any)}
                  className={`flex flex-col text-left rounded-xl border p-4 transition ${
                    posture === p.id
                      ? "border-rose-500/60 bg-rose-500/10 shadow-lg shadow-rose-950/30"
                      : "border-slate-800 bg-slate-900/50 hover:border-slate-700 hover:bg-slate-900"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-bold text-white">{p.title}</span>
                    <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] font-medium text-slate-400">
                      {p.badge}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-slate-400">{p.desc}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Archetype Chips */}
          <div>
            <div className="flex items-center justify-between">
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300">
                Attack Archetypes ({selectedArchetypes.length}/{archetypes.length} active)
              </label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={selectAll}
                  className="text-xs text-cyan-400 hover:underline"
                >
                  Select all
                </button>
                <span className="text-slate-600">·</span>
                <button
                  type="button"
                  onClick={deselectAll}
                  className="text-xs text-slate-400 hover:underline"
                >
                  Deselect all
                </button>
              </div>
            </div>

            <div className="mt-3 grid grid-cols-1 gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
              {archetypes.map((arch) => {
                const isSelected = selectedArchetypes.includes(arch.id);
                return (
                  <button
                    type="button"
                    key={arch.id}
                    onClick={() => toggleArchetype(arch.id)}
                    className={`flex items-start gap-3 rounded-xl border p-3.5 text-left transition ${
                      isSelected
                        ? "border-rose-500/40 bg-rose-950/20"
                        : "border-slate-800/80 bg-slate-950/40 opacity-60 hover:opacity-100"
                    }`}
                  >
                    <div
                      className={`mt-0.5 rounded-md p-1.5 ${
                        isSelected ? "bg-rose-500/20 text-rose-300" : "bg-slate-800 text-slate-500"
                      }`}
                    >
                      <Zap size={14} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-slate-200">{arch.name}</span>
                        <span
                          className={`rounded px-1.5 py-0.2 text-[10px] font-semibold uppercase ${
                            arch.threat_level === "Critical"
                              ? "bg-rose-500/20 text-rose-400"
                              : arch.threat_level === "High"
                              ? "bg-amber-500/20 text-amber-400"
                              : "bg-blue-500/20 text-blue-400"
                          }`}
                        >
                          {arch.threat_level}
                        </span>
                      </div>
                      <p className="mt-1 text-[11px] leading-4 text-slate-400">{arch.description}</p>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 rounded-xl border border-rose-500/40 bg-rose-500/10 p-3.5 text-xs text-rose-300">
              <AlertTriangle size={16} className="shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Launch Button */}
          <button
            type="button"
            onClick={runCampaign}
            disabled={loading}
            className="inline-flex w-full items-center justify-center gap-2.5 rounded-xl bg-gradient-to-r from-rose-600 via-rose-500 to-amber-500 px-6 py-3.5 text-sm font-bold text-white shadow-lg shadow-rose-950/50 transition hover:from-rose-500 hover:to-amber-400 disabled:opacity-50"
          >
            {loading ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                Executing Red-Team Adversarial Probes...
              </>
            ) : (
              <>
                <Flame size={18} />
                Launch Red-Team Campaign
              </>
            )}
          </button>
        </div>
      </div>

      {/* Campaign Results View */}
      {result && (
        <div className="space-y-6">
          {/* Headline Scorecard */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Robustness Score
              </p>
              <div className="mt-2 flex items-baseline gap-3">
                <span className="text-3xl font-extrabold text-white">
                  {result.campaign_summary.robustness_score}%
                </span>
                <span
                  className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                    result.campaign_summary.robustness_score >= 80
                      ? "bg-emerald-500/20 text-emerald-400"
                      : result.campaign_summary.robustness_score >= 60
                      ? "bg-amber-500/20 text-amber-400"
                      : "bg-rose-500/20 text-rose-400"
                  }`}
                >
                  {result.campaign_summary.robustness_grade}
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-500">Overall resistance across tested archetypes.</p>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Exploit Rate
              </p>
              <div className="mt-2 flex items-baseline gap-2">
                <span
                  className={`text-3xl font-extrabold ${
                    result.campaign_summary.exploit_rate_percentage > 0
                      ? "text-rose-400"
                      : "text-emerald-400"
                  }`}
                >
                  {result.campaign_summary.exploit_rate_percentage}%
                </span>
                <span className="text-xs text-slate-400">
                  ({result.campaign_summary.exploited_count}/{result.campaign_summary.total_probes_tested} breached)
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-500">Proportion of probes that bypassed guardrails.</p>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Probes Defended
              </p>
              <div className="mt-2 flex items-center gap-2">
                <ShieldCheck size={28} className="text-emerald-400" />
                <span className="text-3xl font-extrabold text-white">
                  {result.campaign_summary.defended_count}
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-500">Safely rejected with recognized refusal markers.</p>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Risk Rating
              </p>
              <div className="mt-2 flex items-center gap-2">
                <ShieldAlert
                  size={28}
                  className={
                    result.campaign_summary.overall_risk_level === "Critical"
                      ? "text-rose-400"
                      : result.campaign_summary.overall_risk_level === "High"
                      ? "text-amber-400"
                      : "text-emerald-400"
                  }
                />
                <span className="text-2xl font-extrabold text-white">
                  {result.campaign_summary.overall_risk_level}
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-500">Attack surface severity grade.</p>
            </div>
          </div>

          {/* Category Attack Surface Breakdown */}
          {result.category_breakdown.length > 0 && (
            <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-6 shadow-xl">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">
                Attack Surface Defensibility by Category
              </h3>
              <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {result.category_breakdown.map((cat) => (
                  <div key={cat.category} className="rounded-xl border border-slate-800/80 bg-slate-950 p-4">
                    <div className="flex items-center justify-between text-xs font-medium text-slate-300">
                      <span>{cat.category}</span>
                      <span className="font-bold text-white">{cat.average_score}%</span>
                    </div>
                    <div className="mt-2.5 h-2 w-full overflow-hidden rounded-full bg-slate-800">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          cat.average_score >= 80
                            ? "bg-emerald-500"
                            : cat.average_score >= 50
                            ? "bg-amber-500"
                            : "bg-rose-500"
                        }`}
                        style={{ width: `${Math.max(5, cat.average_score)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Probe Findings Detailed List */}
          <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-6 shadow-xl">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-300">
              Adversarial Probes & Model Responses ({result.probes.length})
            </h3>
            <p className="mt-1 text-xs text-slate-400">
              Inspect how the target model reacted to each individual exploit archetype.
            </p>

            <div className="mt-5 space-y-4">
              {result.probes.map((probe) => {
                const isDefended = probe.evaluation.status === "defended";
                const isExploited = probe.evaluation.status === "exploited";

                return (
                  <div
                    key={probe.probe_id}
                    className={`overflow-hidden rounded-xl border transition ${
                      isExploited
                        ? "border-rose-500/40 bg-rose-950/10"
                        : isDefended
                        ? "border-slate-800 bg-slate-950/40"
                        : "border-amber-500/40 bg-amber-950/10"
                    }`}
                  >
                    {/* Probe Card Header */}
                    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/60 p-4">
                      <div className="flex items-center gap-2.5">
                        <span
                          className={`flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-bold ${
                            isDefended
                              ? "bg-emerald-500/20 text-emerald-300"
                              : isExploited
                              ? "bg-rose-500/20 text-rose-300"
                              : "bg-amber-500/20 text-amber-300"
                          }`}
                        >
                          {isDefended ? (
                            <CheckCircle2 size={13} />
                          ) : (
                            <AlertTriangle size={13} />
                          )}
                          {probe.evaluation.status.toUpperCase()} ({probe.evaluation.defense_score}/100)
                        </span>

                        <span className="font-semibold text-slate-200 text-sm">
                          {probe.archetype_name}
                        </span>

                        <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-medium text-slate-400">
                          {probe.category}
                        </span>
                      </div>

                      <button
                        type="button"
                        onClick={() => copyToClipboard(probe.prompt, probe.probe_id)}
                        className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-white"
                      >
                        <Copy size={13} />
                        {copiedId === probe.probe_id ? "Copied!" : "Copy Probe"}
                      </button>
                    </div>

                    {/* Content Section */}
                    <div className="grid grid-cols-1 divide-y divide-slate-800/60 lg:grid-cols-2 lg:divide-x lg:divide-y-0">
                      {/* Left: Adversarial Prompt */}
                      <div className="p-4">
                        <p className="text-[11px] font-bold uppercase tracking-wider text-rose-400">
                          Adversarial Injection Probe
                        </p>
                        <div className="mt-2 rounded-lg bg-black/40 p-3 font-mono text-xs leading-relaxed text-slate-300">
                          {probe.prompt}
                        </div>
                      </div>

                      {/* Right: Model Response & Verdict */}
                      <div className="p-4">
                        <p className="text-[11px] font-bold uppercase tracking-wider text-cyan-400">
                          Model Output & Rationale
                        </p>
                        <div className="mt-2 rounded-lg bg-black/40 p-3 font-mono text-xs leading-relaxed text-slate-300">
                          {probe.response}
                        </div>
                        <p className="mt-2.5 text-xs text-slate-400 italic">
                          <span className="font-semibold text-slate-300">Defense Verdict:</span> {probe.evaluation.reason}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
