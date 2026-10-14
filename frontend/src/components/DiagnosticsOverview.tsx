
import {
  Activity,
  Brain,
  ChartNoAxesCombined,
  FlaskConical,
  Search,
  Sparkles,
} from "lucide-react";

const modules = [
  {
    href: "#rag-dashboard",
    title: "RAG Dashboard",
    description: "Evaluation metrics and trends",
    icon: ChartNoAxesCombined,
  },
  {
    href: "#evaluate",
    title: "RAG Evaluation",
    description: "Test retrieval and answer quality",
    icon: FlaskConical,
  },
  {
    href: "#deep-learning",
    title: "Deep Learning",
    description: "Diagnose and compare neural networks",
    icon: Brain,
  },
  {
    href: "#semantic-monitoring",
    title: "Semantic Monitoring",
    description: "Inspect text distribution changes",
    icon: Sparkles,
  },
  {
    href: "#drift-alerts",
    title: "Drift Alerts",
    description: "Review monitoring alerts",
    icon: Activity,
  },
  {
    href: "#tabular-experiments",
    title: "Tabular Experiments",
    description: "Run tabular drift analysis",
    icon: Search,
  },
];

export default function DiagnosticsOverview() {
  return (
    <section aria-labelledby="modules-heading">
      <div className="mb-4">
        <h2
          id="modules-heading"
          className="text-lg font-semibold text-white"
        >
          Diagnostic Modules
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Jump directly to a ModelAutopsy AI tool.
        </p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {modules.map(({ href, title, description, icon: Icon }) => (
          <a
            key={href}
            href={href}
            className="group rounded-xl border border-slate-800 bg-[#0d1424] p-4 transition hover:border-cyan-400/40 hover:bg-slate-900"
          >
            <div className="flex items-start gap-3">
              <div className="rounded-lg bg-cyan-400/10 p-2 text-cyan-300">
                <Icon size={20} />
              </div>

              <div>
                <h3 className="font-medium text-white group-hover:text-cyan-300">
                  {title}
                </h3>
                <p className="mt-1 text-sm leading-5 text-slate-400">
                  {description}
                </p>
              </div>
            </div>
          </a>
        ))}
      </div>
    </section>
  );
}
