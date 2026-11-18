import { ArrowRight } from "lucide-react";
import { diagnosticModules } from "../constants/modules";

interface DiagnosticsOverviewProps {
  onSelectModule?: (moduleId: string) => void;
}

export default function DiagnosticsOverview({ onSelectModule }: DiagnosticsOverviewProps) {
  return (
    <section aria-labelledby="modules-heading" className="w-full">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2
            id="modules-heading"
            className="text-base font-semibold text-slate-100"
          >
            All Diagnostic Tools
          </h2>
          <p className="mt-0.5 text-xs text-slate-400">
            Click any tool below to start testing your models and data.
          </p>
        </div>
        <span className="rounded-full bg-slate-800 px-3 py-1 text-xs text-slate-300 font-medium border border-slate-700">
          {diagnosticModules.length} Available Tools
        </span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {diagnosticModules.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                if (onSelectModule) {
                  onSelectModule(item.id);
                } else {
                  const el = document.getElementById(item.id);
                  el?.scrollIntoView({ behavior: "smooth" });
                }
              }}
              className="group flex flex-col justify-between rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-left transition hover:border-blue-500/40 hover:bg-slate-800/80 hover:shadow-md focus:outline-none"
            >
              <div>
                <div className="mb-2.5 flex items-center justify-between">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-500/10 text-blue-400 group-hover:bg-blue-500/20">
                    <Icon size={16} />
                  </div>
                  {item.badge && (
                    <span className="rounded-full bg-slate-800 px-2 py-0.5 text-[10px] font-medium text-slate-300 border border-slate-700">
                      {item.badge}
                    </span>
                  )}
                </div>

                <h3 className="text-xs font-semibold text-slate-100 group-hover:text-blue-300">
                  {item.title}
                </h3>
                <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-slate-400">
                  {item.description}
                </p>
              </div>

              <div className="mt-3 flex items-center gap-1 text-xs font-medium text-blue-400 group-hover:text-blue-300">
                <span>Use tool</span>
                <ArrowRight size={12} className="transition-transform group-hover:translate-x-0.5" />
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
