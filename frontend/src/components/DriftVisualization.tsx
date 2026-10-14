
import { useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  ChevronDown,
  ChevronUp,
  Search,
} from "lucide-react";

type JsonRecord = Record<string, unknown>;

interface DriftVisualizationProps {
  result: JsonRecord | null;
}

interface FeatureRow {
  name: string;
  values: JsonRecord;
  severity: string;
}

function isRecord(value: unknown): value is JsonRecord {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function severityOf(values: JsonRecord): string {
  for (const [key, value] of Object.entries(values)) {
    if (
      /severity|risk_level|drift_level/i.test(key) &&
      typeof value === "string"
    ) {
      return value;
    }
  }

  return "Unknown";
}

function displayValue(value: unknown): string {
  if (value == null) return "—";

  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toFixed(4) : "—";
  }

  if (
    typeof value === "string" ||
    typeof value === "boolean"
  ) {
    return String(value);
  }

  if (Array.isArray(value)) {
    return `[${value.length} items]`;
  }

  return "{...}";
}

function severityStyle(severity: string) {
  const value = severity.toLowerCase();

  if (value.includes("critical")) {
    return "border-rose-500/30 bg-rose-500/10 text-rose-300";
  }

  if (value.includes("high")) {
    return "border-orange-500/30 bg-orange-500/10 text-orange-300";
  }

  if (value.includes("medium")) {
    return "border-amber-500/30 bg-amber-500/10 text-amber-300";
  }

  if (value.includes("low")) {
    return "border-cyan-500/30 bg-cyan-500/10 text-cyan-300";
  }

  return "border-slate-700 bg-slate-800/50 text-slate-400";
}

function findFeatureRows(
  value: unknown,
  path = "",
  depth = 0,
): FeatureRow[] {
  if (!isRecord(value) || depth > 5) {
    return [];
  }

  const rows: FeatureRow[] = [];

  for (const [key, child] of Object.entries(value)) {
    const currentPath = path ? `${path}.${key}` : key;

    if (!isRecord(child)) {
      continue;
    }

    const numericOrScalarFields = Object.entries(child).filter(
      ([, fieldValue]) =>
        fieldValue == null ||
        typeof fieldValue === "number" ||
        typeof fieldValue === "string" ||
        typeof fieldValue === "boolean",
    );

    const nestedFields = Object.values(child).some(isRecord);

    const looksLikeFeature =
      /feature|column|drift/i.test(key) &&
      numericOrScalarFields.length > 0;

    if (looksLikeFeature && !nestedFields) {
      rows.push({
        name: currentPath,
        values: child,
        severity: severityOf(child),
      });
    } else {
      rows.push(...findFeatureRows(child, currentPath, depth + 1));
    }
  }

  return rows;
}

export default function DriftVisualization({
  result,
}: DriftVisualizationProps) {
  const [search, setSearch] = useState("");
  const [expanded, setExpanded] = useState(false);

  const featureRows = useMemo(
    () => (result ? findFeatureRows(result) : []),
    [result],
  );

  const filteredRows = featureRows.filter((row) =>
    row.name.toLowerCase().includes(search.toLowerCase()),
  );

  const numericFields = useMemo(() => {
    if (!result) return [];

    const fields: Array<{
      name: string;
      value: number;
    }> = [];

    function walk(value: unknown, path = "", depth = 0) {
      if (!isRecord(value) || depth > 5) return;

      for (const [key, child] of Object.entries(value)) {
        const currentPath = path ? `${path}.${key}` : key;

        if (
          typeof child === "number" &&
          Number.isFinite(child)
        ) {
          fields.push({
            name: currentPath,
            value: child,
          });
        } else if (isRecord(child)) {
          walk(child, currentPath, depth + 1);
        }
      }
    }

    walk(result);

    return fields.slice(0, 30);
  }, [result]);

  if (!result) {
    return (
      <section className="rounded-2xl border border-slate-800 bg-[#0d1424] p-8 text-center">
        <BarChart3
          size={30}
          className="mx-auto mb-3 text-slate-500"
        />
        <h3 className="font-medium text-white">
          Drift visualizations
        </h3>
        <p className="mt-2 text-sm text-slate-500">
          Run a tabular drift experiment to inspect its results here.
        </p>
      </section>
    );
  }

  return (
    <section className="space-y-6">
      <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5">
        <div className="flex items-center gap-3">
          <div className="rounded-lg bg-cyan-400/10 p-2.5 text-cyan-300">
            <BarChart3 size={22} />
          </div>

          <div>
            <h3 className="font-semibold text-white">
              Numeric experiment outputs
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              Numeric fields found in the returned response.
            </p>
          </div>
        </div>

        {numericFields.length === 0 ? (
          <p className="mt-5 text-sm text-slate-500">
            No numeric fields were found in the response.
          </p>
        ) : (
          <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {numericFields.map((field) => (
              <div
                key={field.name}
                className="rounded-xl border border-slate-800 bg-slate-950/60 p-4"
              >
                <p className="break-all text-xs text-slate-500">
                  {field.name}
                </p>
                <p className="mt-2 break-all text-xl font-semibold text-white">
                  {field.value.toLocaleString(undefined, {
                    maximumFractionDigits: 5,
                  })}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-slate-800 bg-[#0d1424] p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="font-semibold text-white">
              Feature-level results
            </h3>
            <p className="mt-1 text-sm text-slate-500">
              Automatically discovered feature or column records.
            </p>
          </div>

          <span className="rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-400">
            {filteredRows.length} features
          </span>
        </div>

        <div className="relative mt-5">
          <Search
            size={16}
            className="absolute left-3 top-3 text-slate-500"
          />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search feature names..."
            className="w-full rounded-lg border border-slate-700 bg-slate-950 py-2.5 pl-9 pr-3 text-sm outline-none focus:border-cyan-400"
          />
        </div>

        {filteredRows.length === 0 ? (
          <div className="mt-5 rounded-xl border border-dashed border-slate-700 p-6 text-center">
            <AlertTriangle
              size={22}
              className="mx-auto mb-2 text-amber-300"
            />
            <p className="text-sm text-slate-300">
              No feature records could be identified automatically.
            </p>
            <p className="mt-2 text-xs leading-5 text-slate-500">
              The response may use a different schema. Expand the raw response
              below to inspect the returned structure.
            </p>
          </div>
        ) : (
          <div className="mt-5 overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full min-w-[600px] text-left text-sm">
              <thead className="bg-slate-900 text-xs uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="px-4 py-3">Feature</th>
                  <th className="px-4 py-3">Severity</th>
                  <th className="px-4 py-3">Metrics</th>
                </tr>
              </thead>

              <tbody>
                {filteredRows.map((row) => (
                  <tr
                    key={row.name}
                    className="border-t border-slate-800"
                  >
                    <td className="px-4 py-3 font-medium text-white">
                      {row.name}
                    </td>

                    <td className="px-4 py-3">
                      <span
                        className={`rounded-full border px-2.5 py-1 text-xs ${severityStyle(row.severity)}`}
                      >
                        {row.severity}
                      </span>
                    </td>

                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(row.values).map(([key, value]) => (
                          <span
                            key={key}
                            className="rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300"
                          >
                            {key}: {displayValue(value)}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-slate-800 bg-[#0d1424]">
        <button
          type="button"
          onClick={() => setExpanded((current) => !current)}
          className="flex w-full items-center justify-between p-5 text-left"
        >
          <span>
            <span className="block font-medium text-white">
              Raw experiment response
            </span>
            <span className="mt-1 block text-xs text-slate-500">
              Inspect the exact JSON returned by the backend.
            </span>
          </span>

          {expanded ? (
            <ChevronUp size={18} className="text-slate-400" />
          ) : (
            <ChevronDown size={18} className="text-slate-400" />
          )}
        </button>

        {expanded && (
          <pre className="max-h-[500px] overflow-auto border-t border-slate-800 p-5 text-xs leading-6 text-cyan-100">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </div>
    </section>
  );
}
