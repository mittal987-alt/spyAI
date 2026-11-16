import { useMemo, useState } from "react";
import axios from "axios";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  FileSpreadsheet,
  LoaderCircle,
  Upload,
} from "lucide-react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

interface ColumnStatistics {
  count: number;
  mean: number | null;
  median: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
}

interface DatasetColumn {
  name: string;
  dtype: string;
  kind: "numeric" | "categorical";
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  constant: boolean;
  statistics?: ColumnStatistics;
  top_values?: Array<{ value: string; count: number }>;
}

interface DatasetProfile {
  dataset: {
    row_count: number;
    column_count: number;
    memory_usage_bytes: number;
    duplicate_rows: number;
    duplicate_percentage: number;
    missing_cells: number;
    missing_percentage: number;
    numeric_column_count: number;
    categorical_column_count: number;
    constant_column_count: number;
  };
  columns: DatasetColumn[];
  numeric_columns: string[];
  categorical_columns: string[];
  constant_columns: string[];
  correlation: Record<string, Record<string, number | null>>;
  preview: {
    columns: string[];
    rows: unknown[][];
  };
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    return (
      error.response?.data?.detail ||
      error.message ||
      "The request failed."
    );
  }

  return error instanceof Error ? error.message : "An unexpected error occurred.";
}

export default function DatasetExplorer() {
  const [file, setFile] = useState<File | null>(null);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [previewRows, setPreviewRows] = useState(10);
  const [selectedColumn, setSelectedColumn] = useState("");
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState("");

  const selectedColumnData = useMemo(() => {
    if (!profile) return undefined;

    return profile.columns.find(
      (column) => column.name === selectedColumn,
    );
  }, [profile, selectedColumn]);

  const missingChartData = useMemo(() => {
    if (!profile) return [];

    return profile.columns
      .filter((column) => column.missing_count > 0)
      .map((column) => ({
        name: column.name,
        missing: column.missing_count,
      }))
      .sort((a, b) => b.missing - a.missing)
      .slice(0, 10);
  }, [profile]);

  async function analyzeDataset() {
    if (!file) {
      setError("Select a CSV file first.");
      return;
    }

    setLoading(true);
    setError("");
    setProfile(null);
    setSelectedColumn("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post<DatasetProfile>(
        `${API_BASE_URL}/api/v1/datasets/profile`,
        formData,
        {
          params: { preview_rows: previewRows },
        },
      );

      setProfile(response.data);
      setSelectedColumn(response.data.columns[0]?.name ?? "");
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  async function downloadReport() {
    if (!file) return;

    setDownloading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(
        `${API_BASE_URL}/api/v1/datasets/report`,
        formData,
        { responseType: "blob" },
      );

      const blob = new Blob([response.data], {
        type: "application/json",
      });

      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");

      link.href = url;
      link.download = "dataset-report.json";
      document.body.appendChild(link);
      link.click();
      link.remove();

      URL.revokeObjectURL(url);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="space-y-6 text-slate-100">
      <header>
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-cyan-500/15 p-3 text-cyan-300">
            <Database size={25} />
          </div>

          <div>
            <h1 className="text-2xl font-bold">Dataset Explorer</h1>
            <p className="mt-1 text-sm text-slate-400">
              Explore your data, identify quality issues, and inspect features.
            </p>
          </div>
        </div>
      </header>

      <section className="rounded-2xl border border-slate-700 bg-slate-900 p-5">
        <div className="flex items-center gap-2">
          <FileSpreadsheet className="text-cyan-300" size={20} />
          <h2 className="text-lg font-semibold">Upload dataset</h2>
        </div>

        <p className="mt-2 text-sm text-slate-400">
          Upload a CSV file. Maximum size: 10 MB, 50,000 rows, and 200 columns.
        </p>

        <div className="mt-4 flex flex-col gap-4 md:flex-row md:items-end">
          <div className="min-w-0 flex-1">
            <label
              htmlFor="dataset-file"
              className="mb-2 block text-sm font-medium"
            >
              CSV file
            </label>

            <input
              id="dataset-file"
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => {
                const nextFile = event.target.files?.[0] ?? null;

                setFile(nextFile);
                setProfile(null);
                setError("");
              }}
              className="block w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm file:mr-4 file:rounded-md file:border-0 file:bg-cyan-500/15 file:px-3 file:py-2 file:text-cyan-200"
            />
          </div>

          <div className="w-full md:w-40">
            <label
              htmlFor="preview-rows"
              className="mb-2 block text-sm font-medium"
            >
              Preview rows
            </label>

            <select
              id="preview-rows"
              value={previewRows}
              onChange={(event) =>
                setPreviewRows(Number(event.target.value))
              }
              className="w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm"
            >
              {[10, 25, 50, 100].map((count) => (
                <option key={count} value={count}>
                  {count} rows
                </option>
              ))}
            </select>
          </div>

          <button
            type="button"
            onClick={analyzeDataset}
            disabled={!file || loading}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-500 px-5 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? (
              <LoaderCircle size={18} className="animate-spin" />
            ) : (
              <Upload size={18} />
            )}
            {loading ? "Analyzing..." : "Analyze dataset"}
          </button>
        </div>

        {file && (
          <p className="mt-3 text-xs text-slate-400">
            Selected: {file.name} · {formatBytes(file.size)}
          </p>
        )}

        {error && (
          <div
            role="alert"
            className="mt-4 flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200"
          >
            <AlertCircle size={18} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </section>

      {!profile && !loading && (
        <section className="rounded-2xl border border-dashed border-slate-700 bg-slate-900/50 p-10 text-center">
          <BarChart3
            size={38}
            className="mx-auto mb-3 text-slate-500"
          />
          <h2 className="font-semibold">Ready to explore your data?</h2>
          <p className="mt-2 text-sm text-slate-400">
            Upload a CSV file and click Analyze dataset to generate your report.
          </p>
        </section>
      )}

      {loading && (
        <div className="flex items-center justify-center gap-3 rounded-xl border border-slate-700 bg-slate-900 p-10 text-slate-300">
          <LoaderCircle className="animate-spin" size={22} />
          Analyzing your dataset...
        </div>
      )}

      {profile && (
        <>
          <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {[
              {
                label: "Total rows",
                value: profile.dataset.row_count.toLocaleString(),
              },
              {
                label: "Total columns",
                value: profile.dataset.column_count.toLocaleString(),
              },
              {
                label: "Missing cells",
                value: profile.dataset.missing_cells.toLocaleString(),
              },
              {
                label: "Duplicate rows",
                value: profile.dataset.duplicate_rows.toLocaleString(),
              },
            ].map((item) => (
              <div
                key={item.label}
                className="rounded-2xl border border-slate-700 bg-slate-900 p-5"
              >
                <p className="text-sm text-slate-400">{item.label}</p>
                <p className="mt-2 text-2xl font-bold">{item.value}</p>
              </div>
            ))}
          </section>

          <section className="rounded-2xl border border-slate-700 bg-slate-900 p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-lg font-semibold">Dataset quality</h2>
                <p className="mt-1 text-sm text-slate-400">
                  Memory usage: {formatBytes(profile.dataset.memory_usage_bytes)}
                </p>
              </div>

              <button
                type="button"
                onClick={downloadReport}
                disabled={downloading}
                className="rounded-lg border border-slate-600 px-4 py-2 text-sm font-medium hover:bg-slate-800 disabled:opacity-50"
              >
                {downloading ? "Preparing report..." : "Download JSON report"}
              </button>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-4 lg:grid-cols-4">
              <div className="rounded-xl bg-slate-950 p-4">
                <p className="text-xs text-slate-400">Missing data</p>
                <p className="mt-2 text-xl font-semibold">
                  {profile.dataset.missing_percentage}%
                </p>
              </div>

              <div className="rounded-xl bg-slate-950 p-4">
                <p className="text-xs text-slate-400">Duplicate rate</p>
                <p className="mt-2 text-xl font-semibold">
                  {profile.dataset.duplicate_percentage}%
                </p>
              </div>

              <div className="rounded-xl bg-slate-950 p-4">
                <p className="text-xs text-slate-400">Numeric features</p>
                <p className="mt-2 text-xl font-semibold">
                  {profile.dataset.numeric_column_count}
                </p>
              </div>

              <div className="rounded-xl bg-slate-950 p-4">
                <p className="text-xs text-slate-400">Constant features</p>
                <p className="mt-2 text-xl font-semibold">
                  {profile.dataset.constant_column_count}
                </p>
              </div>
            </div>

            <div className="mt-5 flex flex-wrap gap-3 text-sm">
              <span className="inline-flex items-center gap-2 rounded-lg bg-emerald-500/10 px-3 py-2 text-emerald-300">
                <CheckCircle2 size={16} />
                {profile.dataset.numeric_column_count} numeric columns
              </span>

              <span className="inline-flex items-center gap-2 rounded-lg bg-blue-500/10 px-3 py-2 text-blue-300">
                {profile.dataset.categorical_column_count} categorical columns
              </span>

              {profile.constant_columns.length > 0 && (
                <span className="rounded-lg bg-amber-500/10 px-3 py-2 text-amber-300">
                  Constant: {profile.constant_columns.join(", ")}
                </span>
              )}
            </div>
          </section>

          <section className="rounded-2xl border border-slate-700 bg-slate-900 p-5">
            <h2 className="text-lg font-semibold">Missing values by column</h2>
            <p className="mt-1 text-sm text-slate-400">
              The ten columns with the most missing values.
            </p>

            {missingChartData.length === 0 ? (
              <div className="mt-5 rounded-xl bg-emerald-500/10 p-5 text-sm text-emerald-300">
                No missing values were detected.
              </div>
            ) : (
              <div className="mt-5 h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={missingChartData}
                    margin={{ top: 10, right: 20, left: 5, bottom: 45 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#334155"
                    />
                    <XAxis
                      dataKey="name"
                      angle={-30}
                      textAnchor="end"
                      interval={0}
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                    />
                    <YAxis
                      allowDecimals={false}
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#0f172a",
                        border: "1px solid #334155",
                        borderRadius: 8,
                      }}
                    />
                    <Bar
                      dataKey="missing"
                      name="Missing values"
                      fill="#22d3ee"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </section>

          <section className="rounded-2xl border border-slate-700 bg-slate-900 p-5">
            <h2 className="text-lg font-semibold">Feature analysis</h2>
            <p className="mt-1 text-sm text-slate-400">
              Select a feature to inspect its statistics or category frequencies.
            </p>

            <div className="mt-4">
              <label
                htmlFor="selected-column"
                className="mb-2 block text-sm font-medium"
              >
                Select column
              </label>

              <select
                id="selected-column"
                value={selectedColumn}
                onChange={(event) => setSelectedColumn(event.target.value)}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 p-3 text-sm md:max-w-md"
              >
                {profile.columns.map((column) => (
                  <option key={column.name} value={column.name}>
                    {column.name}
                  </option>
                ))}
              </select>
            </div>

            {selectedColumnData && (
              <div className="mt-5">
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  {[
                    {
                      label: "Data type",
                      value: selectedColumnData.dtype,
                    },
                    {
                      label: "Unique values",
                      value: selectedColumnData.unique_count,
                    },
                    {
                      label: "Missing count",
                      value: selectedColumnData.missing_count,
                    },
                    {
                      label: "Missing percentage",
                      value: `${selectedColumnData.missing_percentage}%`,
                    },
                  ].map((item) => (
                    <div
                      key={item.label}
                      className="rounded-xl bg-slate-950 p-4"
                    >
                      <p className="text-xs text-slate-400">{item.label}</p>
                      <p className="mt-2 break-words font-semibold">
                        {item.value}
                      </p>
                    </div>
                  ))}
                </div>

                {selectedColumnData.statistics && (
                  <div className="mt-5 overflow-x-auto">
                    <h3 className="mb-3 font-semibold">
                      Numerical statistics
                    </h3>

                    <table className="w-full min-w-[550px] text-left text-sm">
                      <thead className="text-slate-400">
                        <tr>
                          {[
                            "Count",
                            "Mean",
                            "Median",
                            "Std. deviation",
                            "Minimum",
                            "Maximum",
                          ].map((heading) => (
                            <th key={heading} className="p-3">
                              {heading}
                            </th>
                          ))}
                        </tr>
                      </thead>

                      <tbody>
                        <tr className="border-t border-slate-700">
                          {[
                            selectedColumnData.statistics.count,
                            selectedColumnData.statistics.mean,
                            selectedColumnData.statistics.median,
                            selectedColumnData.statistics.std,
                            selectedColumnData.statistics.min,
                            selectedColumnData.statistics.max,
                          ].map((value, index) => (
                            <td key={index} className="p-3">
                              {typeof value === "number"
                                ? Number(value.toFixed(4)).toString()
                                : "—"}
                            </td>
                          ))}
                        </tr>
                      </tbody>
                    </table>
                  </div>
                )}

                {selectedColumnData.top_values && (
                  <div className="mt-5">
                    <h3 className="mb-3 font-semibold">
                      Most frequent categories
                    </h3>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead className="text-slate-400">
                          <tr>
                            <th className="p-3">Value</th>
                            <th className="p-3">Frequency</th>
                          </tr>
                        </thead>

                        <tbody>
                          {selectedColumnData.top_values.map((item) => (
                            <tr
                              key={item.value}
                              className="border-t border-slate-700"
                            >
                              <td className="max-w-xs break-words p-3">
                                {item.value}
                              </td>
                              <td className="p-3">{item.count}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>

          <section className="rounded-2xl border border-slate-700 bg-slate-900 p-5">
            <h2 className="text-lg font-semibold">Dataset preview</h2>
            <p className="mt-1 text-sm text-slate-400">
              Showing the first {profile.preview.rows.length} rows.
            </p>

            <div className="mt-4 max-h-[480px] overflow-auto rounded-lg border border-slate-700">
              <table className="w-full whitespace-nowrap text-left text-sm">
                <thead className="sticky top-0 bg-slate-800 text-slate-300">
                  <tr>
                    {profile.preview.columns.map((column) => (
                      <th key={column} className="p-3 font-semibold">
                        {column}
                      </th>
                    ))}
                  </tr>
                </thead>

                <tbody>
                  {profile.preview.rows.map((row, rowIndex) => (
                    <tr
                      key={rowIndex}
                      className="border-t border-slate-800 hover:bg-slate-800/60"
                    >
                      {row.map((value, columnIndex) => (
                        <td key={columnIndex} className="p-3">
                          {value === null || value === undefined
                            ? "—"
                            : String(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {Object.keys(profile.correlation).length > 1 && (
            <section className="rounded-2xl border border-slate-700 bg-slate-900 p-5">
              <h2 className="text-lg font-semibold">
                Numerical correlation matrix
              </h2>
              <p className="mt-1 text-sm text-slate-400">
                Pearson correlations between numeric columns. Values range
                from -1 to 1; correlation does not establish causation.
              </p>

              <div className="mt-4 max-h-[480px] overflow-auto">
                <table className="w-full text-left text-sm">
                  <thead className="sticky top-0 bg-slate-800">
                    <tr>
                      <th className="p-3">Feature</th>
                      {Object.keys(profile.correlation).map((column) => (
                        <th key={column} className="p-3">
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>

                  <tbody>
                    {Object.entries(profile.correlation).map(
                      ([rowName, values]) => (
                        <tr
                          key={rowName}
                          className="border-t border-slate-800"
                        >
                          <td className="p-3 font-medium">{rowName}</td>

                          {Object.keys(profile.correlation).map(
                            (column) => {
                              const value = values[column];

                              return (
                                <td key={column} className="p-3">
                                  {typeof value === "number"
                                    ? value.toFixed(3)
                                    : "—"}
                                </td>
                              );
                            },
                          )}
                        </tr>
                      ),
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}