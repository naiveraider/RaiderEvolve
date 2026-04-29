"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type ProfileFunction = {
  function: string;
  file: string;
  line: number;
  name: string;
  primitive_calls: number;
  total_calls: number;
  tottime: number;
  cumtime: number;
  percall_tottime: number;
  percall_cumtime: number;
};

type ProfileRun = {
  strategy: string;
  elapsed_seconds: number;
  total_calls: number;
  primitive_calls: number;
  prof_path?: string;
  top_functions: ProfileFunction[];
};

type ProfileSnapshot = {
  profile_id: string;
  created_at: string;
  task: string;
  strategies: string[];
  total_elapsed_seconds: number;
  system_profile: ProfileRun | null;
  runs: ProfileRun[];
};

const API_BASE =
  typeof process.env.NEXT_PUBLIC_API_URL === "string" &&
  process.env.NEXT_PUBLIC_API_URL.trim().length > 0
    ? process.env.NEXT_PUBLIC_API_URL.trim().replace(/\/$/, "")
    : "/api/backend";

function compactName(fn: ProfileFunction): string {
  const file = fn.file.split("/").slice(-2).join("/");
  return `${file}:${fn.line} ${fn.name}`;
}

export default function ProfilePage() {
  const [profile, setProfile] = useState<ProfileSnapshot | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadLatest = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/profiles/latest`, { cache: "no-store" });
      if (!res.ok) throw new Error(await res.text());
      const data = (await res.json()) as ProfileSnapshot;
      setProfile(data);
      const runs = data.system_profile ? [data.system_profile, ...data.runs] : data.runs;
      setSelectedStrategy((prev) =>
        prev && runs.some((run) => run.strategy === prev)
          ? prev
          : runs[0]?.strategy ?? "",
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadLatest();
  }, [loadLatest]);

  const profileRuns = useMemo(() => {
    if (!profile) return [];
    return profile.system_profile ? [profile.system_profile, ...profile.runs] : profile.runs;
  }, [profile]);

  const selectedRun = useMemo(() => {
    if (!profile) return null;
    return profileRuns.find((run) => run.strategy === selectedStrategy) ?? profileRuns[0] ?? null;
  }, [profile, profileRuns, selectedStrategy]);

  const chartData = useMemo(() => {
    return (selectedRun?.top_functions ?? []).slice(0, 12).map((fn) => ({
      name: compactName(fn),
      cumtime: fn.cumtime,
      tottime: fn.tottime,
      calls: fn.total_calls,
    }));
  }, [selectedRun]);

  return (
    <main>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
        <div>
          <h1>cProfile Results</h1>
          <p className="muted" style={{ marginTop: 0 }}>
            Latest backend profiling snapshot from an evolution run.
          </p>
        </div>
        <a href="/" className="muted">Back to evolution</a>
      </div>

      <section className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "center" }}>
          <div>
            <h2>Snapshot</h2>
            {profile ? (
              <p style={{ margin: 0 }}>
                Task <strong>{profile.task}</strong> · total{" "}
                <strong>{profile.total_elapsed_seconds.toFixed(3)}s</strong> ·{" "}
                <span className="muted">{new Date(profile.created_at).toLocaleString()}</span>
              </p>
            ) : (
              <p className="muted" style={{ margin: 0 }}>No profile loaded yet.</p>
            )}
          </div>
          <button type="button" className="primary" disabled={loading} onClick={() => void loadLatest()}>
            {loading ? "Loading..." : "Refresh"}
          </button>
        </div>
        {error && <p style={{ color: "#f87171" }}>{error}</p>}
      </section>

      {profile && selectedRun && (
        <>
          <section className="panel">
            <h2>Strategy</h2>
            <select
              value={selectedRun.strategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              style={{ maxWidth: 280 }}
            >
              {profileRuns.map((run) => (
                <option key={run.strategy} value={run.strategy}>
                  {run.strategy === "system" ? "whole system" : run.strategy} ({run.elapsed_seconds.toFixed(3)}s)
                </option>
              ))}
            </select>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 8,
                marginTop: 12,
              }}
            >
              {[
                ["Elapsed", `${selectedRun.elapsed_seconds.toFixed(3)}s`],
                ["Total calls", selectedRun.total_calls.toLocaleString()],
                ["Primitive calls", selectedRun.primitive_calls.toLocaleString()],
              ].map(([label, value]) => (
                <div key={label} style={{ background: "#0a0d12", border: "1px solid var(--border)", borderRadius: 6, padding: "0.75rem" }}>
                  <div className="muted" style={{ fontSize: 11 }}>{label}</div>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
            {selectedRun.prof_path && (
              <p className="muted">
                Raw cProfile file: <code>{selectedRun.prof_path}</code>
              </p>
            )}
          </section>

          <section className="panel">
            <h2>Top cumulative time</h2>
            <p className="muted">
              Bars show cumulative seconds spent in each function, including callees.
            </p>
            <div className="chart-wrap" style={{ height: 380 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ left: 220, right: 24 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#243044" />
                  <XAxis type="number" stroke="#8b9cb3" />
                  <YAxis dataKey="name" type="category" stroke="#8b9cb3" width={210} tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: "#141a22", border: "1px solid #243044" }} />
                  <Bar dataKey="cumtime" fill="#5eead4" name="cumtime (s)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="panel">
            <h2>Function table</h2>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ textAlign: "left", color: "var(--muted)" }}>
                    <th style={{ padding: "0.35rem" }}>Function</th>
                    <th>Total calls</th>
                    <th>Primitive</th>
                    <th>Self time</th>
                    <th>Cumulative</th>
                    <th>Per call</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedRun.top_functions.map((fn) => (
                    <tr key={fn.function} style={{ borderTop: "1px solid var(--border)" }}>
                      <td style={{ padding: "0.35rem", maxWidth: 520, wordBreak: "break-word" }}>
                        {compactName(fn)}
                      </td>
                      <td>{fn.total_calls.toLocaleString()}</td>
                      <td>{fn.primitive_calls.toLocaleString()}</td>
                      <td>{fn.tottime.toFixed(6)}s</td>
                      <td>{fn.cumtime.toFixed(6)}s</td>
                      <td>{fn.percall_cumtime.toFixed(6)}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </main>
  );
}
