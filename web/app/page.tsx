"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Task = "pacman" | "matrix";
type FitnessPreset = "pacman" | "matrix" | "custom";
type Strategy = "single_llm" | "random_only" | "full";

type CandidateRecord = {
  id: string;
  generation: number;
  code: string;
  fitness: number;
  parents: string[];
  strategy_tag: string;
  mutation_notes: string;
  metrics: Record<string, unknown>;
};

type GenerationLog = {
  generation: number;
  selection_summary: string;
  mutation_explanations: string[];
  best_id: string;
  avg_fitness: number;
  best_fitness: number;
};

type StrategyRunResult = {
  strategy: Strategy;
  best_code: string;
  fitness_curve: number[];
  avg_fitness_per_gen: number[];
  best_per_generation: number[];
  history: GenerationLog[];
  memory_records: CandidateRecord[];
  final_best_fitness: number;
};

type EvolutionResponse = {
  task: Task;
  runs: StrategyRunResult[];
  pseudocode_outline: string | null;
  algorithm_explanation: string | null;
};

type ChartRow = { generation: number } & Partial<Record<Strategy, number | null>>;

// ── default source code shown in the textarea ─────────────────────────────────
const BASELINE_PACMAN = `def search(start, goal, grid):
    """DFS — depth-first search.  Finds a path, but not shortest or cheapest.

    Grid legend:
      '%' = wall (impassable)
      ' ' = open passage (cost 1)
      'M' = mud (cost 5)  ← DFS blunders through mud without hesitation!

    Score = 1000 - total_path_cost  (higher = cheaper path).
    TWO fitness components (both matter):
      1. Path cost    : 1000 - total_cost (mud=5, open=1)
      2. Exploration  : penalty for reading too many grid cells
         (BFS/UCS flood into the open room below the corridor;
          A* stays in the corridor and reaches the goal directly)

    Expected fitness: DFS≈895 → BFS≈928 → UCS≈952 → A*≈960
    """
    stack = [(start, [start])]
    seen = {start}
    while stack:
        s, path = stack.pop()
        if s == goal:
            return path
        r, c = s
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if (0 <= nr < len(grid)
                    and 0 <= nc < len(grid[nr])
                    and grid[nr][nc] != '%'
                    and nxt not in seen):
                seen.add(nxt)
                stack.append((nxt, path + [nxt]))
    return [start]
`;

const BASELINE_MATRIX = `def matmul(a, b):
    """Standard 3×3 matrix multiply: C = A × B
    C[i][j] = sum(A[i][k] * B[k][j] for k in range(3))

    Operations: 27 multiplications, 18 additions → fitness = 1.0 (baseline)

    fitness = 1.0 + (27 - actual_muls) / 27 * 10
      25 muls → 1.74  |  23 muls (Laderman 1976) → 2.35  |  21 muls → 3.08
    Muls counted at RUNTIME — this loop performs 27 real multiplications.
    """
    c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c[i][j] += a[i][k] * b[k][j]
    return c
`;

const DEFAULT_CODE: Record<Task, string> = {
  pacman: BASELINE_PACMAN,
  matrix: BASELINE_MATRIX,
};

const DEFAULT_PRESET: Record<Task, FitnessPreset> = {
  pacman: "pacman",
  matrix: "matrix",
};

/** Prefer NEXT_PUBLIC_API_URL in dev to hit FastAPI directly (avoids proxy timeouts). */
const API_BASE =
  typeof process.env.NEXT_PUBLIC_API_URL === "string" &&
  process.env.NEXT_PUBLIC_API_URL.trim().length > 0
    ? process.env.NEXT_PUBLIC_API_URL.trim().replace(/\/$/, "")
    : "/api/backend";

function bestUpTo(records: CandidateRecord[], maxGen: number): number | null {
  const sub = records.filter((r) => r.generation <= maxGen);
  if (!sub.length) return null;
  return Math.max(...sub.map((r) => r.fitness));
}

export default function Page() {
  const [task, setTask] = useState<Task>("pacman");
  const [sourceCode, setSourceCode] = useState<string>(DEFAULT_CODE.pacman);
  const [generations, setGenerations] = useState(4);
  const [populationSize, setPopulationSize] = useState(6);
  const [topK, setTopK] = useState(3);
  const [fitnessPreset, setFitnessPreset] = useState<FitnessPreset>("pacman");
  const [w1, setW1] = useState(0.5);
  const [w2, setW2] = useState(0.3);
  const [w3, setW3] = useState(0.2);
  // matrix_alpha / matrix_beta are kept for API shape compatibility but the
  // backend now uses hardcoded MUL_WEIGHT=10 / ADD_WEIGHT=0.05 internally.
  const matrixAlpha = 0.0;
  const matrixBeta  = 0.0;
  const [strategies, setStrategies] = useState<Strategy[]>([
    "single_llm",
    "random_only",
    "full",
  ]);
  const [selectionMode, setSelectionMode] = useState<"top_k" | "elite" | "diversity">(
    "top_k"
  );

  // When task changes: sync default code, fitness preset, and fast defaults
  useEffect(() => {
    setSourceCode(DEFAULT_CODE[task]);
    setFitnessPreset(DEFAULT_PRESET[task]);
    if (task === "matrix") {
      // matmul LLM calls are slower per token; use fewer generations + 1 strategy
      setGenerations(3);
      setPopulationSize(3);
      setStrategies(["full"]);
    } else {
      setGenerations(4);
      setPopulationSize(6);
      setStrategies(["single_llm", "random_only", "full"]);
    }
  }, [task]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EvolutionResponse | null>(null);
  const [genCap, setGenCap] = useState(10);
  const [codeRunIdx, setCodeRunIdx] = useState(0);
  const [genRange, setGenRange] = useState<[number, number]>([0, 999]);

  // SSE progress tracking
  const [progress, setProgress] = useState<{
    strategy: string;
    gen: number;
    total: number;
    best: number | null;
    status: string;
  } | null>(null);
  const [strategyDone, setStrategyDone] = useState<string[]>([]);
  // live chart data per strategy streamed before final result arrives
  const [liveChart, setLiveChart] = useState<ChartRow[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const toggleStrategy = (s: Strategy) => {
    setStrategies((prev) => (prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]));
  };

  const cancelEvolve = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const runEvolve = useCallback(async () => {
    if (!strategies.length) {
      setError("Select at least one comparison strategy.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setProgress(null);
    setStrategyDone([]);
    setLiveChart([]);

    const abort = new AbortController();
    abortRef.current = abort;

    const body = {
      task,
      source_code: sourceCode,
      generations,
      population_size: populationSize,
      top_k: topK,
      selection_mode: selectionMode,
      fitness_preset: task === "matrix" ? "matrix" : fitnessPreset,
      custom_weights:
        fitnessPreset === "custom" && task === "pacman" ? { w1, w2, w3 } : null,
      matrix_alpha: matrixAlpha,
      matrix_beta: matrixBeta,
      strategies,
      include_pseudocode_log: true,
    };

    try {
      const res = await fetch(`${API_BASE}/evolve/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: abort.signal,
      });
      if (!res.ok) throw new Error(await res.text());

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      // live chart accumulator: strategy → best_per_gen list
      const liveByStrategy: Record<string, number[]> = {};

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });

        // SSE events are separated by double-newline
        const events = buf.split("\n\n");
        buf = events.pop() ?? "";

        for (const raw of events) {
          const line = raw.replace(/^data: /, "").trim();
          if (!line) continue;
          let evt: Record<string, unknown>;
          try { evt = JSON.parse(line) as Record<string, unknown>; }
          catch { continue; }

          if (evt.type === "progress") {
            setProgress({
              strategy: String(evt.strategy ?? ""),
              gen: Number(evt.gen ?? 0),
              total: Number(evt.total ?? generations),
              best: evt.best != null ? Number(evt.best) : null,
              status: String(evt.status ?? ""),
            });
          } else if (evt.type === "strategy_done") {
            const s = String(evt.strategy);
            setStrategyDone((prev) => [...prev, s]);
            const bpg = (evt.best_per_generation as number[]) ?? [];
            liveByStrategy[s] = bpg;
            // rebuild live chart
            const maxLen = Math.max(0, ...Object.values(liveByStrategy).map((a) => a.length));
            const rows: ChartRow[] = [];
            for (let i = 0; i < maxLen; i++) {
              const row: ChartRow = { generation: i };
              for (const [strat, arr] of Object.entries(liveByStrategy)) {
                row[strat as Strategy] = arr[i] ?? null;
              }
              rows.push(row);
            }
            setLiveChart(rows);
          } else if (evt.type === "done") {
            const data = evt.result as EvolutionResponse;
            setResult(data);
            const full = data.runs.find((r) => r.strategy === "full");
            const maxG =
              full?.best_per_generation.length ??
              Math.max(0, ...data.runs.map((r) => r.best_per_generation.length - 1));
            setGenCap(Math.min(10, maxG));
            setGenRange([0, maxG]);
          } else if (evt.type === "error") {
            throw new Error(String(evt.detail ?? "Unknown server error"));
          }
        }
      }
    } catch (e: unknown) {
      if ((e as { name?: string }).name === "AbortError") {
        setError("Cancelled.");
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setLoading(false);
      setProgress(null);
    }
  }, [
    strategies,
    task,
    sourceCode,
    generations,
    populationSize,
    topK,
    selectionMode,
    fitnessPreset,
    w1,
    w2,
    w3,
  ]);

  const comparisonData = useMemo(() => {
    if (result) {
      const maxLen = Math.max(0, ...result.runs.map((r) => r.best_per_generation.length));
      const rows: ChartRow[] = [];
      for (let i = 0; i < maxLen; i++) {
        const row: ChartRow = { generation: i };
        for (const run of result.runs) {
          row[run.strategy] =
            run.best_per_generation[i] !== undefined ? run.best_per_generation[i]! : null;
        }
        rows.push(row);
      }
      return rows;
    }
    return liveChart;
  }, [result, liveChart]);

  const filteredComparison = useMemo(() => {
    const [a, b] = genRange;
    return comparisonData.filter((row) => row.generation >= a && row.generation <= b);
  }, [comparisonData, genRange]);

  const primaryRun = result?.runs[codeRunIdx] ?? result?.runs[0];
  const fullRun = result?.runs.find((r) => r.strategy === "full");

  // Pacman-specific per-generation metrics extracted from the best record each gen
  type PacmanMetricRow = {
    generation: number;
    steps: number | null;
    cost: number | null;
    cells: number | null;
    runtime_ms: number | null;
    success: number | null;
  };
  const pacmanMetrics = useMemo((): PacmanMetricRow[] => {
    if (!result || task !== "pacman") return [];
    const run = result.runs.find((r) => r.strategy === "full") ?? result.runs[0];
    if (!run) return [];
    // best record per generation
    const byGen = new Map<number, CandidateRecord>();
    for (const rec of run.memory_records) {
      const cur = byGen.get(rec.generation);
      if (!cur || rec.fitness > cur.fitness) byGen.set(rec.generation, rec);
    }
    const maxGen = Math.max(0, ...byGen.keys());
    return Array.from({ length: maxGen + 1 }, (_, i) => {
      const m = byGen.get(i)?.metrics as Record<string, unknown> | undefined;
      return {
        generation: i,
        steps:      (m?.avg_steps    as number | undefined) ?? null,
        cost:       (m?.avg_cost     as number | undefined) ?? null,
        cells:      (m?.avg_cells_accessed as number | undefined) ?? null,
        runtime_ms: (m?.eval_time_ms as number | undefined) ?? null,
        success:    m?.success_rate != null ? Math.round((m.success_rate as number) * 100) : null,
      };
    });
  }, [result, task]);

  const bestUpToValue = useMemo(() => {
    if (!fullRun) return null;
    return bestUpTo(fullRun.memory_records, genCap);
  }, [fullRun, genCap]);

  const exportCsv = async () => {
    if (!result) return;
    const res = await fetch(`${API_BASE}/export/fitness-csv`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(result),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "fitness.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main>
      <h1>Evolve System</h1>
      <p className="muted" style={{ marginTop: 0 }}>
        OpenEvolve-style loop: selection → context → LLM / random / template mutation →
        evaluation → memory.
      </p>

      <div className="row">
        <div>
          <section className="panel">
            <h2>Input</h2>
            <label>Task</label>
            <select value={task} onChange={(e) => setTask(e.target.value as Task)}>
              <option value="pacman">Pacman agent (choose_action)</option>
              <option value="matrix">3×3 matrix multiply (matmul)</option>
            </select>
            <p className="muted" style={{ marginTop: "0.75rem" }}>
              Edit or replace the baseline below, or leave it as-is to evolve from scratch.
            </p>
            <label style={{ marginTop: "0.5rem" }}>Source code</label>
            <textarea
              value={sourceCode}
              onChange={(e) => setSourceCode(e.target.value)}
            />
          </section>

          <section className="panel">
            <h2>Controls</h2>
            <label>Generations (N)</label>
            <input
              type="number"
              min={1}
              max={200}
              value={generations}
              onChange={(e) => setGenerations(+e.target.value)}
            />
            <label style={{ marginTop: "0.5rem" }}>Population size</label>
            <input
              type="number"
              min={2}
              max={32}
              value={populationSize}
              onChange={(e) => setPopulationSize(+e.target.value)}
            />
            <label style={{ marginTop: "0.5rem" }}>Top-k / elite size</label>
            <input
              type="number"
              min={1}
              max={16}
              value={topK}
              onChange={(e) => setTopK(+e.target.value)}
            />
            <label style={{ marginTop: "0.5rem" }}>Selection</label>
            <select
              value={selectionMode}
              onChange={(e) => setSelectionMode(e.target.value as typeof selectionMode)}
            >
              <option value="top_k">Top-k</option>
              <option value="diversity">Diversity-aware</option>
              <option value="elite">Elite only</option>
            </select>
            <label style={{ marginTop: "0.5rem" }}>Fitness preset</label>
            <select
              value={fitnessPreset}
              onChange={(e) => setFitnessPreset(e.target.value as FitnessPreset)}
            >
              <option value="pacman">Pacman (w1·score + w2·survival − w3·steps)</option>
              <option value="matrix">Matrix (correctness + operation cost)</option>
              <option value="custom">Custom weights (Pacman task only)</option>
            </select>
            {fitnessPreset === "custom" && task === "pacman" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                <div>
                  <label>w1</label>
                  <input type="number" step={0.05} value={w1} onChange={(e) => setW1(+e.target.value)} />
                </div>
                <div>
                  <label>w2</label>
                  <input type="number" step={0.05} value={w2} onChange={(e) => setW2(+e.target.value)} />
                </div>
                <div>
                  <label>w3</label>
                  <input type="number" step={0.05} value={w3} onChange={(e) => setW3(+e.target.value)} />
                </div>
              </div>
            )}
            {task === "matrix" && (
              <div style={{
                marginTop: 8,
                padding: "10px 12px",
                background: "var(--surface2, #1e1e2e)",
                borderRadius: 6,
                fontSize: "0.82rem",
                lineHeight: 1.65,
                color: "var(--text2, #cdd6f4)",
                fontFamily: "monospace",
              }}>
                <div style={{ fontWeight: 600, marginBottom: 4, fontFamily: "sans-serif", fontSize: "0.78rem", textTransform: "uppercase", letterSpacing: "0.05em", opacity: 0.7 }}>Fitness = correctness + operation cost</div>
                <div>fitness = <strong>correctness</strong> (1 if correct, −10 if wrong)</div>
                <div>&nbsp;&nbsp;&nbsp;&nbsp;+ <strong style={{color:"#a6e3a1"}}>mul savings</strong> = (27 − actual_muls) / 27 × 10</div>
                <div>&nbsp;&nbsp;&nbsp;&nbsp;+ <strong>add savings</strong> = (27 − actual_adds) / 27 × 0.05</div>
                <div style={{ marginTop: 6, opacity: 0.75, fontFamily: "sans-serif" }}>
                  Operations counted at <em>runtime</em> — loops × iterations, not AST symbols.
                </div>
                <div style={{ marginTop: 4, borderTop: "1px solid #313244", paddingTop: 6, display: "grid", gridTemplateColumns: "1fr 1fr", gap: "2px 16px" }}>
                  <span>Standard loop (27 muls)</span><span style={{color:"#f38ba8"}}>1.000</span>
                  <span>25 muls</span><span style={{color:"#fab387"}}>≈ 1.74</span>
                  <span>Laderman 1976 (23 muls)</span><span style={{color:"#a6e3a1"}}>≈ 2.35</span>
                </div>
              </div>
            )}
            <label style={{ marginTop: "0.75rem" }}>Comparison strategies</label>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
              {(
                [
                  ["single_llm", "No evolution (single LLM)"],
                  ["random_only", "Random / template only"],
                  ["full", "LLM-guided evolution"],
                ] as const
              ).map(([k, label]) => (
                <label key={k} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <input
                    type="checkbox"
                    checked={strategies.includes(k)}
                    onChange={() => toggleStrategy(k)}
                  />
                  {label}
                </label>
              ))}
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: "1rem" }}>
              <button
                type="button"
                className="primary"
                style={{ flex: 1 }}
                disabled={loading}
                onClick={() => void runEvolve()}
              >
                {loading ? "Running…" : "Run evolution"}
              </button>
              {loading && (
                <button
                  type="button"
                  style={{
                    padding: "0.5rem 0.9rem",
                    background: "#374151",
                    color: "#f87171",
                    border: "1px solid #4b5563",
                    borderRadius: 6,
                    cursor: "pointer",
                  }}
                  onClick={cancelEvolve}
                >
                  Cancel
                </button>
              )}
            </div>

            {/* Progress bar */}
            {loading && progress && (
              <div style={{ marginTop: "0.75rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>
                  <span>
                    {progress.strategy} — gen {progress.gen}/{progress.total}
                    {progress.status === "llm_call" ? " (LLM…)" : ""}
                  </span>
                  {progress.best != null && (
                    <span>best {progress.best.toFixed(4)}</span>
                  )}
                </div>
                <div style={{ height: 6, background: "#1e293b", borderRadius: 3, overflow: "hidden" }}>
                  <div
                    style={{
                      height: "100%",
                      width: `${Math.round((progress.gen / Math.max(1, progress.total)) * 100)}%`,
                      background: "var(--accent)",
                      borderRadius: 3,
                      transition: "width 0.3s ease",
                    }}
                  />
                </div>
                {strategyDone.length > 0 && (
                  <p className="muted" style={{ marginTop: 4, fontSize: 12 }}>
                    Done: {strategyDone.join(", ")}
                  </p>
                )}
              </div>
            )}

            {error && <p style={{ color: "#f87171", marginTop: "0.5rem" }}>{error}</p>}
          </section>
        </div>

        <div>
          <section className="panel">
            <h2>Visualization</h2>
            <p className="muted">
              Best fitness per generation (per strategy).
              {loading && liveChart.length > 0 && (
                <span style={{ color: "var(--accent)", marginLeft: 8 }}>● live</span>
              )}
            </p>
            <div className="chart-wrap">
              {filteredComparison.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={filteredComparison}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#243044" />
                    <XAxis dataKey="generation" stroke="#8b9cb3" />
                    <YAxis stroke="#8b9cb3" />
                    <Tooltip
                      contentStyle={{ background: "#141a22", border: "1px solid #243044" }}
                    />
                    <Legend />
                    {(result?.runs.map((r) => r.strategy) ?? (Object.keys(
                      filteredComparison[0] ?? {}
                    ).filter((k) => k !== "generation") as Strategy[])).map((strat, i) => (
                      <Line
                        key={strat}
                        type="monotone"
                        dataKey={strat}
                        stroke={["#5eead4", "#fbbf24", "#a78bfa"][i % 3]}
                        dot={false}
                        connectNulls
                        name={strat}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <p className="muted">Run evolution to plot curves.</p>
              )}
            </div>
            <label>Generation range filter</label>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="number"
                min={0}
                value={genRange[0]}
                onChange={(e) => setGenRange([+e.target.value, genRange[1]])}
              />
              <span>–</span>
              <input
                type="number"
                min={0}
                value={genRange[1]}
                onChange={(e) => setGenRange([genRange[0], +e.target.value])}
              />
            </div>
          </section>

          {/* ── Pacman runtime metrics ───────────────────────────── */}
          {task === "pacman" && pacmanMetrics.length > 0 && (
            <section className="panel">
              <h2>Pacman metrics over generations</h2>
              <p className="muted">
                Best candidate per generation (full-evolution run). Lower cost/steps/cells = better.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>

                {/* Steps */}
                <div>
                  <p style={{ fontSize: "0.8rem", margin: "0 0 4px", color: "var(--muted)" }}>
                    Avg Steps (path length)
                  </p>
                  <div style={{ height: 160 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={pacmanMetrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#243044" />
                        <XAxis dataKey="generation" stroke="#8b9cb3" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#8b9cb3" tick={{ fontSize: 11 }} width={40} />
                        <Tooltip contentStyle={{ background: "#141a22", border: "1px solid #243044", fontSize: 12 }} />
                        <Line type="monotone" dataKey="steps" stroke="#5eead4" dot={false} connectNulls name="steps" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Path Cost */}
                <div>
                  <p style={{ fontSize: "0.8rem", margin: "0 0 4px", color: "var(--muted)" }}>
                    Avg Path Cost (mud=5, open=1)
                  </p>
                  <div style={{ height: 160 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={pacmanMetrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#243044" />
                        <XAxis dataKey="generation" stroke="#8b9cb3" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#8b9cb3" tick={{ fontSize: 11 }} width={40} />
                        <Tooltip contentStyle={{ background: "#141a22", border: "1px solid #243044", fontSize: 12 }} />
                        <Line type="monotone" dataKey="cost" stroke="#fbbf24" dot={false} connectNulls name="cost" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Runtime — wall-clock ms the generated code took to run */}
                <div>
                  <p style={{ fontSize: "0.8rem", margin: "0 0 4px", color: "var(--muted)" }}>
                    Code Runtime (ms)
                  </p>
                  <div style={{ height: 160 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={pacmanMetrics}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#243044" />
                        <XAxis dataKey="generation" stroke="#8b9cb3" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#8b9cb3" tick={{ fontSize: 11 }} width={40} />
                        <Tooltip contentStyle={{ background: "#141a22", border: "1px solid #243044", fontSize: 12 }} />
                        <Line type="monotone" dataKey="runtime_ms" stroke="#f87171" dot={false} connectNulls name="ms" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

              </div>

              {/* Generation count summary */}
              <div style={{
                marginTop: 12,
                display: "grid",
                gridTemplateColumns: "repeat(4, 1fr)",
                gap: 8,
                fontSize: "0.82rem",
              }}>
                {[
                  { label: "Generations",   value: pacmanMetrics.length },
                  { label: "Final steps",   value: pacmanMetrics.at(-1)?.steps?.toFixed(0)      ?? "—" },
                  { label: "Final cost",    value: pacmanMetrics.at(-1)?.cost?.toFixed(1)        ?? "—" },
                  { label: "Final runtime", value: pacmanMetrics.at(-1)?.runtime_ms != null ? `${pacmanMetrics.at(-1)!.runtime_ms} ms` : "—" },
                ].map(({ label, value }) => (
                  <div key={label} style={{
                    padding: "8px 10px",
                    background: "var(--surface2, #1e1e2e)",
                    borderRadius: 6,
                    textAlign: "center",
                  }}>
                    <div style={{ color: "var(--muted)", fontSize: "0.72rem", marginBottom: 2 }}>{label}</div>
                    <strong>{String(value)}</strong>
                  </div>
                ))}
              </div>
            </section>
          )}

          <section className="panel">
            <h2>Figure 2 — best up to generation</h2>
            <p className="muted">
              Uses the full evolution run&apos;s memory. Drag to pick a generation cap.
            </p>
            <input
              type="range"
              min={0}
              max={Math.max(
                0,
                ...(fullRun?.memory_records.map((r) => r.generation) ?? [0])
              )}
              value={genCap}
              onChange={(e) => setGenCap(+e.target.value)}
              style={{ width: "100%" }}
            />
            <p>
              Best score up to generation {genCap}:{" "}
              <strong>{bestUpToValue != null ? bestUpToValue.toFixed(4) : "—"}</strong>
            </p>
          </section>
        </div>
      </div>

      {result && (
        <>
          <section className="panel">
            <h2>Comparison summary</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ textAlign: "left", color: "var(--muted)" }}>
                  <th style={{ padding: "0.35rem" }}>Strategy</th>
                  <th>Final best fitness</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {result.runs.map((r) => (
                  <tr key={r.strategy} style={{ borderTop: "1px solid var(--border)" }}>
                    <td style={{ padding: "0.35rem" }}>{r.strategy}</td>
                    <td>{r.final_best_fitness.toFixed(4)}</td>
                    <td className="muted">
                      {r.strategy === "full" && "Hybrid LLM + random + template"}
                      {r.strategy === "random_only" && "No LLM mutation"}
                      {r.strategy === "single_llm" && "Single-shot LLM vs seed"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button type="button" className="primary" style={{ marginTop: "0.75rem" }} onClick={exportCsv}>
              Download fitness CSV
            </button>
          </section>

          <section className="panel">
            <h2>Best solution code</h2>
            <label>Strategy</label>
            <select
              value={codeRunIdx}
              onChange={(e) => setCodeRunIdx(+e.target.value)}
              style={{ marginBottom: "0.5rem" }}
            >
              {result.runs.map((r, i) => (
                <option key={r.strategy} value={i}>
                  {r.strategy} (fitness {r.final_best_fitness.toFixed(2)})
                </option>
              ))}
            </select>
            <code className="pre">{primaryRun?.best_code ?? ""}</code>
          </section>

          <section className="panel">
            <h2>Evolution history (full run)</h2>
            {fullRun ? (
              <div className="log">
                {fullRun.history.map((h) => (
                  <div key={h.generation} style={{ marginBottom: "0.75rem" }}>
                    <strong>Gen {h.generation}</strong> — best {h.best_fitness.toFixed(4)}, avg{" "}
                    {h.avg_fitness.toFixed(4)}
                    <div className="muted">{h.selection_summary}</div>
                    {h.mutation_explanations?.length > 0 && (
                      <div className="muted">{h.mutation_explanations.join(" · ")}</div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">Include &quot;LLM-guided evolution&quot; to see full logs.</p>
            )}
          </section>

          {(result.pseudocode_outline || result.algorithm_explanation) && (
            <section className="panel">
              <h2>Pseudocode &amp; algorithm notes</h2>
              <code className="pre">{result.pseudocode_outline ?? ""}</code>
              <p style={{ marginTop: "0.75rem" }}>{result.algorithm_explanation}</p>
            </section>
          )}
        </>
      )}
    </main>
  );
}
