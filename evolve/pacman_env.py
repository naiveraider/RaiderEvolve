"""
Fitness evaluator for evolved Pacman path-finding agents (weighted maze).

WHY weighted mazes show real A* improvement:
  On unweighted grids BFS is already optimal — A* finds the exact same path.
  In a weighted maze ('M' = mud, cost 5) the short straight path is expensive;
  the long outer path that avoids mud is cheap.  BFS ignores costs and walks
  through mud.  UCS / A* minimise total cost and find the cheap route.

  Gradient:   DFS < BFS < UCS / Dijkstra < A* (with Manhattan heuristic)

The evolved code must define:

    def search(start, goal, grid) -> list[(row, col)]

where
  - start / goal : (row, col) tuples
  - grid         : 2-D list of chars
                   '%' = impassable wall
                   ' ' = open cell (cost 1)
                   'M' = mud       (cost 5) — expensive to walk through
  - return       : ordered list of (row, col) from start to goal

Scoring:
    score = 1000 − total_path_cost
    total_path_cost = sum of cell costs along the path (M costs 5, all others 1)

Higher score = found a cheaper (mud-avoiding) route.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Any

from evolve.pacman import parse_weighted_layout, score_weighted_path

# Evaluation layouts:
#   mudMaze      — small mud maze: tests cost-awareness  (DFS<BFS<UCS≈A*)
#   largeMudMaze — large open maze: tests search efficiency (UCS<A* via exploration penalty)
_EVAL_LAYOUTS = ["mudMaze", "largeMudMaze"]

# Exploration penalty weight.  score -= EXPLORE_ALPHA * (unique_cells_read / total_cells * 100)
# Calibrated so UCS→A* gap ≈ 10–20 pts while UCS still beats BFS on cost savings.
_EXPLORE_ALPHA = 0.2

# Hard per-run timeout — any reasonable algorithm finishes in < 200 ms.
_RUN_TIMEOUT_SEC = 3.0


# ── counting grid proxy ────────────────────────────────────────────────────────

class _CountingRow:
    """
    Row proxy: increments the parent's `_visited` set only when a cell VALUE
    is read via row[c].  Calls to len(row) do NOT count — this ensures we
    measure unique cells inspected for content, not bounds checks.
    """
    __slots__ = ("_row", "_r", "_visited")

    def __init__(self, row, r: int, visited: set):
        self._row = row
        self._r = r
        self._visited = visited

    def __len__(self) -> int:
        return len(self._row)

    def __getitem__(self, c: int):
        self._visited.add((self._r, c))
        return self._row[c]

    def count(self, val):
        return self._row.count(val)

    def __eq__(self, other):
        return self._row == other


class _CountingGrid:
    """
    Transparent grid wrapper that tracks **unique (r, c) cell reads**.

    Agents write:  grid[r][c]  (normal access)
      grid[r]       → returns a _CountingRow (no count)
      row[c]        → records (r,c) in self._visited, returns cell value

    `self.accesses` = len(unique cells read = nodes genuinely inspected.
    Fewer accesses ≈ more efficient search (A* uses far fewer than UCS).
    """
    def __init__(self, grid: list):
        self._g = grid
        self._visited: set = set()

    def __len__(self) -> int:
        return len(self._g)

    def __getitem__(self, r: int) -> _CountingRow:
        return _CountingRow(self._g[r], r, self._visited)

    @property
    def accesses(self) -> int:
        return len(self._visited)


# ── agent loading ──────────────────────────────────────────────────────────────

def load_agent(code: str):
    """Execute code and return the `search` callable."""
    ns: dict[str, Any] = {"__builtins__": __builtins__}
    exec(code, ns, ns)  # noqa: S102
    if "search" not in ns:
        raise ValueError(
            "Agent must define: search(start, goal, grid) -> list[(r, c)]"
        )
    fn = ns["search"]
    if not callable(fn):
        raise ValueError("`search` must be callable")
    return fn


# ── timeout helper ─────────────────────────────────────────────────────────────

def _call_with_timeout(fn, args: tuple, timeout: float):
    result: list = [None]
    error:  list = [None]

    def _target():
        try:
            result[0] = fn(*args)
        except Exception as exc:
            error[0] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, TimeoutError(f"search exceeded {timeout}s")
    return result[0], error[0]


# ── single layout evaluation ───────────────────────────────────────────────────

def _run_one(search_fn, layout_name: str) -> tuple[float, dict]:
    """
    Run search_fn on one weighted layout; return (adjusted_score, detail).

    adjusted_score = raw_score − exploration_penalty
    exploration_penalty = EXPLORE_ALPHA * (cells_accessed / total_cells * 100)

    This rewards algorithms that explore fewer nodes (A* > UCS on large mazes).
    """
    start, goal, grid = parse_weighted_layout(layout_name)
    cg = _CountingGrid([row[:] for row in grid])
    total_cells = len(grid) * (len(grid[0]) if grid else 1)

    path, exc = _call_with_timeout(search_fn, (start, goal, cg), _RUN_TIMEOUT_SEC)

    if exc is not None or path is None:
        return -2000.0, {"layout": layout_name, "error": str(exc)}

    raw_score, cost, reached = score_weighted_path(path, start, goal, grid)
    explore_ratio   = cg.accesses / max(total_cells, 1)          # 0-1
    explore_penalty = _EXPLORE_ALPHA * explore_ratio * 100        # 0–30 pts
    adj_score       = raw_score - explore_penalty

    return float(adj_score), {
        "layout":           layout_name,
        "raw_score":        raw_score,
        "score":            round(adj_score, 2),
        "cost":             cost,
        "reached":          reached,
        "steps":            len(path) - 1 if path else 0,
        "cells_accessed":   cg.accesses,
        "total_cells":      total_cells,
        "explore_penalty":  round(explore_penalty, 2),
    }


# ── fitness ────────────────────────────────────────────────────────────────────

def pacman_fitness(
    code: str,
    rng: random.Random,
    w1: float,
    w2: float,
    w3: float,
    runs: int = 2,
    _t0: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Evaluate agent on weighted mud mazes.

    Fitness = avg_score − w3 * avg_cost
    (w1 and w2 are kept for API compatibility but avg_score already
    encodes quality; w3 adds a secondary cost penalty.)
    """
    wsum = w1 + w2 + w3
    if wsum <= 0:
        w1, w2, w3 = 0.5, 0.3, 0.2
        wsum = 1.0
    w1, w2, w3 = w1 / wsum, w2 / wsum, w3 / wsum

    t_start = time.perf_counter()

    try:
        agent = load_agent(code)
    except Exception as e:
        return -1e6, {"error": str(e)}

    chosen = [_EVAL_LAYOUTS[i % len(_EVAL_LAYOUTS)] for i in range(runs)]

    scores:    list[float] = []
    costs:     list[float] = []
    steps_list: list[int] = []
    successes: int = 0
    details:   list[dict]  = []

    for layout in chosen:
        score, detail = _run_one(agent, layout)
        scores.append(score)
        costs.append(detail.get("cost", 99999))
        steps_list.append(detail.get("steps", 0))
        if detail.get("reached"):
            successes += 1
        details.append(detail)

    avg_score   = sum(scores) / len(scores)
    avg_cost    = sum(costs)  / len(costs)
    avg_steps   = sum(steps_list) / len(steps_list)
    avg_explore = sum(d.get("cells_accessed", 0) for d in details) / len(details)
    eval_ms     = round((time.perf_counter() - t_start) * 1000, 1)

    fitness = avg_score - w3 * avg_cost * 0.01

    metrics = {
        "avg_score":          avg_score,
        "avg_cost":           avg_cost,
        "avg_steps":          avg_steps,
        "avg_cells_accessed": avg_explore,
        "success_rate":       successes / len(chosen),
        "eval_time_ms":       eval_ms,
        "layouts_used":       chosen,
        "run_details":        details,
        "weights":            {"w1": w1, "w2": w2, "w3": w3},
    }
    return fitness, metrics


# ── baseline ───────────────────────────────────────────────────────────────────

def baseline_pacman_code() -> str:
    """
    DFS baseline — finds *a* path but neither shortest nor cheapest.
    On mud mazes DFS wanders through many mud cells and takes long detours,
    giving a much lower score than BFS, UCS, or A*.

    Expected evolution chain (improving fitness each generation):
      DFS (baseline) → BFS → UCS / Dijkstra → A* with Manhattan heuristic
    """
    return '''\
def search(start, goal, grid):
    """DFS — depth-first search, finds a path but not shortest or cheapest.

    Grid legend:
      \'%\' = wall (impassable)
      \' \' = open passage (cost 1)
      \'M\' = mud (cost 5) — DFS blunders through it without hesitation!

    TWO fitness components (both matter):
      1. Path cost : 1000 - total_cost (mud=5, open=1)
         DFS takes the muddy short-cut → high cost → low raw_score
      2. Exploration penalty: -0.2 * (unique_cells_read / total_cells * 100)
         On largeMudMaze the large open room hangs below the corridor.
         BFS/UCS flood into the room (~80% of cells); A* stays in the
         corridor guided by Manhattan heuristic (~22% of cells).

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
                    and grid[nr][nc] != \'%\'
                    and nxt not in seen):
                seen.add(nxt)
                stack.append((nxt, path + [nxt]))
    return [start]
'''
