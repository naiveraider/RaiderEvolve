from __future__ import annotations

from evolve.memory_store import MemoryStore
from evolve.models import CandidateRecord, TaskType


# Context size caps — smaller = faster LLM calls (fewer input tokens).
_CODE_CHARS: dict[str, int] = {
    "matrix": 1200,
    "pacman": 2500,
}
_N_BEST: dict[str, int] = {
    "matrix": 2,
    "pacman": 3,
}
_DEFAULT_CODE_CHARS = 2000
_DEFAULT_N_BEST     = 3


def build_llm_context(
    task: TaskType,
    population: list[CandidateRecord],
    memory: MemoryStore,
    generation: int,
    max_generations: int,
    single_shot: bool = False,
    algo_description: str = "",
) -> str:
    """
    Build the LLM prompt context.

    single_shot=True  (used by single_llm strategy)
    ───────────────────────────────────────────────
    Minimal prompt: only the function contract + current code.
    No fitness scores, no history, no algorithmic hints.
    The LLM must figure everything out on its own — its result is
    the fair "one-shot" baseline that the full evolution loop must beat.

    single_shot=False  (used by full / random_only strategies)
    ──────────────────────────────────────────────────────────
    Progressively richer hints as generation advances:
      gen < 40 % → problem description only
      gen ≥ 40 % → hint: cost-aware / goal-directed direction
      gen ≥ 70 % → concrete algorithm names + expected fitness scale
    """
    task_key   = task.value
    code_chars = _CODE_CHARS.get(task_key, _DEFAULT_CODE_CHARS)
    n_best     = _N_BEST.get(task_key, _DEFAULT_N_BEST)

    lines: list[str] = []

    # ── header + history (full mode only) ───────────────────────────────────
    if not single_shot:
        best_hist  = memory.best_n(n_best)
        worst_hist = memory.worst_n(1)
        prev_best  = max(
            (r.fitness for r in memory.records if r.generation < generation),
            default=None,
        )
        cur_best = max((r.fitness for r in population), default=None)

        lines.append(f"Gen {generation + 1}/{max_generations}.")
        if cur_best  is not None: lines.append(f"Best fitness now: {cur_best:.4f}")
        if prev_best is not None: lines.append(f"Prev best: {prev_best:.4f}")

        lines.append("\nTop solutions:")
        for i, r in enumerate(best_hist, 1):
            muls_info = ""
            if task == TaskType.MATRIX and r.metrics:
                muls = r.metrics.get("actual_muls")
                adds = r.metrics.get("actual_adds")
                if muls is not None:
                    muls_info = f"  muls={muls} adds={adds}"
            lines.append(f"{i}. fitness={r.fitness:.4f}{muls_info}")
            lines.append("CODE:\n" + r.code[:code_chars])

        lines.append("\nWorst (avoid):")
        for r in worst_hist:
            muls_info = ""
            if task == TaskType.MATRIX and r.metrics:
                muls = r.metrics.get("actual_muls")
                if muls is not None:
                    muls_info = f"  muls={muls}"
            lines.append(f"fitness={r.fitness:.4f}{muls_info}\n" + r.code[:600])
    else:
        # single_shot: only show current best code, no scores, no history
        best = max(population, key=lambda r: r.fitness) if population else None
        if best:
            lines.append("Current code:\n" + best.code[:code_chars])

    # ── task instructions ────────────────────────────────────────────────────
    if task == TaskType.PACMAN:
        if single_shot:
            # Bare minimum — contract only, zero algorithmic hints
            lines.append(
                "\nTASK: Improve this path-finder. Output ONLY a Python function.\n"
                "Contract: def search(start, goal, grid) -> list[(row, col)]\n"
                "  - start, goal: (row, col) tuples\n"
                "  - grid: 2-D list of chars  ('%'=wall, ' '=open, 'M'=mud)\n"
                "Return the path as a list of (row, col) cells from start to goal.\n"
                "Make the function find a better path. Output only the function."
            )
        else:
            # Full evolution: always provide complete algorithmic guidance.
            # The LLM needs rich hints every generation to climb DFS→BFS→UCS→A*.
            lines.append(
                "\nTASK: Improve the Pacman weighted-maze path-finder. Output ONLY python code.\n"
                "Contract: def search(start, goal, grid) -> list[(row, col)]\n"
                "  - start, goal : (row, col) tuples\n"
                "  - grid        : 2-D list of chars\n"
                "                  '%' = wall (impassable)\n"
                "                  ' ' = open passage (cost 1)\n"
                "                  'M' = mud (cost 5) — expensive!\n"
                "Scoring: 1000 - total_path_cost  (avoid mud to get higher score)\n"
                "The baseline uses DFS which wanders randomly and ignores mud cost entirely.\n"
                "EVALUATION MAZES:\n"
                "  mudMaze      — small maze with mud band; rewards cost-awareness\n"
                "  largeMudMaze — large maze (16×45): start at top-left of corridor,\n"
                "                 goal at top-right; a large open room hangs BELOW.\n"
                "                 BFS/UCS flood into the room → explore ~80% of cells.\n"
                "                 A* knows where the goal is → stays in the corridor → ~22%.\n"
                "\n"
                "Expected evolution chain (each step measurably improves fitness):\n"
                "  DFS (≈895) → BFS (≈928) → UCS/Dijkstra (≈952) → A* (≈960)\n"
                "\n"
                "TWO scoring components (both matter):\n"
                "  1. Path cost : 1000 − total_cost\n"
                "       'M' cells cost 5, ' ' cost 1.  DFS/BFS take mud → high cost.\n"
                "       UCS/A* avoid mud → low cost (+20 pts vs BFS on mudMaze).\n"
                "  2. Exploration penalty : −0.2 × (unique_cells_read / total_cells × 100)\n"
                "       BFS/UCS flood into the room → ~80% explored → −16 pts per run.\n"
                "       A* with Manhattan heuristic stays in corridor → ~22% → −4 pts.\n"
                "       Net A* advantage over UCS on largeMudMaze: ≈12 pts.\n"
                "\n"
                "Step-by-step improvements:\n"
                "  BFS : collections.deque, popleft; marks visited on enqueue.\n"
                "  UCS : heapq, priority=g; cost = 5 if grid[r][c]=='M' else 1.\n"
                "  A*  : heapq, priority=g+h; h=abs(r−goal[0])+abs(c−goal[1]).\n"
                "        The heuristic keeps the search in the corridor, away from the room."
            )

    else:  # MATRIX
        if single_shot:
            # Bare minimum — no fitness scale, no algorithm name
            lines.append(
                "\nTASK: def matmul(a, b) — multiply two 3×3 matrices (nested lists).\n"
                "The result must be numerically correct.\n"
                "GOAL: use fewer scalar multiplications than the current code.\n"
                "OUTPUT: Python function only, no markdown."
            )
        else:
            # Full evolution: always provide complete guidance
            lines.append(
                "\nTASK: def matmul(a,b) — 3×3 nested lists in/out. Must be numerically correct.\n"
                "GOAL: reduce scalar multiplications (counted at RUNTIME — loops × body iterations).\n"
                "FITNESS = 1.0 + (27-actual_muls)/27*10  (each saved mul = +0.37)\n"
                "  baseline 27 muls → 1.00 | 25 muls → 1.74 | 23 muls → 2.35 | 21 muls → 3.08\n"
                "Extra additions are FREE — trading adds for muls always improves fitness.\n"
                "Strategy: instead of computing each of the 27 products independently,\n"
                "  precompute products of linear combinations, e.g.:\n"
                "  m1 = (a[0][0]+a[0][1]) * b[1][0]  ← 1 mul gives partial sums for 2 cells\n"
                "  This Strassen/Laderman approach reduces to 23 muls for 3×3.\n"
                "OUTPUT: Python function only, no markdown."
            )

    if algo_description.strip():
        lines.append(
            "\nALGORITHM/PROBLEM DESCRIPTION (provided by user):\n"
            + algo_description.strip()
        )

    return "\n".join(lines)


SYSTEM_PROMPT = "You are an expert code optimizer. Respond with executable Python only, no markdown prose."
