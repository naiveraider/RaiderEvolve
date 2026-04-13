from __future__ import annotations

import random

# ── Pacman search templates ────────────────────────────────────────────────────
# Each entry is (description, full replacement for the search function body hint).
# Template mutation swaps BFS fringe type or injects a heuristic nudge.

_PACMAN_FRINGE_VARIANTS = [
    # BFS (deque, FIFO)
    (
        "bfs_deque",
        "from collections import deque\n    fringe = deque([(start, [start])])\n"
        "    seen = {start}\n"
        "    while fringe:\n"
        "        state, path = fringe.popleft()\n",
    ),
    # DFS (list, LIFO) — may find longer path
    (
        "dfs_stack",
        "    fringe = [(start, [start])]\n"
        "    seen = {start}\n"
        "    while fringe:\n"
        "        state, path = fringe.pop()\n",
    ),
    # Greedy best-first toward goal (Manhattan)
    (
        "greedy_bfs",
        "    import heapq\n"
        "    h = lambda s: abs(s[0]-goal[0]) + abs(s[1]-goal[1])\n"
        "    fringe = [(h(start), start, [start])]\n"
        "    seen = {start}\n"
        "    while fringe:\n"
        "        _, state, path = heapq.heappop(fringe)\n",
    ),
]


def template_mutate(code: str, task: str, rng: random.Random) -> tuple[str, str]:
    """Apply a task-specific structural nudge to the code."""
    lines = code.splitlines()
    if not lines:
        return code, "template_skip"

    if task == "pacman":
        # Inject a heuristic hint comment near the neighbour-expansion loop.
        hint_lines = [
            "    # try: prioritise neighbour closest to goal (Manhattan)",
            "    # try: A* with h=abs(r-goal[0])+abs(c-goal[1])",
            "    # try: bidirectional BFS to reduce explored states",
            "    # try: iterative deepening for memory efficiency",
        ]
        hint = rng.choice(hint_lines)
        # Insert near a line that mentions 'for' or 'fringe', else random
        targets = [i for i, l in enumerate(lines) if "for" in l or "fringe" in l]
        idx = rng.choice(targets) if targets else rng.randrange(len(lines))
        lines.insert(idx, hint)
        return "\n".join(lines), f"template_hint: {hint.strip()}"

    # matrix task — inject the verified Laderman 1976 23-mul decomposition.
    # This is a CORRECT, working solution (fitness ≈ 2.35) that acts as a
    # structural jump from the 27-mul baseline so the LLM can then focus on
    # improving beyond 23 rather than re-discovering Strassen from scratch.
    from evolve.matrix_task import laderman_matrix_code
    return laderman_matrix_code(), "template_laderman: 23-mul seed"
