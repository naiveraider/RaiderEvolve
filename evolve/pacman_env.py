from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable


DIRS = ("N", "S", "E", "W")
MOVES = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


@dataclass
class PacmanEvalResult:
    score: int
    survival_steps: int
    total_steps: int
    early_death: bool
    inefficient_moves: int


def load_agent(code: str) -> Callable[[dict[str, Any]], str]:
    ns: dict[str, Any] = {"__builtins__": __builtins__}
    exec(code, ns, ns)  # noqa: S102
    if "choose_action" not in ns:
        raise ValueError("Pacman agent must define choose_action(state) -> 'N'|'S'|'E'|'W'")
    fn = ns["choose_action"]
    if not callable(fn):
        raise ValueError("choose_action must be callable")
    return fn


def _wall(grid: list[list[str]], r: int, c: int) -> bool:
    if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]):
        return True
    return grid[r][c] == "#"


def run_episode(
    agent: Callable[[dict[str, Any]], str],
    rng: random.Random,
    max_steps: int = 220,
) -> PacmanEvalResult:
    grid = [
        list("##########"),
        list("#........#"),
        list("#..#.....#"),
        list("#........#"),
        list("#....#...#"),
        list("#........#"),
        list("#.....#..#"),
        list("#........#"),
        list("##########"),
    ]
    pac = (4, 4)
    foods = {(r, c) for r in range(1, 8) for c in range(1, 9) if grid[r][c] == "."}
    ghosts = [(2, 5), (6, 6)]
    score = 0
    last_pos = pac
    inefficient = 0
    early_death = False

    step = -1
    for step in range(max_steps):
        if not foods:
            break
        foods_list = sorted(foods)
        nf = None
        if foods_list:
            nf = min(foods_list, key=lambda p: abs(p[0] - pac[0]) + abs(p[1] - pac[1]))
        state = {
            "step": step,
            "pacman": pac,
            "foods": foods_list,
            "ghosts": list(ghosts),
            "nearest_food": nf,
            "score": score,
        }
        try:
            action = str(agent(state)).upper()
            if action not in MOVES:
                action = "E"
        except Exception:
            action = "E"

        dr, dc = MOVES[action]
        nr, nc = pac[0] + dr, pac[1] + dc
        if _wall(grid, nr, nc):
            inefficient += 1
            nr, nc = pac
        if (nr, nc) == last_pos and action != "E":
            inefficient += 1
        last_pos = pac
        pac = (nr, nc)

        if pac in foods:
            foods.remove(pac)
            score += 10

        new_ghosts = []
        for gr, gc in ghosts:
            opts = []
            for d in DIRS:
                tr, tc = gr + MOVES[d][0], gc + MOVES[d][1]
                if not _wall(grid, tr, tc):
                    opts.append((tr, tc))
            if not opts:
                new_ghosts.append((gr, gc))
            else:
                new_ghosts.append(rng.choice(opts))
        ghosts = new_ghosts

        if pac in ghosts:
            early_death = True
            break

    steps_taken = step + 1
    survival = steps_taken if early_death else max(steps_taken, 1)

    return PacmanEvalResult(
        score=score,
        survival_steps=float(survival),
        total_steps=steps_taken,
        early_death=early_death,
        inefficient_moves=inefficient,
    )


def pacman_fitness(
    code: str,
    rng: random.Random,
    w1: float,
    w2: float,
    w3: float,
    runs: int = 3,
) -> tuple[float, dict[str, Any]]:
    wsum = w1 + w2 + w3
    if wsum <= 0:
        w1, w2, w3 = 0.5, 0.3, 0.2
        wsum = 1.0
    w1, w2, w3 = w1 / wsum, w2 / wsum, w3 / wsum

    try:
        agent = load_agent(code)
    except Exception as e:
        return -1e6, {"error": str(e)}

    scores: list[int] = []
    surv: list[int] = []
    steps: list[int] = []
    deaths = 0
    ineff = 0
    for _ in range(runs):
        r = run_episode(agent, rng)
        scores.append(r.score)
        surv.append(r.survival_steps)
        steps.append(r.total_steps)
        if r.early_death:
            deaths += 1
        ineff += r.inefficient_moves

    avg_score = sum(scores) / len(scores)
    avg_surv = sum(surv) / len(surv)
    avg_steps = sum(steps) / len(steps)
    base = w1 * avg_score + w2 * avg_surv - w3 * avg_steps
    penalty = 25.0 * deaths + 0.5 * ineff
    fitness = base - penalty
    metrics = {
        "avg_score": avg_score,
        "avg_survival": avg_surv,
        "avg_steps": avg_steps,
        "early_deaths": deaths,
        "inefficient_moves": ineff,
        "weights": {"w1": w1, "w2": w2, "w3": w3},
    }
    return fitness, metrics


def baseline_pacman_code() -> str:
    return '''# pacman-agent
def choose_action(state):
    px, py = state["pacman"]
    foods = state["foods"]
    if not foods:
        return "E"
    fx, fy = min(foods, key=lambda p: abs(p[0] - px) + abs(p[1] - py))
    if fx > px:
        return "E"
    if fx < px:
        return "W"
    if fy > py:
        return "S"
    if fy < py:
        return "N"
    return "E"
'''
