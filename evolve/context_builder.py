from __future__ import annotations

from evolve.memory_store import MemoryStore
from evolve.models import CandidateRecord, TaskType


def build_llm_context(
    task: TaskType,
    population: list[CandidateRecord],
    memory: MemoryStore,
    generation: int,
    max_generations: int,
) -> str:
    best_hist = memory.best_n(5)
    worst_hist = memory.worst_n(3)
    prev_best = max((r.fitness for r in memory.records if r.generation < generation), default=None)
    cur_best = max((r.fitness for r in population), default=None)

    lines: list[str] = []
    lines.append(f"Generation {generation + 1} / {max_generations}.")
    if cur_best is not None:
        lines.append(f"Current population best fitness: {cur_best:.4f}")
    if prev_best is not None:
        lines.append(f"Previous generation best (historical): {prev_best:.4f}")

    lines.append("\nTop historical solutions:")
    for i, r in enumerate(best_hist[:5], 1):
        lines.append(f"{i}. fitness={r.fitness:.4f} gen={r.generation}")
        lines.append("CODE:\n" + r.code[:6000])

    lines.append("\nLow performers (avoid repeating mistakes):")
    for i, r in enumerate(worst_hist[:2], 1):
        lines.append(f"{i}. fitness={r.fitness:.4f}")
        lines.append(r.code[:2000])

    if task == TaskType.PACMAN:
        lines.append(
            "\nTASK: Improve Python Pacman agent. Output ONLY python code.\n"
            "Contract: define choose_action(state) returning one of 'N','S','E','W'.\n"
            "State keys: pacman (r,c), foods list, ghosts list, nearest_food, score, step."
        )
    else:
        lines.append(
            "\nTASK: Improve 3x3 matrix multiply. Output ONLY python code.\n"
            "Contract: def matmul(a,b) with a,b as 3x3 nested lists, return 3x3 list.\n"
            "Minimize scalar multiplications and additions while staying correct."
        )
    return "\n".join(lines)


SYSTEM_PROMPT = "You are an expert code optimizer. Respond with executable Python only, no markdown prose."
