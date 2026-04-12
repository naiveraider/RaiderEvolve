from __future__ import annotations

import random

from evolve.models import CandidateRecord, SelectionMode


def _code_distance(a: str, b: str) -> int:
    return sum(1 for i, (ca, cb) in enumerate(zip(a, b)) if ca != cb) + abs(len(a) - len(b))


def select_population(
    candidates: list[CandidateRecord],
    mode: SelectionMode,
    top_k: int,
    rng: random.Random,
) -> list[CandidateRecord]:
    if not candidates:
        return []
    sorted_c = sorted(candidates, key=lambda x: x.fitness, reverse=True)

    if mode == SelectionMode.ELITE:
        k = max(1, min(top_k, len(sorted_c)))
        return sorted_c[:k]

    if mode == SelectionMode.TOP_K:
        k = max(1, min(top_k, len(sorted_c)))
        return sorted_c[:k]

    # Diversity-aware: greedily pick high fitness while keeping minimum string distance
    k = max(1, min(top_k, len(sorted_c)))
    chosen: list[CandidateRecord] = []
    min_dist = max(8, len(sorted_c[0].code) // 40)
    for c in sorted_c:
        if len(chosen) >= k:
            break
        if all(_code_distance(c.code, o.code) >= min_dist for o in chosen):
            chosen.append(c)
    for c in sorted_c:
        if len(chosen) >= k:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen[:k]


def tournament_pick(candidates: list[CandidateRecord], rng: random.Random, k: int = 3) -> CandidateRecord:
    sample = rng.sample(candidates, min(k, len(candidates)))
    return max(sample, key=lambda x: x.fitness)
