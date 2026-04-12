from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any

from evolve.models import CandidateRecord


def code_hash(code: str) -> str:
    return hashlib.sha256(code.strip().encode()).hexdigest()


@dataclass
class FitnessCacheEntry:
    fitness: float
    metrics: dict[str, Any]


class MemoryStore:
    """Stores all candidates, parent links, and deduplicated fitness cache."""

    def __init__(self) -> None:
        self.records: list[CandidateRecord] = []
        self._fitness_cache: dict[str, FitnessCacheEntry] = {}
        self._seen_code: set[str] = set()

    def remember_code(self, code: str) -> bool:
        h = code_hash(code)
        if h in self._seen_code:
            return False
        self._seen_code.add(h)
        return True

    def get_cached_fitness(self, code: str) -> FitnessCacheEntry | None:
        return self._fitness_cache.get(code_hash(code))

    def put_cached_fitness(self, code: str, fitness: float, metrics: dict[str, Any]) -> None:
        self._fitness_cache[code_hash(code)] = FitnessCacheEntry(fitness=fitness, metrics=metrics)

    def add(
        self,
        generation: int,
        code: str,
        fitness: float,
        parents: list[str],
        strategy_tag: str,
        mutation_notes: str,
        metrics: dict[str, Any],
    ) -> CandidateRecord:
        rec = CandidateRecord(
            id=str(uuid.uuid4()),
            generation=generation,
            code=code,
            fitness=fitness,
            parents=parents,
            strategy_tag=strategy_tag,
            mutation_notes=mutation_notes,
            metrics=metrics,
        )
        self.records.append(rec)
        self.put_cached_fitness(code, fitness, metrics)
        return rec

    def best_n(self, n: int) -> list[CandidateRecord]:
        ranked = sorted(self.records, key=lambda r: r.fitness, reverse=True)
        return ranked[:n]

    def worst_n(self, n: int) -> list[CandidateRecord]:
        ranked = sorted(self.records, key=lambda r: r.fitness)
        return ranked[:n]

    def by_generation(self, gen: int) -> list[CandidateRecord]:
        return [r for r in self.records if r.generation == gen]

    def best_up_to_generation(self, max_gen: int) -> float | None:
        subset = [r for r in self.records if r.generation <= max_gen]
        if not subset:
            return None
        return max(r.fitness for r in subset)
