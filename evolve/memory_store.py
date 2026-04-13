from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from evolve.models import CandidateRecord


def code_hash(code: str) -> str:
    return hashlib.sha256(code.strip().encode()).hexdigest()


@dataclass
class FitnessCacheEntry:
    fitness: float
    metrics: dict[str, Any]


class MemoryStore:
    """
    Stores all candidates, parent links, and deduplicated fitness cache.

    Retrieval uses linear fitness ranking (unchanged from the original design).

    Qdrant integration: every added candidate is persisted to Qdrant in the
    background (fire-and-forget). This does NOT change any retrieval logic —
    best_n / worst_n still sort the in-memory list by fitness.
    """

    def __init__(self, task: str = "", run_id: Optional[str] = None) -> None:
        self.records: list[CandidateRecord] = []
        self._fitness_cache: dict[str, FitnessCacheEntry] = {}
        self._seen_code: set[str] = set()
        self._task = task
        self._run_id = run_id or str(uuid.uuid4())

    # ── code deduplication ──────────────────────────────────────────────────

    def remember_code(self, code: str) -> bool:
        h = code_hash(code)
        if h in self._seen_code:
            return False
        self._seen_code.add(h)
        return True

    # ── fitness cache ───────────────────────────────────────────────────────

    def get_cached_fitness(self, code: str) -> Optional[FitnessCacheEntry]:
        return self._fitness_cache.get(code_hash(code))

    def put_cached_fitness(self, code: str, fitness: float, metrics: dict[str, Any]) -> None:
        self._fitness_cache[code_hash(code)] = FitnessCacheEntry(fitness=fitness, metrics=metrics)

    # ── add candidate ───────────────────────────────────────────────────────

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

        # Persist to Qdrant in background — fire-and-forget, never blocks.
        from evolve.qdrant_store import qdrant_logger
        qdrant_logger.log(
            record_id=rec.id,
            run_id=self._run_id,
            task=self._task,
            generation=generation,
            fitness=fitness,
            strategy_tag=strategy_tag,
            mutation_notes=mutation_notes,
            code=code,
        )
        return rec

    # ── retrieval (linear fitness ranking — unchanged) ──────────────────────

    def best_n(self, n: int) -> list[CandidateRecord]:
        return sorted(self.records, key=lambda r: r.fitness, reverse=True)[:n]

    def worst_n(self, n: int) -> list[CandidateRecord]:
        return sorted(self.records, key=lambda r: r.fitness)[:n]

    def by_generation(self, gen: int) -> list[CandidateRecord]:
        return [r for r in self.records if r.generation == gen]

    def best_up_to_generation(self, max_gen: int) -> Optional[float]:
        subset = [r for r in self.records if r.generation <= max_gen]
        if not subset:
            return None
        return max(r.fitness for r in subset)
