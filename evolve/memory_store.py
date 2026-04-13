from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
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

    Semantic retrieval (best_n with query_code):
      When Qdrant is configured, best_n() uses vector similarity to find
      candidates that are semantically close to the current population's best
      code AND have high fitness, giving the LLM more relevant context.

      Falls back to linear fitness-ranked retrieval if Qdrant is unavailable
      or no query_code is provided.
    """

    def __init__(self) -> None:
        self.records: list[CandidateRecord] = []
        self._fitness_cache: dict[str, FitnessCacheEntry] = {}
        self._seen_code: set[str] = set()
        self._run_id = str(uuid.uuid4())

        # Lazy-init Qdrant — only imported and connected if QDRANT_URL is set
        from evolve.qdrant_store import QdrantMemory
        self._qdrant = QdrantMemory(self._run_id)

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

        # Async-safe: upsert is a fire-and-soft-fail operation
        self._qdrant.upsert(rec.id, code, fitness, generation)
        return rec

    # ── retrieval ───────────────────────────────────────────────────────────

    def best_n(self, n: int, query_code: Optional[str] = None) -> list[CandidateRecord]:
        """
        Return the top-n candidates by fitness.

        If ``query_code`` is provided and Qdrant is available, uses semantic
        similarity to prefer candidates that are structurally similar to the
        current query (current population's best solution), then ranks by
        fitness among the retrieved set.

        Falls back to linear fitness ranking when:
          - query_code is None
          - Qdrant is not configured / unreachable
          - Semantic search returns fewer results than expected
        """
        if query_code and self._qdrant.available:
            hits = self._qdrant.search(query_code, n)
            if hits:
                id_set = {rec_id for rec_id, _ in hits}
                semantic_records = [r for r in self.records if r.id in id_set]
                if semantic_records:
                    return sorted(semantic_records, key=lambda r: r.fitness, reverse=True)[:n]

        # Linear fallback
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
