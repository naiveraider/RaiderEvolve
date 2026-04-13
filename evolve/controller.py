from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from evolve.context_builder import SYSTEM_PROMPT, build_llm_context
from evolve.llm_client import improve_code_sync
from evolve.matrix_task import baseline_matrix_code, matrix_correctness_and_ops
from evolve.memory_store import MemoryStore, code_hash
from evolve.models import (
    CandidateRecord,
    EvolutionStrategy,
    FitnessPreset,
    GenerationLog,
    SelectionMode,
    StrategyRunResult,
    TaskType,
)
from evolve.pacman_env import baseline_pacman_code, pacman_fitness
from evolve.random_mutation import random_mutate
from evolve.selection import select_population
from evolve.template_mutation import template_mutate

ProgressCallback = Optional[Callable[[dict], None]]


@dataclass
class EvalConfig:
    task: TaskType
    fitness_preset: FitnessPreset
    w1: float
    w2: float
    w3: float
    matrix_alpha: float
    matrix_beta: float


def _evaluate(
    code: str,
    memory: MemoryStore,
    cfg: EvalConfig,
    rng: random.Random,
) -> tuple[float, dict, bool]:
    cached = memory.get_cached_fitness(code)
    if cached:
        return cached.fitness, cached.metrics, True
    if cfg.task == TaskType.PACMAN:
        fit, meta = pacman_fitness(code, rng, cfg.w1, cfg.w2, cfg.w3)
    else:
        fit, meta = matrix_correctness_and_ops(code, cfg.matrix_alpha, cfg.matrix_beta)
    memory.put_cached_fitness(code, fit, meta)
    return fit, meta, False


def _initial_code(task: TaskType, user_code: str) -> str:
    if user_code.strip():
        return user_code.strip()
    return baseline_pacman_code() if task == TaskType.PACMAN else baseline_matrix_code()


def _weights_from_preset(req) -> tuple[float, float, float]:
    if req.fitness_preset == FitnessPreset.PACMAN:
        return 0.5, 0.3, 0.2
    if req.fitness_preset == FitnessPreset.MATRIX:
        return 1.0, 0.0, 0.0
    cw = req.custom_weights
    if cw is None:
        return 0.5, 0.3, 0.2
    s = cw.w1 + cw.w2 + cw.w3
    if s <= 0:
        return 0.5, 0.3, 0.2
    return cw.w1 / s, cw.w2 / s, cw.w3 / s


def run_evolution_run(
    req,
    strategy: EvolutionStrategy,
    progress_cb: ProgressCallback = None,
) -> StrategyRunResult:
    rng = random.Random(req.seed if req.seed is not None else uuid.uuid4().int % (2**32))
    memory = MemoryStore(task=req.task.value)
    w1, w2, w3 = _weights_from_preset(req)
    cfg = EvalConfig(
        task=req.task,
        fitness_preset=req.fitness_preset,
        w1=w1,
        w2=w2,
        w3=w3,
        matrix_alpha=req.matrix_alpha,
        matrix_beta=req.matrix_beta,
    )

    seed_code = _initial_code(req.task, req.source_code)
    population: list[CandidateRecord] = []

    def eval_and_store(
        gen: int,
        code: str,
        parents: list[str],
        tag: str,
        note: str,
    ) -> CandidateRecord:
        fit, meta, _reuse = _evaluate(code, memory, cfg, rng)
        return memory.add(gen, code, fit, parents, tag, note, meta)

    # Generation 0: evaluate seed
    root = eval_and_store(0, seed_code, [], "seed", "baseline")
    population: list[CandidateRecord] = [root]
    if strategy != EvolutionStrategy.SINGLE_LLM and req.population_size > 1:
        seen: set[str] = {code_hash(seed_code)}

        for _ in range(req.population_size - 1):
            code_v, note_v = random_mutate(seed_code, rng)
            h = code_hash(code_v)
            if h in seen:
                continue
            seen.add(h)
            population.append(eval_and_store(0, code_v, [root.id], "bootstrap", note_v))
        population = select_population(
            population, req.selection_mode, min(req.top_k, len(population)), rng
        )

    fitness_curve: list[float] = []
    avg_per_gen: list[float] = []
    best_per_gen: list[float] = []
    history: list[GenerationLog] = []

    if strategy == EvolutionStrategy.SINGLE_LLM:
        if progress_cb:
            progress_cb({"gen": 0, "total": 1, "strategy": strategy.value, "status": "llm_call"})
        ctx = build_llm_context(req.task, population, memory, 0, 1, single_shot=True)
        child_code = improve_code_sync(SYSTEM_PROMPT, ctx)
        child = eval_and_store(0, child_code, [root.id], "single_llm", "one-shot LLM refinement")
        population = [max([root, child], key=lambda x: x.fitness)]
        fitness_curve.append(population[0].fitness)
        avg_per_gen.append(population[0].fitness)
        best_per_gen.append(population[0].fitness)
        history.append(
            GenerationLog(
                generation=0,
                selection_summary="single LLM output vs seed",
                mutation_explanations=["LLM one-shot"],
                best_id=population[0].id,
                avg_fitness=population[0].fitness,
                best_fitness=population[0].fitness,
            )
        )
        if progress_cb:
            progress_cb({"gen": 1, "total": 1, "strategy": strategy.value,
                         "best": population[0].fitness, "avg": population[0].fitness, "status": "done"})
        return StrategyRunResult(
            strategy=strategy,
            best_code=population[0].code,
            fitness_curve=fitness_curve,
            avg_fitness_per_gen=avg_per_gen,
            best_per_generation=best_per_gen,
            history=history,
            memory_records=list(memory.records),
            final_best_fitness=population[0].fitness,
        )

    best0 = max(p.fitness for p in population)
    valid0 = [p for p in population if p.fitness > -999]
    avg0 = sum(p.fitness for p in valid0) / len(valid0) if valid0 else best0
    best_id0 = max(population, key=lambda x: x.fitness).id
    fitness_curve.append(best0)
    avg_per_gen.append(avg0)
    best_per_gen.append(best0)
    history.append(
        GenerationLog(
            generation=0,
            selection_summary="initial population (seed + optional bootstrap mutants)",
            mutation_explanations=["baseline evaluated"],
            best_id=best_id0,
            avg_fitness=avg0,
            best_fitness=best0,
        )
    )

    for gen in range(req.generations):
        if progress_cb:
            progress_cb({"gen": gen + 1, "total": req.generations,
                         "strategy": strategy.value, "status": "running"})

        # 1) Selection
        parents = select_population(population, req.selection_mode, min(req.top_k, len(population)), rng)
        sel_summary = f"kept {len(parents)} parents via {req.selection_mode.value}"

        candidates: list[tuple[str, str, str, list[str]]] = []
        mutations_log: list[str] = []

        # LLM call: once per generation (best parent only) — not once per parent
        if strategy == EvolutionStrategy.FULL:
            best_parent = max(parents, key=lambda x: x.fitness)
            ctx = build_llm_context(req.task, parents, memory, gen, req.generations)
            try:
                parent_cap = 1500 if cfg.task == TaskType.MATRIX else 3000
                llm_code = improve_code_sync(
                    SYSTEM_PROMPT,
                    ctx + f"\nPARENT_CODE:\n{best_parent.code[:parent_cap]}",
                )
                mutations_log.append("LLM refinement (best parent)")
                candidates.append((llm_code, "llm", "llm_mutation", [best_parent.id]))
            except Exception as e:
                mutations_log.append(f"LLM failed: {e}")

        for p in parents:
            # 3–4) Random / template mutations
            if strategy == EvolutionStrategy.RANDOM_ONLY:
                code_r, note_r = random_mutate(p.code, rng)
                candidates.append((code_r, "random", note_r, [p.id]))
                code_t, note_t = template_mutate(p.code, req.task.value, rng)
                candidates.append((code_t, "template", note_t, [p.id]))
            else:
                code_r, note_r = random_mutate(p.code, rng)
                candidates.append((code_r, "hybrid_random", note_r, [p.id]))
                code_t, note_t = template_mutate(p.code, req.task.value, rng)
                candidates.append((code_t, "hybrid_template", note_t, [p.id]))

        # include elites unchanged
        for p in parents[:2]:
            candidates.append((p.code, "elite_copy", "elitism", [p.id]))

        scored: list[CandidateRecord] = []
        seen_hashes: set[str] = set()
        for code, tag, note, par in candidates:
            h = code_hash(code)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            rec = eval_and_store(gen + 1, code, par, tag, note)
            scored.append(rec)

        if not scored:
            scored = parents

        # 6) Ranking
        scored.sort(key=lambda x: x.fitness, reverse=True)
        best = scored[0]
        # Exclude evaluation failures (fitness <= -999) from the average so that
        # LLM-generated code with syntax errors doesn't distort the avg metric.
        valid = [s for s in scored if s.fitness > -999]
        avg_f = sum(s.fitness for s in valid) / len(valid) if valid else scored[0].fitness

        avg_per_gen.append(avg_f)
        best_per_gen.append(best.fitness)
        fitness_curve.append(best.fitness)

        history.append(
            GenerationLog(
                generation=gen + 1,
                selection_summary=sel_summary,
                mutation_explanations=mutations_log or [c.mutation_notes for c in scored[:3]],
                best_id=best.id,
                avg_fitness=avg_f,
                best_fitness=best.fitness,
            )
        )

        if progress_cb:
            progress_cb({
                "gen": gen + 1,
                "total": req.generations,
                "strategy": strategy.value,
                "best": best.fitness,
                "avg": avg_f,
                "status": "gen_done",
            })

        # 7) Memory already updated via eval_and_store
        # 8) Next generation population
        k = max(1, min(req.top_k, len(scored)))
        population = scored[:k]

    final_best = memory.best_n(1)[0] if memory.records else population[0]
    if progress_cb:
        progress_cb({
            "gen": req.generations,
            "total": req.generations,
            "strategy": strategy.value,
            "best": final_best.fitness,
            "status": "done",
        })
    return StrategyRunResult(
        strategy=strategy,
        best_code=final_best.code,
        fitness_curve=fitness_curve,
        avg_fitness_per_gen=avg_per_gen,
        best_per_generation=best_per_gen,
        history=history,
        memory_records=list(memory.records),
        final_best_fitness=final_best.fitness,
    )


PSEUDOCODE_OUTLINE = """Algorithm:
1. Initialize population from user/baseline code
2. Evaluate fitness (with deduplication cache)
3. Select parents (top-k / elite / diversity)
4. Build LLM context from memory (best, worst, trends)
5. Generate candidates: LLM mutation + random + template (+ elites)
6. Rank by fitness
7. Update memory with parent-child links
8. Repeat for N generations
"""
