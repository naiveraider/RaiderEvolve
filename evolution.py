import random

from config import (
    CANDIDATE_GENERATION,
    GENERATIONS,
    HYBRID_STRATEGIES,
    POP_SIZE,
    SELECTION_MODE,
    TOP_K,
)


class EvolutionEngine:
    """
    Evolutionary engine with configurable:
    - Selection: single best or top-k.
    - Candidate generation: perturb, templates, swap, human, llm, or hybrid.
    """

    def __init__(
        self,
        problem,
        pop_size=None,
        top_k=None,
        generations=None,
        selection_mode=None,
        candidate_generation=None,
        hybrid_strategies=None,
        human_hook=None,
    ):
        self.problem = problem
        self.pop_size = pop_size if pop_size is not None else POP_SIZE
        self.top_k = top_k if top_k is not None else TOP_K
        self.generations = generations if generations is not None else GENERATIONS
        self.selection_mode = selection_mode or SELECTION_MODE
        self.candidate_generation = candidate_generation or CANDIDATE_GENERATION
        self.hybrid_strategies = hybrid_strategies or HYBRID_STRATEGIES
        self.human_hook = human_hook  # optional human-in-the-loop callback: fn(candidate) -> candidate
        self.population = []

    def _pick_strategy(self):
        if self.candidate_generation != "hybrid":
            return self.candidate_generation
        r = random.random()
        for strategy, prob in self.hybrid_strategies.items():
            r -= prob
            if r <= 0:
                return strategy
        return list(self.hybrid_strategies.keys())[0]

    def _generate_child(self, parent):
        strategy = self._pick_strategy()
        if strategy == "perturb":
            child = self.problem.mutate_perturb(parent)
        elif strategy == "templates":
            child = self.problem.mutate_from_template(parent)
        elif strategy == "swap":
            child = self.problem.mutate_swap(parent)
        elif strategy == "human":
            child = self._human_modify(parent)
        elif strategy == "llm":
            child = self.problem.improve_with_llm(parent)
        else:
            child = self.problem.mutate(parent)
        return child

    def _human_modify(self, candidate):
        if self.human_hook:
            return self.human_hook(candidate)
        import copy
        return copy.deepcopy(candidate)

    def run(self):
        self.population = self.problem.initialize_population(self.pop_size)
        history = []

        for gen in range(self.generations):
            for c in self.population:
                self.problem.evaluate(c)

            self.population.sort(key=lambda x: (x["fitness"] or -1e9), reverse=True)

            if self.selection_mode == "best":
                self.population = self.population[:1]
            else:
                self.population = self.population[: self.top_k]

            best = self.population[0]
            history.append(best["fitness"])

            new_population = []
            while len(new_population) < self.pop_size:
                parent = random.choice(self.population)
                child = self._generate_child(parent)
                self.problem.evaluate(child)
                new_population.append(child)

            self.population = new_population

        best = max(self.population, key=lambda x: (x["fitness"] or -1e9))
        return best, history