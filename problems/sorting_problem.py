import copy
import random

from .base import BaseProblem
from .templates import SORTING_TEMPLATES


class SortingProblem(BaseProblem):
    """Sorting (e.g. quicksort) evolution: perturb pivot selection, templates, swap."""

    def __init__(self, problem_name="sorting"):
        self.problem_name = problem_name
        try:
            from llm_client import LLMClient
            self.llm = LLMClient()
        except Exception:
            self.llm = None

    def improve_with_llm(self, candidate):
        if self.llm:
            return self.llm.improve(self.problem_name, candidate)
        return copy.deepcopy(candidate)

    # Cost multipliers per pivot strategy (simulated)
    _COST = {"first": 1.2, "random": 1.0, "middle": 0.95, "median_of_three": 0.9}

    def initialize_population(self, pop_size):
        population = []
        for _ in range(pop_size):
            population.append({
                "pivot_strategy": random.choice(["first", "random"]),
                "fitness": None,
            })
        return population

    def mutate(self, candidate):
        child = copy.deepcopy(candidate)
        child["pivot_strategy"] = random.choice(["first", "random", "middle", "median_of_three"])
        return child

    def mutate_perturb(self, candidate):
        child = copy.deepcopy(candidate)
        child["pivot_strategy"] = random.choice(SORTING_TEMPLATES)
        return child

    def get_templates(self):
        return list(SORTING_TEMPLATES)

    def _apply_template(self, candidate, template_name):
        name = template_name if isinstance(template_name, str) else SORTING_TEMPLATES[template_name % len(SORTING_TEMPLATES)]
        candidate["pivot_strategy"] = name

    def mutate_from_template(self, candidate):
        child = copy.deepcopy(candidate)
        name = random.choice(SORTING_TEMPLATES)
        self._apply_template(child, name)
        return child

    def mutate_swap(self, candidate):
        """Swap to next/previous pivot strategy in list (swap two 'lines' of config)."""
        child = copy.deepcopy(candidate)
        idx = SORTING_TEMPLATES.index(child["pivot_strategy"]) if child["pivot_strategy"] in SORTING_TEMPLATES else 0
        child["pivot_strategy"] = SORTING_TEMPLATES[(idx + 1) % len(SORTING_TEMPLATES)]
        return child

    def evaluate(self, candidate):
        data = [random.randint(0, 1000) for _ in range(1000)]
        strat = candidate.get("pivot_strategy", "first")
        mult = self._COST.get(strat, 1.2)
        cost = self.simulate_cost(data, mult)
        candidate["fitness"] = -cost
        return -cost

    def simulate_cost(self, data, multiplier):
        return len(data) * multiplier

    def serialize_solution(self, candidate):
        return dict(candidate)