import copy
import random
import re
import subprocess

from config import LLM_MUTATION_RATIO, PACMAN_FITNESS_WEIGHTS
from llm_client import LLMClient
from .base import BaseProblem
from .templates import PACMAN_TEMPLATES


class PacmanProblem(BaseProblem):
    """Pacman search agent evolution. Fitness = w1*score + w2*survival_time - w3*cost(steps)."""

    def __init__(self, problem_name="pacman"):
        self.problem_name = problem_name
        self.llm = LLMClient()
        self._weights = PACMAN_FITNESS_WEIGHTS  # (w1, w2, w3)

    def improve_with_llm(self, candidate):
        return self.llm.improve(self.problem_name, candidate)

    def initialize_population(self, pop_size):
        population = []
        for _ in range(pop_size):
            population.append({
                "algorithm": "breadthFirstSearch",
                "heuristic": "nullHeuristic",
                "fitness": None,
            })
        return population

    def mutate(self, candidate):
        child = copy.deepcopy(candidate)
        if random.random() < 0.5:
            child["algorithm"] = random.choice([
                "breadthFirstSearch", "depthFirstSearch",
                "uniformCostSearch", "astar",
            ])
        else:
            child["heuristic"] = random.choice([
                "nullHeuristic", "manhattanHeuristic",
            ])
        return child

    def mutate_perturb(self, candidate):
        return self.mutate(candidate)

    def get_templates(self):
        return list(range(len(PACMAN_TEMPLATES)))

    def _apply_template(self, candidate, template_name):
        idx = template_name if isinstance(template_name, int) else int(template_name)
        t = PACMAN_TEMPLATES[idx % len(PACMAN_TEMPLATES)]
        candidate["algorithm"] = t["algorithm"]
        candidate["heuristic"] = t["heuristic"]

    def mutate_from_template(self, candidate):
        child = copy.deepcopy(candidate)
        idx = random.randrange(len(PACMAN_TEMPLATES))
        self._apply_template(child, idx)
        return child

    def mutate_swap(self, candidate):
        """Swap algorithm and heuristic with another valid choice (swap two 'lines' of config)."""
        child = copy.deepcopy(candidate)
        algorithms = ["breadthFirstSearch", "depthFirstSearch", "uniformCostSearch", "astar"]
        heuristics = ["nullHeuristic", "manhattanHeuristic"]
        if random.random() < 0.5:
            i = algorithms.index(child["algorithm"])
            child["algorithm"] = algorithms[(i + 1) % len(algorithms)]
        else:
            i = heuristics.index(child["heuristic"])
            child["heuristic"] = heuristics[(i + 1) % len(heuristics)]
        return child

    def evaluate(self, candidate):
        agent_args = f"fn={candidate['algorithm']}"
        if candidate.get("algorithm") == "astar":
            agent_args += f",heuristic={candidate.get('heuristic', 'nullHeuristic')}"
        cmd = [
            "python", "pacman.py",
            "-l", "mediumMaze",
            "-p", "SearchAgent",
            "-a", agent_args,
            "-q",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout or result.stderr or ""

        score_match = re.search(r"Score:\s*(-?\d+)", output)
        score = int(score_match.group(1)) if score_match else 0

        steps_match = re.search(r"(?:Total\s+)?(?:moves|steps?):\s*(\d+)", output, re.I)
        steps = int(steps_match.group(1)) if steps_match else 0
        survival_time = steps  # or time-to-win; use steps as proxy if no separate field

        w1, w2, w3 = self._weights
        # Normalize so cost doesn't dominate: use steps/1000 or similar
        cost = steps
        fitness = w1 * score + w2 * survival_time - w3 * (cost / 1000.0)
        candidate["fitness"] = fitness
        candidate["_score"] = score
        candidate["_steps"] = steps
        candidate["_survival_time"] = survival_time
        return fitness

    def serialize_solution(self, candidate):
        return {
            "algorithm": candidate["algorithm"],
            "heuristic": candidate["heuristic"],
            "fitness": candidate["fitness"],
            "score": candidate.get("_score"),
            "steps": candidate.get("_steps"),
        }