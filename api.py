from fastapi import FastAPI
from evolution import EvolutionEngine
from problems.pacman_problem import PacmanProblem
from problems.sorting_problem import SortingProblem

app = FastAPI()

PROBLEM_REGISTRY = {
    "pacman": PacmanProblem,
    "sorting": SortingProblem
}

@app.post("/solve/{problem_name}")
def solve(problem_name: str):

    if problem_name not in PROBLEM_REGISTRY:
        return {"error": "Unknown problem"}

    problem = PROBLEM_REGISTRY[problem_name](problem_name=problem_name)
    engine = EvolutionEngine(problem)

    best, history = engine.run()

    return {
        "best_solution": problem.serialize_solution(best),
        "fitness_history": history
    }