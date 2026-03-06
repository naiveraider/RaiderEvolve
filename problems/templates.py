"""
Predefined code/parameter templates for template-based candidate generation.
Maps template names to parameter sets or code fragments per problem.
"""

# Pacman: algorithm + heuristic combinations as "templates"
PACMAN_TEMPLATES = [
    {"algorithm": "breadthFirstSearch", "heuristic": "nullHeuristic"},
    {"algorithm": "depthFirstSearch", "heuristic": "nullHeuristic"},
    {"algorithm": "uniformCostSearch", "heuristic": "nullHeuristic"},
    {"algorithm": "astar", "heuristic": "nullHeuristic"},
    {"algorithm": "astar", "heuristic": "manhattanHeuristic"},
]

# Sorting: pivot strategy as template
SORTING_TEMPLATES = [
    "first",
    "random",
    "middle",
    "median_of_three",
]
