#!/usr/bin/env python3
"""
Minimal UC Berkeley Pacman-style runner for evolution.
Usage: python pacman.py -l <layout> -p SearchAgent -a fn=<algorithm>[,heuristic=<name>] [-q]
Outputs: Score: <int> and Total moves: <int> for fitness evaluation.
"""
from __future__ import print_function
import argparse
import collections
import heapq
import math
import sys

# --------------- Layouts ---------------

TINY_MAZE = """\
%%%%
%.P%
%..%
%%%%"""

SMALL_MAZE = """\
%%%%%%%%%
%.......%
%.%%.%%.%
%.%....P%
%.%%.%%.%
%.......%
%%%%%%%%%"""

MEDIUM_MAZE = """\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%.....%.....%.....%.....%...%
%.%%%.%.%%%.%.%%%.%.%%%.%.%.%
%.....%.....%.....%.....%...%
%.%%%.%.%%%.%.%%%.%.%%%.%.%.%
%.....%.....P.....%.....%...%
%.%%%.%.%%%.%.%%%.%.%%%.%.%.%
%.....%.....%.....%.....%...%
%.%%%.%.%%%.%.%%%.%.%%%.%.%.%
%.....%.....%.....%.....%...%
%.%%%.%.%%%.%.%%%.%.%%%.%.%.%
%.....%.....%.....%.....%...%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

OPEN_MAZE = """\
%%%%%%%%%%%%%%%%%%
%................%
%................%
%................%
%........P.......%
%................%
%................%
%................%
%%%%%%%%%%%%%%%%%%"""

TRICKY_MAZE = """\
%%%%%%%%%%%%%%%%%%%%%
%...%.......%.......%
%.%.%.%.%%%.%.%%%.%.%
%...%...%...%.%...%.%
%%%.%%%.%.%%%.%.%%%.%
%...........P.......%
%.%%%.%%%.%.%%%.%%%.%
%...%.%...%.....%...%
%.%.%.%.%%%.%%%.%.%.%
%...%.......%.......%
%%%%%%%%%%%%%%%%%%%%%"""

# Weighted mazes: ' '=cost-1 passage, 'M'=mud(cost 5), '%'=wall, 'P'=start, 'G'=goal.
#
# Design guarantee for a measurable BFS < UCS gradient:
#   Direct path   : fewer hops BUT goes through mud  → BFS picks this, high cost
#   Detour path   : more hops  AND avoids mud entirely → UCS/A* picks this, low cost
#
# BFS (unweighted): picks fewest-hop path → muddy shortcut → low score
# UCS / A*        : picks cheapest-cost path → clean detour → high score

# Direct (9 hops, 7 mud cells, cost≈38); clean detour (20 hops, cost=21)
MUD_MAZE = """\
%%%%%%%%%%
%G       %
%MMMMM   %
%MMMMM   %
%MMMMM   %
%MMMMM   %
%MMMMM   %
%MMMMM   %
%MMMMM   %
%        %
%P       %
%%%%%%%%%%"""

# Larger mud field; direct (11 hops, 9 mud, cost≈48); detour (26 hops, cost=27)
MUD_MAZE2 = """\
%%%%%%%%%%%%%
%G          %
%MMMMMMM    %
%MMMMMMM    %
%MMMMMMM    %
%MMMMMMM    %
%MMMMMMM    %
%MMMMMMM    %
%MMMMMMM    %
%MMMMMMM    %
%MMMMMMM    %
%           %
%P          %
%%%%%%%%%%%%%"""

# Efficiency maze: goal at the far-right of the TOP corridor; a large open room
# hangs below.  Every step downward is equally cheap (uniform cost = 1).
#
# UCS (cost-only): fans out uniformly — floods the entire room as fast as it
#   travels right along the corridor  → visits most of the maze's ~700 cells.
# A* (cost + Manhattan heuristic): row 0 cells have the lowest h(n) = |c-goal_c|
#   so A* stays in the corridor, ignoring the room below → visits ~50-80 cells.
#
# The mud strip partway along the corridor forces cost-awareness (BFS takes the
# muddy shortcut; UCS/A* go around) — keeps the BFS < UCS gradient alive.
LARGE_MUD_MAZE = """\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%P  MMMMMMMMM                            G  %
%                                           %
%                                           %
%                                           %
%                                           %
%                                           %
%                                           %
%                                           %
%                                           %
%                                           %
%                                           %
%                                           %
%                                           %
%                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

LAYOUTS = {
    "tinyMaze":     TINY_MAZE,
    "smallMaze":    SMALL_MAZE,
    "mediumMaze":   MEDIUM_MAZE,
    "openMaze":     OPEN_MAZE,
    "trickyMaze":   TRICKY_MAZE,
    "mudMaze":      MUD_MAZE,
    "mudMaze2":     MUD_MAZE2,
    "largeMudMaze": LARGE_MUD_MAZE,
}


def parse_layout(name):
    raw = LAYOUTS.get(name, LAYOUTS["mediumMaze"])
    grid = [list(line) for line in raw.split("\n") if line.strip()]
    if not grid:
        return [["%", " ", "%"], ["%", "P", "%"], ["%", "%", "%"]]
    max_w = max(len(row) for row in grid)
    for row in grid:
        row.extend(["%"] * (max_w - len(row)))
    return grid


def _reachable_from(grid, start):
    from collections import deque
    reachable = {start}
    q = deque([start])
    h, w = len(grid), len(grid[0]) if grid else 0
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != "%" and (nr, nc) not in reachable:
                reachable.add((nr, nc))
                q.append((nr, nc))
    return reachable


def find_positions(grid):
    pacman = None
    dots = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "P":
                pacman = (r, c)
                row[c] = " "
            elif cell == ".":
                dots.append((r, c))
    if pacman is None:
        pacman = (1, 1)
    reachable = _reachable_from(grid, pacman)
    goal = None
    for d in dots:
        if d in reachable:
            goal = d
            break
    if goal is None and dots:
        goal = dots[0]
    if goal is None:
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell == " " and (r, c) != pacman and (r, c) in reachable:
                    goal = (r, c)
                    break
            if goal:
                break
    return pacman, goal or pacman, grid


def neighbors(pos, grid):
    r, c = pos
    h = len(grid)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < len(grid[nr]) and grid[nr][nc] != "%":
            yield (nr, nc), 1


# --------------- Heuristics ---------------

def null_heuristic(state, goal):
    return 0


def manhattan_heuristic(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def euclidean_heuristic(state, goal):
    return math.sqrt((state[0] - goal[0]) ** 2 + (state[1] - goal[1]) ** 2)


def chebyshev_heuristic(state, goal):
    return max(abs(state[0] - goal[0]), abs(state[1] - goal[1]))


HEURISTICS = {
    "nullHeuristic": null_heuristic,
    "manhattanHeuristic": manhattan_heuristic,
    "euclideanHeuristic": euclidean_heuristic,
    "chebyshevHeuristic": chebyshev_heuristic,
}


# --------------- Search algorithms ---------------

def breadth_first_search(start, goal, grid):
    fringe = collections.deque([(start, [start])])
    seen = {start}
    while fringe:
        state, path = fringe.popleft()
        if state == goal:
            return path
        for nxt, _ in neighbors(state, grid):
            if nxt not in seen:
                seen.add(nxt)
                fringe.append((nxt, path + [nxt]))
    return [start]


def depth_first_search(start, goal, grid):
    fringe = [(start, [start])]
    seen = {start}
    while fringe:
        state, path = fringe.pop()
        if state == goal:
            return path
        for nxt, _ in neighbors(state, grid):
            if nxt not in seen:
                seen.add(nxt)
                fringe.append((nxt, path + [nxt]))
    return [start]


def uniform_cost_search(start, goal, grid):
    fringe = [(0, start, [start])]
    seen = {start: 0}
    while fringe:
        cost, state, path = heapq.heappop(fringe)
        if state == goal:
            return path
        for nxt, step_cost in neighbors(state, grid):
            new_cost = cost + step_cost
            if nxt not in seen or seen[nxt] > new_cost:
                seen[nxt] = new_cost
                heapq.heappush(fringe, (new_cost, nxt, path + [nxt]))
    return [start]


def astar_search(start, goal, grid, heuristic_name="nullHeuristic"):
    h_fn = HEURISTICS.get(heuristic_name, null_heuristic)

    def f(g, s):
        return g + h_fn(s, goal)

    fringe = [(f(0, start), 0, start, [start])]
    seen = {start: 0}
    while fringe:
        _, g, state, path = heapq.heappop(fringe)
        if state == goal:
            return path
        for nxt, step_cost in neighbors(state, grid):
            new_g = g + step_cost
            if nxt not in seen or seen[nxt] > new_g:
                seen[nxt] = new_g
                heapq.heappush(fringe, (f(new_g, nxt), new_g, nxt, path + [nxt]))
    return [start]


def greedy_best_first_search(start, goal, grid, heuristic_name="manhattanHeuristic"):
    h_fn = HEURISTICS.get(heuristic_name, manhattan_heuristic)
    fringe = [(h_fn(start, goal), start, [start])]
    seen = {start}
    while fringe:
        _, state, path = heapq.heappop(fringe)
        if state == goal:
            return path
        for nxt, _ in neighbors(state, grid):
            if nxt not in seen:
                seen.add(nxt)
                heapq.heappush(fringe, (h_fn(nxt, goal), nxt, path + [nxt]))
    return [start]


ALGORITHMS = {
    "breadthFirstSearch": lambda s, g, gr, **kw: breadth_first_search(s, g, gr),
    "depthFirstSearch": lambda s, g, gr, **kw: depth_first_search(s, g, gr),
    "uniformCostSearch": lambda s, g, gr, **kw: uniform_cost_search(s, g, gr),
    "astar": lambda s, g, gr, heuristic_name="nullHeuristic", **kw: astar_search(s, g, gr, heuristic_name),
    "greedyBestFirst": lambda s, g, gr, heuristic_name="manhattanHeuristic", **kw: greedy_best_first_search(s, g, gr, heuristic_name),
}


CELL_COST = {
    " ": 1,   # open passage
    ".": 1,   # food (treated as open for cost purposes)
    "M": 5,   # mud — expensive to traverse
    "P": 1,   # start cell
    "G": 1,   # goal cell
}


def parse_weighted_layout(name: str):
    """
    Parse a layout that may contain 'M' (mud) cells.
    Returns (start, goal, grid) where grid keeps 'M' so agents can see it.
    'G' is replaced with ' ' in the grid after recording the goal position.
    """
    raw = LAYOUTS.get(name, LAYOUTS["mediumMaze"])
    grid = [list(line) for line in raw.split("\n") if line.strip()]
    if not grid:
        return (1, 1), (1, 2), [["%", " ", "%"], ["%", " ", "%"], ["%", "%", "%"]]
    max_w = max(len(row) for row in grid)
    for row in grid:
        row.extend(["%"] * (max_w - len(row)))

    start = goal = None
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "P":
                start = (r, c)
                row[c] = " "
            elif cell == "G":
                goal = (r, c)
                row[c] = " "
    if start is None:
        start = (1, 1)
    if goal is None:
        # fall back: use first reachable non-wall cell other than start
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell in (" ", "M") and (r, c) != start:
                    goal = (r, c)
            if goal:
                break
    return start, goal, grid


def path_cost(path: list, grid: list) -> int:
    """Sum of CELL_COST for every cell in path (start cell included)."""
    total = 0
    for pos in path:
        r, c = pos
        ch = grid[r][c] if 0 <= r < len(grid) and 0 <= c < len(grid[r]) else "%"
        total += CELL_COST.get(ch, 1)
    return total


def score_weighted_path(path: list, start: tuple, goal: tuple, grid: list) -> tuple:
    """
    Validate path and return (score, total_cost, reached).
    score = 1000 - total_cost  (higher score = cheaper path).
    """
    if not path:
        return -2000, 99999, False

    path = [tuple(p) for p in path]
    if path[0] != tuple(start):
        return -1000, 99999, False

    prev = path[0]
    for pos in path[1:]:
        r, c = pos
        if not (0 <= r < len(grid) and 0 <= c < len(grid[r])):
            break
        if grid[r][c] == "%":
            break
        if abs(r - prev[0]) + abs(c - prev[1]) != 1:
            break
        prev = pos
    else:
        reached = path[-1] == tuple(goal)
        cost = path_cost(path, grid)
        return (1000 - cost if reached else -cost), cost, reached

    # path broke early
    cost = path_cost(path[:path.index(prev) + 1], grid)
    return -cost, cost, False


def find_all_foods(layout_name: str):
    """
    Parse a layout and return (pacman_start, food_positions, grid).

    Only foods **reachable** from pacman_start are included; unreachable dots
    in disconnected chambers are silently dropped.
    grid has 'P' replaced with ' ' and all '.' removed (clean for the agent).
    """
    grid = parse_layout(layout_name)
    pacman = None
    all_foods = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == "P":
                pacman = (r, c)
                row[c] = " "
            elif cell == ".":
                all_foods.append((r, c))
                row[c] = " "
    if pacman is None:
        pacman = (1, 1)
    # Filter to foods the agent can actually reach.
    reachable = _reachable_from(grid, pacman)
    foods = [f for f in all_foods if f in reachable]
    return pacman, foods, grid


def simulate_collection(
    path: list,
    pacman_start: tuple,
    food_positions: list,
    grid: list,
) -> tuple:
    """
    Walk `path` and collect food dots.

    Returns (score, steps, n_collected, success).
      score = n_collected * 100  +  (500 if all collected)  -  steps_taken
    Path validity is enforced: bad steps stop simulation early.
    """
    food_set = set(map(tuple, food_positions))
    n_total   = len(food_set)
    collected = 0

    if not path:
        return -1000, 0, 0, False

    # Normalise to tuples
    path = [tuple(p) for p in path]

    if path[0] != tuple(pacman_start):
        return -500, 0, 0, False

    pos = path[0]
    steps = 0
    for nxt in path[1:]:
        nxt = tuple(nxt)
        r, c = nxt
        # out of bounds
        if not (0 <= r < len(grid) and 0 <= c < len(grid[r])):
            break
        # wall
        if grid[r][c] == "%":
            break
        # must be adjacent
        if abs(r - pos[0]) + abs(c - pos[1]) != 1:
            break
        pos   = nxt
        steps += 1
        if nxt in food_set:
            food_set.discard(nxt)
            collected += 1

    success = (len(food_set) == 0)
    score   = collected * 100 + (500 if success else 0) - steps
    return score, steps, collected, success


def run_search_agent(layout_name, fn_name, heuristic_name="nullHeuristic"):
    grid = parse_layout(layout_name)
    start, goal, grid = find_positions(grid)
    algo = ALGORITHMS.get(fn_name)
    if algo is None:
        algo = ALGORITHMS["breadthFirstSearch"]
    path = algo(start, goal, grid, heuristic_name=heuristic_name)
    num_moves = max(0, len(path) - 1)
    score = 1000 - num_moves
    return score, num_moves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layout", default="mediumMaze")
    parser.add_argument("-p", "--pacman", default="SearchAgent")
    parser.add_argument("-a", "--agentArgs", default="")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    fn_name = "breadthFirstSearch"
    heuristic_name = "nullHeuristic"
    for part in args.agentArgs.split(","):
        part = part.strip()
        if part.startswith("fn="):
            fn_name = part[3:].strip()
        elif part.startswith("heuristic="):
            heuristic_name = part[10:].strip()

    if args.pacman != "SearchAgent":
        print("Score: 0", file=sys.stderr)
        print("Total moves: 0")
        sys.exit(0)

    score, moves = run_search_agent(args.layout, fn_name, heuristic_name)
    print("Score: {}".format(score))
    print("Total moves: {}".format(moves))


if __name__ == "__main__":
    main()
