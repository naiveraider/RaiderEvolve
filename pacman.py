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
import sys

# --------------- Layouts (Berkeley-style names) ---------------
# Connected maze so P and goal (.) are in same component
MEDIUM_MAZE = """
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
%.....%.....%.....%.....%.%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
""".strip()

LAYOUTS = {
    "mediumMaze": MEDIUM_MAZE,
    "tinyMaze": """
%%%%
%.P%
%..%
%%%%
""".strip(),
}


def parse_layout(name):
    raw = LAYOUTS.get(name, LAYOUTS["mediumMaze"])
    grid = [list(line) for line in raw.split("\n") if line.strip()]
    if not grid:
        return [["%", " ", "%"], ["%", "P", "%"], ["%", "%", "%"]]
    max_w = max(len(row) for row in grid)
    for row in grid:
        row.extend([" "] * (max_w - len(row)))
    return grid


def _reachable_from(grid, start):
    """Set of positions reachable from start (BFS)."""
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


HEURISTICS = {
    "nullHeuristic": null_heuristic,
    "manhattanHeuristic": manhattan_heuristic,
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


ALGORITHMS = {
    "breadthFirstSearch": lambda s, g, gr: breadth_first_search(s, g, gr),
    "depthFirstSearch": lambda s, g, gr: depth_first_search(s, g, gr),
    "uniformCostSearch": lambda s, g, gr: uniform_cost_search(s, g, gr),
    "astar": lambda s, g, gr, h="nullHeuristic": astar_search(s, g, gr, h),
}


def run_search_agent(layout_name, fn_name, heuristic_name="nullHeuristic"):
    grid = parse_layout(layout_name)
    start, goal, grid = find_positions(grid)
    if fn_name == "astar":
        path = astar_search(start, goal, grid, heuristic_name)
    else:
        path = ALGORITHMS[fn_name](start, goal, grid)
    num_moves = max(0, len(path) - 1)
    # Berkeley-style score: higher is better; penalize moves (e.g. 1000 - moves)
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
    # Always print to stdout so evaluator can parse (e.g. evolution subprocess)
    print("Score: {}".format(score))
    print("Total moves: {}".format(moves))


if __name__ == "__main__":
    main()
