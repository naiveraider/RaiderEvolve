from __future__ import annotations

import ast
from typing import Any

import numpy as np


class OpCountVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.mul = 0
        self.add = 0

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mult):
            self.mul += 1
        elif isinstance(node.op, ast.Add):
            self.add += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.op, ast.Mult):
            self.mul += 1
        elif isinstance(node.op, ast.Add):
            self.add += 1
        self.generic_visit(node)


def count_ops_static(code: str) -> tuple[int, int]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 9999, 9999
    v = OpCountVisitor()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "matmul":
            for stmt in node.body:
                v.visit(stmt)
            break
    return v.mul, v.add


def load_matmul(code: str):
    ns: dict[str, Any] = {"__builtins__": __builtins__}
    exec(code, ns, ns)  # noqa: S102
    if "matmul" not in ns:
        raise ValueError("Code must define matmul(a, b)")
    return ns["matmul"]


def matrix_correctness_and_ops(code: str, alpha: float, beta: float) -> tuple[float, dict[str, Any]]:
    mul_s, add_s = count_ops_static(code)
    metrics: dict[str, Any] = {
        "static_mul": mul_s,
        "static_add": add_s,
        "correctness": 0.0,
    }
    try:
        fn = load_matmul(code)
    except Exception as e:
        metrics["error"] = str(e)
        return -1e6, metrics

    ok = True
    rng = np.random.default_rng(0)
    for _ in range(12):
        a = rng.integers(-3, 4, size=(3, 3))
        b = rng.integers(-3, 4, size=(3, 3))
        try:
            got = np.array(fn(a.tolist(), b.tolist()))
            exp = a @ b
            if got.shape != (3, 3) or not np.allclose(got, exp):
                ok = False
                break
        except Exception as e:
            ok = False
            metrics["error"] = str(e)
            break

    correctness = 1.0 if ok else 0.0
    metrics["correctness"] = correctness
    fitness = correctness - alpha * mul_s - beta * add_s
    if not ok:
        fitness = -1e3 - alpha * mul_s - beta * add_s
    return fitness, metrics


def baseline_matrix_code() -> str:
    return '''def matmul(a, b):
    n = 3
    c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]
    return c
'''
