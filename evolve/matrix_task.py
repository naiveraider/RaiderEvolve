from __future__ import annotations

from typing import Any

import numpy as np

_EVAL_TIMEOUT = 3.0  # seconds — kill runaway LLM-generated matmul code

# ── Runtime operation counter ─────────────────────────────────────────────────
# Static AST counting is broken: a single `*` inside a triple-nested loop
# shows up as ONE multiplication, not 27.  We instead wrap inputs in a
# TrackedNum that intercepts every actual arithmetic operation at runtime.

class _TrackedNum:
    """Number proxy that counts scalar multiplications and additions/subtractions."""
    __slots__ = ("v",)
    muls: int = 0
    adds: int = 0

    def __init__(self, v):        self.v = int(v) if isinstance(v, np.integer) else v

    # helpers
    @staticmethod
    def _val(o):
        return o.v if isinstance(o, _TrackedNum) else o

    def __add__(self, o):      _TrackedNum.adds += 1; return _TrackedNum(self.v + self._val(o))
    def __radd__(self, o):     _TrackedNum.adds += 1; return _TrackedNum(self._val(o) + self.v)
    def __sub__(self, o):      _TrackedNum.adds += 1; return _TrackedNum(self.v - self._val(o))
    def __rsub__(self, o):     _TrackedNum.adds += 1; return _TrackedNum(self._val(o) - self.v)
    def __mul__(self, o):      _TrackedNum.muls += 1; return _TrackedNum(self.v * self._val(o))
    def __rmul__(self, o):     _TrackedNum.muls += 1; return _TrackedNum(self._val(o) * self.v)
    def __neg__(self):         return _TrackedNum(-self.v)
    def __pos__(self):         return _TrackedNum(+self.v)
    def __abs__(self):         return _TrackedNum(abs(self.v))
    def __int__(self):         return int(self.v)
    def __float__(self):       return float(self.v)
    def __repr__(self):        return repr(self.v)
    def __eq__(self, o):       return self.v == self._val(o)
    def __lt__(self, o):       return self.v <  self._val(o)
    def __le__(self, o):       return self.v <= self._val(o)
    def __gt__(self, o):       return self.v >  self._val(o)
    def __ge__(self, o):       return self.v >= self._val(o)
    def __hash__(self):        return hash(self.v)


def _wrap(mat):
    return [[_TrackedNum(x) for x in row] for row in mat]

def _unwrap(mat) -> list[list[float]]:
    return [[float(x.v) if isinstance(x, _TrackedNum) else float(x) for x in row]
            for row in mat]


_EVAL_RNG = np.random.default_rng(42)
# Fixed test matrices — generated once, reused every call (no RNG overhead per eval).
_TEST_PAIRS: list[tuple[list, list, Any]] = [
    (a.tolist(), b.tolist(), a @ b)
    for a, b in [
        (_EVAL_RNG.integers(-3, 4, size=(3, 3)), _EVAL_RNG.integers(-3, 4, size=(3, 3)))
        for _ in range(4)
    ]
]


# ── Standard 3×3 baseline op counts (measured at runtime) ────────────────────
# 3-nested-loop: 27 multiplications, 27 augmented-additions (c[i][j] += ... × 27)
BASELINE_MULS = 27
BASELINE_ADDS = 27


# ── Code loader ───────────────────────────────────────────────────────────────

def load_matmul(code: str):
    ns: dict[str, Any] = {"__builtins__": __builtins__}
    exec(code, ns, ns)  # noqa: S102
    if "matmul" not in ns:
        raise ValueError("Code must define matmul(a, b)")
    return ns["matmul"]


# ── Fitness ───────────────────────────────────────────────────────────────────
# fitness = correctness_bonus + mul_savings_score + add_savings_score
#
# mul_savings_score = (BASELINE_MULS - actual_muls) / BASELINE_MULS * MUL_WEIGHT
#   MUL_WEIGHT = 10  → saving all 27 muls adds 10 pts; saving 4 muls (Laderman) ≈ +1.5 pts
# add_savings_score = small bonus for fewer additions (secondary)
#
# Incorrect code → heavy penalty (-10) so correctness is never sacrificed.
#
# Scale (for reference):
#   standard 3-loop (27 muls, 18 adds)  → fitness = 1.00  (baseline)
#   Laderman / 23 muls                  → fitness ≈ 2.48  (+148%)
#   21 muls (theoretical best known)    → fitness ≈ 3.22  (+222%)
#   wrong answer                        → fitness = -10.0

MUL_WEIGHT = 10.0
# ADD_WEIGHT is intentionally tiny: the goal is to minimise *multiplications*.
# Trading 70 extra additions for 4 fewer multiplications must be profitable.
# With ADD_WEIGHT=0.05: saving 4 muls = +1.48, adding 70 adds = -0.13 → net +1.35 ✓
ADD_WEIGHT  = 0.05


def _eval_once(fn) -> tuple[bool, int, int, str]:
    """
    Run fn over all _TEST_PAIRS with TrackedNum inputs.
    Returns (correct, avg_muls, avg_adds, error_msg).
    Single loop: correctness + op counting in one pass (no duplicate calls).
    """
    total_muls = total_adds = 0
    for a_lst, b_lst, expected in _TEST_PAIRS:
        _TrackedNum.muls = _TrackedNum.adds = 0
        try:
            result = fn(_wrap(a_lst), _wrap(b_lst))
            got = np.array(_unwrap(result))
        except Exception as exc:
            return False, 9999, 9999, str(exc)
        if got.shape != (3, 3) or not np.allclose(got, expected):
            return False, 9999, 9999, "wrong result"
        total_muls += _TrackedNum.muls
        total_adds  += _TrackedNum.adds
    n = len(_TEST_PAIRS)
    return True, total_muls // n, total_adds // n, ""


def matrix_correctness_and_ops(
    code: str,
    alpha: float = 0.0,   # kept for API compat
    beta:  float = 0.0,
) -> tuple[float, dict[str, Any]]:
    metrics: dict[str, Any] = {"correctness": 0.0}

    try:
        fn = load_matmul(code)
    except Exception as e:
        metrics["error"] = str(e)
        return -1e6, metrics

    # Run with timeout — daemon thread returns control immediately on timeout;
    # the background thread finishes on its own (Python can't kill threads).
    import threading
    result_box: list = [None]
    exc_box:    list = [None]

    def _run() -> None:
        try:
            result_box[0] = _eval_once(fn)
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=_EVAL_TIMEOUT)

    if t.is_alive():
        metrics["error"] = f"timeout >{_EVAL_TIMEOUT}s"
        return -10.0, metrics
    if exc_box[0] is not None:
        metrics["error"] = str(exc_box[0])
        return -1e6, metrics

    ok, actual_muls, actual_adds, err = result_box[0]

    metrics["correctness"] = 1.0 if ok else 0.0
    metrics["actual_muls"]  = actual_muls
    metrics["actual_adds"]  = actual_adds
    metrics["baseline_muls"] = BASELINE_MULS
    metrics["baseline_adds"] = BASELINE_ADDS
    if err:
        metrics["error"] = err

    if not ok:
        metrics["note"] = "incorrect — no operation bonus"
        return -10.0, metrics

    mul_savings = (BASELINE_MULS - actual_muls) / BASELINE_MULS
    add_savings = (BASELINE_ADDS - actual_adds) / BASELINE_ADDS
    fitness = 1.0 + mul_savings * MUL_WEIGHT + add_savings * ADD_WEIGHT
    metrics["mul_savings_score"] = round(mul_savings * MUL_WEIGHT, 4)
    metrics["add_savings_score"] = round(add_savings * ADD_WEIGHT, 4)
    return fitness, metrics


def baseline_matrix_code() -> str:
    return '''\
def matmul(a, b):
    """Standard 3-nested-loop: 27 multiplications → fitness = 1.0 (baseline).
    Goal: reduce scalar multiplications via Strassen-style factoring.
    fitness = 1.0 + (27-actual_muls)/27*10  (each saved mul = +0.37)
    Muls counted at RUNTIME — this loop does 27, not 1.
    Target: 23 muls (Laderman 1976) → fitness ≈ 2.35
    """
    n = 3
    c = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] += a[i][k] * b[k][j]
    return c
'''


def laderman_matrix_code() -> str:
    """Laderman 1976 — 23 multiplications, fitness ≈ 2.35.
    Used as a known-good seed injected into the initial population so the LLM
    can start from the frontier rather than re-discovering it from scratch.
    """
    return '''\
def matmul(a, b):
    """Laderman 1976: 23 scalar multiplications → fitness ≈ 2.35.
    Precomputes 23 products of linear combinations of a/b entries,
    then recovers all 9 output cells via additions only.
    Goal: find a correct decomposition with fewer than 23 multiplications.
    Current best known lower bound: 19. Practical target: 21-22.
    """
    m = [None] * 24  # 1-indexed to match the original paper
    m[1]  = (a[0][0]+a[0][1]+a[0][2]-a[1][0]-a[1][1]-a[2][1]-a[2][2])*b[1][1]
    m[2]  = (a[0][0]-a[1][0])*(-b[0][1]+b[1][1])
    m[3]  = a[1][1]*(-b[0][0]+b[0][1]+b[1][0]-b[1][1]-b[1][2]-b[2][0]+b[2][2])
    m[4]  = (-a[0][0]+a[1][0]+a[1][1])*(b[0][0]-b[0][1]+b[1][1])
    m[5]  = (a[1][0]+a[1][1])*(-b[0][0]+b[0][1])
    m[6]  = a[0][0]*b[0][0]
    m[7]  = (-a[0][0]+a[2][0]+a[2][1])*(b[0][0]-b[0][2]+b[1][2])
    m[8]  = (-a[0][0]+a[2][0])*(b[0][2]-b[1][2])
    m[9]  = (a[2][0]+a[2][1])*(-b[0][0]+b[0][2])
    m[10] = (a[0][0]+a[0][1]+a[0][2]-a[1][1]-a[1][2]-a[2][0]-a[2][1])*b[1][2]
    m[11] = a[2][1]*(-b[0][0]+b[0][2]+b[1][0]-b[1][1]-b[1][2]-b[2][0]+b[2][1])
    m[12] = (-a[0][2]+a[2][1]+a[2][2])*(b[1][1]+b[2][0]-b[2][1])
    m[13] = (a[0][2]-a[2][2])*(b[1][1]-b[2][1])
    m[14] = a[0][2]*b[2][0]
    m[15] = (a[2][1]+a[2][2])*(-b[2][0]+b[2][1])
    m[16] = (-a[0][2]+a[1][1]+a[1][2])*(b[1][2]+b[2][0]-b[2][2])
    m[17] = (a[0][2]-a[1][2])*(b[1][2]-b[2][2])
    m[18] = (a[1][1]+a[1][2])*(-b[2][0]+b[2][2])
    m[19] = a[0][1]*b[1][0]
    m[20] = a[1][2]*b[2][1]
    m[21] = a[1][0]*b[0][2]
    m[22] = a[2][0]*b[0][1]
    m[23] = a[2][2]*b[2][2]
    return [
        [m[6]+m[14]+m[19],
         m[1]+m[4]+m[5]+m[6]+m[12]+m[14]+m[15],
         m[6]+m[7]+m[9]+m[10]+m[14]+m[16]+m[18]],
        [m[2]+m[3]+m[4]+m[6]+m[14]+m[16]+m[17],
         m[2]+m[4]+m[5]+m[6]+m[20],
         m[14]+m[16]+m[17]+m[18]+m[21]],
        [m[6]+m[7]+m[8]+m[11]+m[12]+m[13]+m[14],
         m[12]+m[13]+m[14]+m[15]+m[22],
         m[6]+m[7]+m[8]+m[9]+m[23]],
    ]
'''
