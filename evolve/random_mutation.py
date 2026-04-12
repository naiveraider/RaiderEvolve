from __future__ import annotations

import random
import re


def random_line_swap(code: str, rng: random.Random) -> str:
    lines = code.splitlines()
    if len(lines) < 2:
        return code
    i, j = rng.sample(range(len(lines)), 2)
    lines[i], lines[j] = lines[j], lines[i]
    return "\n".join(lines)


def random_param_tweak(code: str, rng: random.Random) -> str:
    def bump(m: re.Match[str]) -> str:
        val = float(m.group(0))
        delta = rng.choice([-0.5, -0.25, 0.25, 0.5, 1.0])
        return str(max(0.0, val + delta))

    return re.sub(r"\b\d+\.\d+\b", bump, code, count=rng.randint(1, 3))


def random_mutate(code: str, rng: random.Random) -> tuple[str, str]:
    op = rng.choice(["swap", "tweak", "both"])
    out = code
    notes: list[str] = []
    if op in ("swap", "both"):
        out = random_line_swap(out, rng)
        notes.append("random_line_swap")
    if op in ("tweak", "both"):
        out = random_param_tweak(out, rng)
        notes.append("param_tweak")
    return out, "; ".join(notes)
