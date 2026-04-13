"""
my round 2 sweep script for the pacman task.

i wanted to test the custom fitness weights (w1, w2, w3) so this hits the
deployed backend 8 times - 4 weight configs x 2 strategies (full evolution
and random_only as the baseline) - and dumps each json response into runs/
so build_csv.py can flatten them later.

i kept seed=42 on every run so anything different between configs is from
the weights, not from rng. takes about 12 minutes total because the full
runs are slow.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib import request, error

BACKEND = "http://43.130.56.234:3000/api/backend/evolve/sync"
OUT_DIR = Path(__file__).parent / "runs"
OUT_DIR.mkdir(exist_ok=True)
LOG = Path(__file__).parent / "progress.log"

# (label, w1, w2, w3). each weight has to be in [0, 1] and the backend
# normalises them. when i looked at evolve/pacman_env.py line 235 the math
# is fitness = avg_score - w3_norm * avg_cost * 0.01, so w1 and w2 dont
# actually do anything. i picked these 4 configs so the normalised w3
# spreads almost the full range and i can see how much it really moves.
CONFIGS = [
    ("A_score_heavy",  0.90, 0.09, 0.01),
    ("B_baseline",     0.50, 0.30, 0.20),
    ("C_balanced",     0.25, 0.25, 0.50),
    ("D_cost_heavy",   0.05, 0.05, 0.90),
]

STRATEGIES = ["full", "random_only"]

BASE_REQ = {
    "task": "pacman",
    "source_code": "",
    "generations": 4,
    "population_size": 5,
    "top_k": 3,
    "selection_mode": "diversity",
    "fitness_preset": "custom",
    "matrix_alpha": 0.01,
    "matrix_beta": 0.005,
    "include_pseudocode_log": False,
    "seed": 42,
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def run_one(label: str, w1: float, w2: float, w3: float, strategy: str) -> dict:
    payload = dict(BASE_REQ)
    payload["custom_weights"] = {"w1": w1, "w2": w2, "w3": w3}
    payload["strategies"] = [strategy]
    body = json.dumps(payload).encode()
    req = request.Request(
        BACKEND,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with request.urlopen(req, timeout=360) as resp:
        data = json.loads(resp.read())
    dt = time.time() - t0
    out = OUT_DIR / f"{label}__{strategy}.json"
    with out.open("w") as f:
        json.dump(data, f, indent=2)
    run = data["runs"][0]
    log(
        f"  ok {label}/{strategy}  t={dt:5.1f}s  "
        f"best={run['final_best_fitness']:.2f}  "
        f"curve_len={len(run['best_per_generation'])}"
    )
    return data


def main() -> None:
    LOG.write_text("")
    log(f"starting sweep — {len(CONFIGS)} configs x {len(STRATEGIES)} strategies")
    for label, w1, w2, w3 in CONFIGS:
        log(f"config {label}  w=({w1},{w2},{w3})")
        for strat in STRATEGIES:
            try:
                run_one(label, w1, w2, w3, strat)
            except error.HTTPError as e:
                log(f"  HTTP {e.code} on {label}/{strat}: {e.read()[:200]!r}")
            except Exception as e:  # noqa: BLE001
                log(f"  ERROR on {label}/{strat}: {e!r}")
    log("sweep complete")


if __name__ == "__main__":
    main()
