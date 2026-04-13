"""
takes the json files in runs/ that sweep.py wrote and turns them into
mdarahman_data.csv. one row per (config, strategy, generation), so 4
configs x 2 strategies x 5 gens = 40 rows.

columns i pull out: config, w1, w2, w3, strategy, generation, best_fitness,
avg_fitness, and then the metrics from the best candidate of each gen
(avg_score, avg_cost, avg_steps, avg_cells_accessed, eval_time_ms,
success_rate). that way the csv has everything the rubric asks for.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

RUNS = Path(__file__).parent / "runs"
OUT = Path(__file__).parent / "mdarahman_data.csv"

CONFIGS = {
    "A_score_heavy": (0.90, 0.09, 0.01),
    "B_baseline":    (0.50, 0.30, 0.20),
    "C_balanced":    (0.25, 0.25, 0.50),
    "D_cost_heavy":  (0.05, 0.05, 0.90),
}


def best_record_for_gen(mem: list[dict], gen: int) -> dict | None:
    cands = [r for r in mem if r.get("generation") == gen]
    if not cands:
        return None
    return max(cands, key=lambda r: float(r.get("fitness", -1e12)))


def main() -> None:
    rows = []
    for label, (w1, w2, w3) in CONFIGS.items():
        for strategy in ("full", "random_only"):
            p = RUNS / f"{label}__{strategy}.json"
            if not p.exists():
                print(f"SKIP missing {p.name}")
                continue
            data = json.loads(p.read_text())
            run = data["runs"][0]
            best_curve = run["best_per_generation"]
            avg_curve = run["avg_fitness_per_gen"]
            mem = run["memory_records"]
            for gen, (bf, af) in enumerate(zip(best_curve, avg_curve)):
                rec = best_record_for_gen(mem, gen) or {}
                m = rec.get("metrics", {}) or {}
                rows.append({
                    "config": label,
                    "w1": w1, "w2": w2, "w3": w3,
                    "strategy": strategy,
                    "generation": gen,
                    "best_fitness": round(bf, 4),
                    "avg_fitness": round(af, 4),
                    "best_avg_score": round(m.get("avg_score", 0.0), 4),
                    "best_avg_cost": round(m.get("avg_cost", 0.0), 4),
                    "best_avg_steps": round(m.get("avg_steps", 0.0), 4),
                    "best_avg_cells_accessed": round(m.get("avg_cells_accessed", 0.0), 2),
                    "best_eval_time_ms": round(m.get("eval_time_ms", 0.0), 2),
                    "best_success_rate": m.get("success_rate", 0.0),
                })
    fields = list(rows[0].keys())
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
