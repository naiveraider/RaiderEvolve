"""
makes the 4 plots i need for the round 2 doc, straight from the runs/
json files. run this after sweep.py.

    plot_fitness_vs_generation.png   best fitness per gen, one line per config
    plot_benchmark_compare.png       full vs random_only bars, final best
    plot_steps_vs_generation.png     best avg_steps per gen
    plot_runtime_vs_generation.png   best eval_time_ms per gen

i kept the styling simple on purpose so the lines are easy to read in the
docx at full width.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(__file__).parent / "runs"
OUT = Path(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)

CONFIGS = ["A_score_heavy", "B_baseline", "C_balanced", "D_cost_heavy"]
COLORS = {"A_score_heavy": "#1f77b4", "B_baseline": "#2ca02c",
          "C_balanced": "#ff7f0e", "D_cost_heavy": "#d62728"}


def load(label: str, strat: str) -> dict | None:
    p = RUNS / f"{label}__{strat}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["runs"][0]


def best_metric_per_gen(run: dict, key: str) -> list[float]:
    gens = max((r["generation"] for r in run["memory_records"]), default=-1) + 1
    out = []
    for g in range(gens):
        cands = [r for r in run["memory_records"] if r["generation"] == g]
        if not cands:
            out.append(float("nan"))
            continue
        best = max(cands, key=lambda r: float(r.get("fitness", -1e12)))
        out.append(float(best.get("metrics", {}).get(key, 0.0)))
    return out


def _save(fig, name: str) -> None:
    p = OUT / name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.name)


def plot_fitness() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label in CONFIGS:
        run = load(label, "full")
        if not run:
            continue
        curve = run["best_per_generation"]
        ax.plot(range(len(curve)), curve, marker="o",
                label=label, color=COLORS[label])
    ax.set_title("Best fitness per generation — full evolution (seed 42)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, "plot_fitness_vs_generation.png")


def plot_benchmark() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(CONFIGS))
    full_vals, rand_vals = [], []
    for label in CONFIGS:
        f = load(label, "full")
        r = load(label, "random_only")
        full_vals.append(f["final_best_fitness"] if f else 0)
        rand_vals.append(r["final_best_fitness"] if r else 0)
    width = 0.35
    ax.bar([i - width/2 for i in x], full_vals, width,
           label="Full evolution", color="#2ca02c")
    ax.bar([i + width/2 for i in x], rand_vals, width,
           label="Random-only baseline", color="#7f7f7f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(CONFIGS, rotation=15, ha="right")
    ax.set_ylabel("Final best fitness")
    ax.set_title("Full evolution vs random-only benchmark (final best)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    _save(fig, "plot_benchmark_compare.png")


def plot_steps() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label in CONFIGS:
        run = load(label, "full")
        if not run:
            continue
        curve = best_metric_per_gen(run, "avg_steps")
        ax.plot(range(len(curve)), curve, marker="s",
                label=label, color=COLORS[label])
    ax.set_title("Best-candidate path length (avg_steps) per generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Avg steps (lower = better)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "plot_steps_vs_generation.png")


def plot_runtime() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label in CONFIGS:
        run = load(label, "full")
        if not run:
            continue
        curve = best_metric_per_gen(run, "eval_time_ms")
        ax.plot(range(len(curve)), curve, marker="^",
                label=label, color=COLORS[label])
    ax.set_title("Best-candidate evaluation runtime per generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("eval_time_ms")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "plot_runtime_vs_generation.png")


if __name__ == "__main__":
    plot_fitness()
    plot_benchmark()
    plot_steps()
    plot_runtime()
