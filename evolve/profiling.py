from __future__ import annotations

import cProfile
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import pstats

T = TypeVar("T")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROFILE_DIR = _REPO_ROOT / "profiles"
_ACTIVE = threading.local()


def _short_path(filename: str) -> str:
    path = Path(filename)
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except Exception:
        return path.name


def _summarize_profile(
    profiler: cProfile.Profile,
    name: str,
    elapsed_seconds: float,
    limit: int = 30,
) -> dict[str, Any]:
    stats = pstats.Stats(profiler)
    rows: list[dict[str, Any]] = []

    for (filename, line, func_name), (primitive_calls, total_calls, tottime, cumtime, _callers) in stats.stats.items():
        rows.append(
            {
                "function": f"{_short_path(filename)}:{line}:{func_name}",
                "file": _short_path(filename),
                "line": line,
                "name": func_name,
                "primitive_calls": primitive_calls,
                "total_calls": total_calls,
                "tottime": round(tottime, 6),
                "cumtime": round(cumtime, 6),
                "percall_tottime": round(tottime / total_calls, 6) if total_calls else 0.0,
                "percall_cumtime": round(cumtime / primitive_calls, 6) if primitive_calls else 0.0,
            }
        )

    rows.sort(key=lambda row: row["cumtime"], reverse=True)
    return {
        "strategy": name,
        "profiling_mode": "cprofile",
        "elapsed_seconds": round(elapsed_seconds, 6),
        "total_calls": stats.total_calls,
        "primitive_calls": stats.prim_calls,
        "top_functions": rows[:limit],
    }


def _summarize_wall_time(name: str, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "strategy": name,
        "profiling_mode": "wall_time_only",
        "elapsed_seconds": round(elapsed_seconds, 6),
        "total_calls": 0,
        "primitive_calls": 0,
        "top_functions": [],
    }


def run_profiled(
    name: Any,
    fn: Callable[[], T],
    profile_id: str | None = None,
) -> tuple[T, dict[str, Any]]:
    profile_name = getattr(name, "value", str(name))
    if getattr(_ACTIVE, "enabled", False) or sys.getprofile() is not None:
        # cProfile cannot be nested in one thread. Keep timing for inner phases
        # while the outer "system" profiler records the full call graph.
        started = time.perf_counter()
        result = fn()
        return result, _summarize_wall_time(profile_name, time.perf_counter() - started)

    profiler = cProfile.Profile()
    started = time.perf_counter()
    _ACTIVE.enabled = True
    try:
        try:
            profiler.enable()
        except RuntimeError:
            _ACTIVE.enabled = False
            result = fn()
            return result, _summarize_wall_time(profile_name, time.perf_counter() - started)
        try:
            result = fn()
        finally:
            profiler.disable()
    finally:
        _ACTIVE.enabled = False
    elapsed_seconds = time.perf_counter() - started
    summary = _summarize_profile(profiler, profile_name, elapsed_seconds)
    if profile_id:
        _PROFILE_DIR.mkdir(exist_ok=True)
        prof_path = _PROFILE_DIR / f"{profile_id}_{profile_name}.prof"
        profiler.dump_stats(str(prof_path))
        summary["prof_path"] = str(prof_path.relative_to(_REPO_ROOT))
    return result, summary
