from __future__ import annotations

import asyncio
import csv
import io
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from evolve.controller import PSEUDOCODE_OUTLINE, run_evolution_run
from evolve.llm_client import LLMRequestError
from evolve.models import EvolutionRequest, EvolutionResponse, EvolutionStrategy, JobStatus
from evolve.profiling import run_profiled
from evolve.settings import settings

app = FastAPI(title="Evolve System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_JOBS: dict[str, JobStatus] = {}
_PROFILES: dict[str, dict[str, Any]] = {}
_LATEST_PROFILE_ID: str | None = None
_PROFILE_LOCK = threading.Lock()
_PROFILE_LIMIT = 20
_PROFILE_DIR = Path(__file__).resolve().parent / "profiles"


def _time_phase(name: EvolutionStrategy, fn) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - started
    return result, {
        "strategy": name.value,
        "profiling_mode": "wall_time_only",
        "elapsed_seconds": round(elapsed, 6),
        "total_calls": 0,
        "primitive_calls": 0,
        "top_functions": [],
    }


def _save_profile_snapshot(
    profile_id: str,
    req: EvolutionRequest,
    runs: list[dict[str, Any]],
    system_profile: dict[str, Any] | None = None,
) -> str:
    global _LATEST_PROFILE_ID
    snapshot = {
        "profile_id": profile_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": req.task.value,
        "strategies": [run["strategy"] for run in runs],
        "total_elapsed_seconds": (
            float(system_profile["elapsed_seconds"])
            if system_profile
            else round(sum(float(run["elapsed_seconds"]) for run in runs), 6)
        ),
        "system_profile": system_profile,
        "runs": runs,
    }
    _PROFILE_DIR.mkdir(exist_ok=True)
    (_PROFILE_DIR / f"{profile_id}.json").write_text(
        json.dumps(snapshot, indent=2),
        encoding="utf-8",
    )
    with _PROFILE_LOCK:
        _PROFILES[profile_id] = snapshot
        _LATEST_PROFILE_ID = profile_id
        while len(_PROFILES) > _PROFILE_LIMIT:
            oldest_id = next(iter(_PROFILES))
            del _PROFILES[oldest_id]
    return profile_id


def _run_job(job_id: str, req: EvolutionRequest) -> None:
    st = _JOBS[job_id]
    try:
        st.status = "running"
        st.message = "evolving"
        strategies = req.strategies or [
            EvolutionStrategy.SINGLE_LLM,
            EvolutionStrategy.RANDOM_ONLY,
            EvolutionStrategy.FULL,
        ]
        profile_id = str(uuid.uuid4()) if settings.enable_profile else None

        def _execute_job() -> tuple[list[Any], list[dict[str, Any]]]:
            runs = []
            profile_runs = []
            total = len(strategies)
            for i, strat in enumerate(strategies):
                st.progress = (i / total) * 100.0
                result, profile = _time_phase(
                    strat,
                    lambda s=strat: run_evolution_run(req, s),
                )
                runs.append(result)
                profile_runs.append(profile)
            return runs, profile_runs

        if profile_id:
            (runs, profile_runs), system_profile = run_profiled("system", _execute_job, profile_id)
            _save_profile_snapshot(profile_id, req, profile_runs, system_profile)
        else:
            runs, _profile_runs = _execute_job()
        st.result = EvolutionResponse(
            task=req.task,
            runs=runs,
            pseudocode_outline=PSEUDOCODE_OUTLINE if req.include_pseudocode_log else None,
            algorithm_explanation=(
                "Evolution alternates selection, hybrid mutation, and evaluation "
                "to maximize task fitness while logging full ancestry."
                if req.include_pseudocode_log
                else None
            ),
        )
        st.status = "done"
        st.progress = 100.0
        st.message = f"complete; cProfile={profile_id}" if profile_id else "complete"
    except Exception as e:
        st.status = "error"
        st.error = str(e)
        st.message = "failed"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "evolve"}


@app.post("/evolve", response_model=JobStatus)
def start_evolve(req: EvolutionRequest, bg: BackgroundTasks) -> JobStatus:
    job_id = str(uuid.uuid4())
    st = JobStatus(job_id=job_id, status="queued", progress=0.0, message="queued")
    _JOBS[job_id] = st
    bg.add_task(_run_job_sync, job_id, req)
    return st


def _run_job_sync(job_id: str, req: EvolutionRequest) -> None:
    _run_job(job_id, req)


@app.get("/evolve/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    st = _JOBS.get(job_id)
    if not st:
        raise HTTPException(404, "job not found")
    return st


@app.post("/evolve/sync", response_model=EvolutionResponse)
def evolve_sync(req: EvolutionRequest) -> EvolutionResponse:
    strategies = req.strategies or [
        EvolutionStrategy.SINGLE_LLM,
        EvolutionStrategy.RANDOM_ONLY,
        EvolutionStrategy.FULL,
    ]
    try:
        profile_id = str(uuid.uuid4()) if settings.enable_profile else None

        def _execute_sync() -> tuple[list[Any], list[dict[str, Any]]]:
            profiled = [
                _time_phase(s, lambda strat=s: run_evolution_run(req, strat))
                for s in strategies
            ]
            return [result for result, _profile in profiled], [
                profile for _result, profile in profiled
            ]

        if profile_id:
            (runs, profile_runs), system_profile = run_profiled("system", _execute_sync, profile_id)
            _save_profile_snapshot(profile_id, req, profile_runs, system_profile)
        else:
            runs, _profile_runs = _execute_sync()
    except LLMRequestError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return EvolutionResponse(
        task=req.task,
        runs=runs,
        pseudocode_outline=PSEUDOCODE_OUTLINE if req.include_pseudocode_log else None,
        algorithm_explanation=(
            "OpenEvolve-style loop with LLM, random, and template mutation."
            if req.include_pseudocode_log
            else None
        ),
    )


@app.post("/evolve/stream")
async def evolve_stream(req: EvolutionRequest) -> StreamingResponse:
    """Server-Sent Events endpoint — yields progress events then the final result."""
    strategies = req.strategies or [
        EvolutionStrategy.SINGLE_LLM,
        EvolutionStrategy.RANDOM_ONLY,
        EvolutionStrategy.FULL,
    ]
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _worker() -> None:
        profile_id = str(uuid.uuid4()) if settings.enable_profile else None
        try:
            def _execute_stream() -> tuple[EvolutionResponse, list[dict[str, Any]]]:
                results = []
                profile_runs = []
                for strat in strategies:
                    def _cb(data: dict, _s=strat) -> None:
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", **data})

                    result, profile = _time_phase(
                        strat,
                        lambda s=strat: run_evolution_run(req, s, progress_cb=_cb),
                    )
                    results.append(result)
                    profile_runs.append(profile)
                    # Emit a lightweight per-strategy summary (not the full object)
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {
                            "type": "strategy_done",
                            "strategy": strat.value,
                            "final_best": result.final_best_fitness,
                            "best_per_generation": result.best_per_generation,
                            "avg_fitness_per_gen": result.avg_fitness_per_gen,
                        },
                    )
                final = EvolutionResponse(
                    task=req.task,
                    runs=results,
                    pseudocode_outline=PSEUDOCODE_OUTLINE if req.include_pseudocode_log else None,
                    algorithm_explanation=(
                        "OpenEvolve-style loop with LLM, random, and template mutation."
                        if req.include_pseudocode_log
                        else None
                    ),
                )
                return final, profile_runs

            if profile_id:
                (final, profile_runs), system_profile = run_profiled("system", _execute_stream, profile_id)
                _save_profile_snapshot(profile_id, req, profile_runs, system_profile)
            else:
                final, _profile_runs = _execute_stream()
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "done", "result": final.model_dump(), "profile_id": profile_id},
            )
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "detail": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    async def _generator():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class BestUpToBody(BaseModel):
    memory_records: list[dict[str, Any]]
    max_generation: int = Field(ge=0)


@app.post("/analytics/best-up-to")
def best_up_to(body: BestUpToBody) -> Dict[str, Optional[float]]:
    subset = [r for r in body.memory_records if r.get("generation", 0) <= body.max_generation]
    if not subset:
        return {"best": None}
    best = max(subset, key=lambda r: float(r.get("fitness", 0.0)))
    return {"best": float(best["fitness"])}


@app.get("/profiles/latest")
def latest_profile() -> dict[str, Any]:
    with _PROFILE_LOCK:
        if not _LATEST_PROFILE_ID:
            raise HTTPException(404, "no cProfile results available yet")
        return _PROFILES[_LATEST_PROFILE_ID]


@app.get("/profiles/{profile_id}")
def get_profile(profile_id: str) -> dict[str, Any]:
    with _PROFILE_LOCK:
        profile = _PROFILES.get(profile_id)
        if not profile:
            raise HTTPException(404, "cProfile result not found")
        return profile


@app.post("/export/fitness-csv")
def export_fitness_csv(body: EvolutionResponse) -> Response:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["strategy", "generation", "avg_fitness", "best_fitness"])
    for run in body.runs:
        for g, (avg_f, best_f) in enumerate(
            zip(run.avg_fitness_per_gen, run.best_per_generation, strict=False)
        ):
            w.writerow([run.strategy.value, g, avg_f, best_f])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=fitness.csv"},
    )
