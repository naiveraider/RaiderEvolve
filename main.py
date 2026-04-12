from __future__ import annotations

import asyncio
import csv
import io
import json
import threading
import uuid
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from evolve.controller import PSEUDOCODE_OUTLINE, run_evolution_run
from evolve.llm_client import LLMRequestError
from evolve.models import EvolutionRequest, EvolutionResponse, EvolutionStrategy, JobStatus

app = FastAPI(title="Evolve System", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_JOBS: dict[str, JobStatus] = {}


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
        runs = []
        total = len(strategies)
        for i, strat in enumerate(strategies):
            st.progress = (i / total) * 100.0
            runs.append(run_evolution_run(req, strat))
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
        st.message = "complete"
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
        runs = [run_evolution_run(req, s) for s in strategies]
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
        results = []
        try:
            for strat in strategies:
                def _cb(data: dict, _s=strat) -> None:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", **data})

                result = run_evolution_run(req, strat, progress_cb=_cb)
                results.append(result)
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
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "done", "result": final.model_dump()},
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
