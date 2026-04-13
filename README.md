# RaiderEvolve — LLM-Guided Evolutionary Code Optimization

**Course:** CS-5381-D01 &nbsp;|&nbsp; **Group No. 3**  
**Members:** Lirong Chen · Jacob Ray · Zhuo Qian · MD A Rahman

---

## Table of Contents

1. [Project Description](#1-project-description)
2. [Features](#2-features)
3. [System Architecture](#3-system-architecture)
4. [Demo Video](#4-demo-video)
5. [Prerequisites](#5-prerequisites)
6. [Requirements & Dependencies](#6-requirements--dependencies)
7. [Step-by-Step Execution](#7-step-by-step-execution)
   - [7.1 Local Development (Backend)](#71-local-development-backend)
   - [7.2 Local Development (Frontend)](#72-local-development-frontend)
   - [7.3 Docker (Production)](#73-docker-production)
8. [Environment Variables](#8-environment-variables)
9. [API Endpoints](#9-api-endpoints)
10. [Task Descriptions & Data Formats](#10-task-descriptions--data-formats)
    - [10.1 Pacman Weighted-Maze Path-Finding](#101-pacman-weighted-maze-path-finding)
    - [10.2 3×3 Matrix Multiplication](#102-33-matrix-multiplication)
    - [10.3 Candidate Record Schema](#103-candidate-record-schema)
    - [10.4 Exported CSV Format](#104-exported-csv-format)
    - [10.5 Per-Member Data Reports](#105-per-member-data-reports)
11. [LLM Techniques](#11-llm-techniques)
12. [Tests](#12-tests)
13. [Optional Extensions](#13-optional-extensions)
14. [References](#14-references)

---

## 1. Project Description

**RaiderEvolve** is a closed-loop code-evolution framework inspired by OpenEvolve [10] and AlphaEvolve [11]. It treats a Python function as "DNA" and iteratively refines it across generations using a large language model (LLM) as the primary mutation operator.

Each generation executes an **OpenEvolve-style cycle**:

| Step | What happens |
|------|-------------|
| **1. Select** | Top-k / elite / diversity-aware parents are chosen from the in-memory candidate store. |
| **2. Mutate** | Three parallel mutation strategies produce child candidates: LLM-based semantic refactoring, random line-level perturbation, and template injection of proven building blocks. |
| **3. Evaluate** | Each child is executed inside a sandboxed runner with a hard 3-second timeout and scored against a task-specific fitness function. |
| **4. Store** | All candidates (with fitness, metrics, and lineage) are persisted in an in-memory deduplicated store and optionally logged to Qdrant. |
| **5. Next gen** | The ranked population seeds the next generation's context prompt. |

### Supported Tasks

| Task | Baseline | Fitness goal |
|------|----------|-------------|
| **Pacman path-finding** | DFS on a weighted mud maze | Minimise path cost + exploration penalty; expected chain DFS → BFS → UCS → A\* |
| **3×3 matrix multiplication** | Standard 3-nested-loop (27 multiplications) | Minimise scalar multiplication count; target: Laderman 1976 (23 muls) [12] |

---

## 2. Features

- **Two built-in optimisation tasks** — Pacman weighted-maze path-finding and 3×3 matrix multiplication.
- **Three evolution strategies** runnable side-by-side from the UI:
  - `single_llm` — single LLM call with a minimal prompt (one-shot baseline).
  - `random_only` — random line-level mutations only (no LLM).
  - `full` — full LLM-guided evolutionary loop (the complete system).
- **Three selection modes** — `top_k`, `elite`, `diversity`.
- **Configurable fitness weights** — sliders in the UI control `w1/w2/w3` for Pacman and `w_muls / w_adds / w_time / w_length / w_readability` for the matrix task.
- **Runtime operation counting** — matrix multiplications and additions are counted at execution time via a transparent `_TrackedNum` proxy, not by static AST analysis.
- **Sandboxed evaluation** — each candidate runs in a `threading.Thread` with a 3-second hard timeout; failures return a sentinel penalty (`-1e6`) and are excluded from average-fitness reporting.
- **Server-Sent Events (SSE) streaming** — live per-generation updates pushed to the browser without polling.
- **Deduplicating memory store** — SHA-256 keyed; previously evaluated code is never re-sent to the LLM.
- **Qdrant background persistence** — all candidates are logged to a remote Qdrant collection in a fire-and-forget daemon thread (zero impact on evolution latency).
- **Recharts visualisation** — fitness curve, average fitness, best-per-generation, plus task-specific metrics (path cost, steps, runtime, multiplication count).
- **Docker Compose one-command deployment** for reproducible grader evaluation.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Browser / User                    │
│   - Select task, paste code, set parameters         │
│   - Watch live charts (SSE stream)                  │
│   - Download fitness CSV                            │
└─────────────────────────┬───────────────────────────┘
                          │  HTTP + SSE (port 3000)
┌─────────────────────────▼───────────────────────────┐
│          Next.js Frontend  (port 3000)              │
│  web/app/page.tsx  ·  Recharts  ·  AbortController  │
│  - Strategy comparison panel                        │
│  - Configurable fitness weight sliders              │
│  - Algorithm / Problem Description textarea         │
└─────────────────────────┬───────────────────────────┘
                          │  /api/backend/*  (proxy)
┌─────────────────────────▼───────────────────────────┐
│           FastAPI Backend  (port 8000)              │
│  main.py  (uvicorn)                                 │
│  POST /evolve/stream   ← SSE streaming run          │
│  POST /evolve/sync     ← synchronous run            │
│  POST /evolve          ← async job                  │
│  GET  /evolve/{job_id} ← poll job                   │
│  POST /analytics/best-up-to                         │
│  POST /export/fitness-csv                           │
│  GET  /health                                       │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│              Evolution Engine                       │
│                                                     │
│  controller.py  ── generation loop                  │
│    ├─ context_builder.py  ── prompt assembly        │
│    ├─ llm_client.py       ── OpenAI-compat API      │
│    │                          (retry + backoff)     │
│    ├─ pacman_env.py       ── Pacman evaluator        │
│    ├─ matrix_task.py      ── matrix evaluator       │
│    ├─ memory_store.py     ── dedup candidate store  │
│    ├─ selection.py        ── top-k / elite /        │
│    │                          diversity             │
│    ├─ random_mutation.py  ── random perturbation    │
│    └─ template_mutation.py── template injection     │
└──────────────┬────────────────────┬─────────────────┘
               │  HTTPS             │  HTTP (async)
┌──────────────▼──────┐   ┌─────────▼──────────────── ┐
│  OpenAI-compat LLM  │   │  Qdrant Vector DB          │
│  (gpt-4o-mini /     │   │  (background audit log)    │
│   xAI / Groq)       │   │  43.130.56.234:6333        │
└─────────────────────┘   └────────────────────────────┘
```

### Source File Index

| Path | Role |
|------|------|
| `main.py` | FastAPI app, job registry, SSE queue |
| `evolve/controller.py` | Generation loop, orchestration |
| `evolve/context_builder.py` | LLM prompt assembly (few-shot, history, hints) |
| `evolve/llm_client.py` | LLM API wrapper with exponential-backoff retry |
| `evolve/pacman_env.py` | Sandboxed Pacman fitness evaluator |
| `evolve/pacman.py` | Berkeley CS188 Pacman game engine [12] |
| `evolve/matrix_task.py` | Matrix task evaluator + `_TrackedNum` counter |
| `evolve/memory_store.py` | In-memory candidate store with dedup cache |
| `evolve/qdrant_store.py` | Fire-and-forget Qdrant persistence layer |
| `evolve/selection.py` | Selection strategy implementations |
| `evolve/random_mutation.py` | Random line-level mutation |
| `evolve/template_mutation.py` | Template injection (e.g., Laderman seed) |
| `evolve/models.py` | Pydantic request / response models |
| `evolve/settings.py` | `pydantic-settings` config from `.env` |
| `web/app/page.tsx` | Next.js UI (React, Recharts, SSE client) |
| `data/` | Per-member Round 2 experiment reports |

---

## 4. Demo Video

> Replace the placeholder URL with the final submission link before uploading.

**Link:** `https://<insert-video-url-here>`

The screencast covers:

1. Starting backend and frontend locally.
2. Selecting the **Pacman** task, leaving the DFS baseline, choosing all three strategies.
3. Running 4 generations and watching the live fitness curve update via SSE.
4. Switching to the **Matrix** task, adjusting the multiplication-weight slider, running evolution.
5. Comparing `single_llm` vs `random_only` vs `full` fitness curves.
6. Downloading the fitness CSV.

---

## 5. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 or newer | Backend runtime |
| Node.js | 20 LTS or newer | Frontend only |
| npm | 9+ | Bundled with Node 20 |
| Docker & Compose v2 | latest | Optional — for one-command deployment |
| LLM API key | — | Any OpenAI-compatible provider (OpenAI, xAI, Groq). Without a key the backend runs a deterministic mock. |
| RAM | 8 GB minimum | 16 GB recommended for concurrent strategies |
| CPU | Any modern multi-core | No GPU required |

---

## 6. Requirements & Dependencies

### Python (backend)

`requirements.txt`:

```
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.6.0
pydantic-settings>=2.2.0
httpx>=0.27.0
numpy>=1.26.0
qdrant-client>=1.9.0
pytest>=8.0.0
```

Install:

```bash
pip install -r requirements.txt
```

### Node.js (frontend)

`web/package.json` key dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `next` | ^14.2 | React framework + API proxy |
| `react` | ^18.3 | UI library |
| `recharts` | ^2.13 | Fitness / metrics charts |
| `typescript` | ^5 | Type-safe frontend |

Install:

```bash
cd web && npm install
```

---

## 7. Step-by-Step Execution

### 7.1 Local Development (Backend)

Run all commands from the **repository root**:

```bash
# 1. Create and activate a virtual environment (first time only)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Configure your API key (first time only)
cp .env.example .env
#    Open .env and set OPENAI_API_KEY=sk-...

# 4. Start the backend server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

> **Important:** `uvicorn` must be started from the project root so that `.env` is automatically discovered by `pydantic-settings`.

Verify the backend is running:

```bash
curl http://127.0.0.1:8000/health
# Expected: {"status":"ok","service":"evolve"}
```

---

### 7.2 Local Development (Frontend)

Open a **second terminal**:

```bash
# From the repository root
cd web
npm install        # first time only
npm run dev        # starts hot-reload dev server on port 3000
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

In development mode, the Next.js proxy forwards `/api/backend/*` to the FastAPI backend with an extended timeout that covers SSE streaming runs.

**Tip — bypass proxy if you see `ECONNRESET` / `socket hang up`:**

```bash
# web/.env.local
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Then restart `npm run dev`. The browser will call the backend directly (CORS is already open on all origins).

> The `[DEP0060] util._extend` deprecation warning comes from a third-party dependency and is harmless. Suppress it with: `NODE_OPTIONS=--no-deprecation npm run dev`

---

### 7.3 Docker (Production)

#### Prerequisites

- Docker Engine and Docker Compose v2 (`docker compose` command).
- A `.env` file in the project root containing a valid API key.

#### One-Command Start (recommended)

```bash
# Build images and start all services in the background
docker compose up --build -d
```

| Service | URL |
|---------|-----|
| Frontend | [http://localhost:3000](http://localhost:3000) |
| Backend  | [http://localhost:8000](http://localhost:8000) |

View live logs:

```bash
docker compose logs -f            # all services
docker compose logs -f backend    # backend only
docker compose logs -f web        # frontend only
```

Stop everything:

```bash
docker compose down
```

#### Build and Run Services Individually

**Backend only:**

```bash
docker build -t raiderevolve-backend .
docker run --rm -p 8000:8000 --env-file .env raiderevolve-backend
```

**Frontend only:**

```bash
docker build -t raiderevolve-web ./web \
  --build-arg BACKEND_URL=http://host.docker.internal:8000
docker run --rm -p 3000:3000 \
  -e BACKEND_URL=http://host.docker.internal:8000 \
  raiderevolve-web
```

> On Linux, add `--add-host=host.docker.internal:host-gateway` to the `docker run` command because Docker Desktop's `host.docker.internal` alias is not available natively.

---

## 8. Environment Variables

Create a `.env` file in the **repository root** (copy from `.env.example`):

```dotenv
# ── LLM API ────────────────────────────────────────────────────────────
# Either name is accepted; OPENAI_API_KEY takes precedence.
OPENAI_API_KEY=sk-...
# LLM_API_KEY=sk-...

# Optional: OpenAI-compatible third-party endpoint
# OPENAI_BASE_URL=https://api.x.ai/v1               # xAI Grok
# OPENAI_BASE_URL=https://api.groq.com/openai/v1    # Groq

# Model to use (default: gpt-4o-mini)
# LLM_MODEL=grok-3-mini
# LLM_MODEL=llama3-8b-8192

# ── Qdrant (optional background persistence) ───────────────────────────
QDRANT_URL=http://43.130.56.234:6333
QDRANT_API_KEY=<your-qdrant-api-key>
```

If `OPENAI_API_KEY` is absent or set to `YOUR_API_KEY`, the backend falls back to a **deterministic mock** that returns the parent code unchanged. The system remains fully runnable for UI testing without any real API key.

---

## 9. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check; returns `{"status":"ok"}` |
| `POST` | `/evolve/stream` | **SSE streaming run** — recommended; sends `data: {...}` events per generation in real time |
| `POST` | `/evolve/sync` | Synchronous run — blocks until all strategies finish, returns `EvolutionResponse` |
| `POST` | `/evolve` | Async job — returns `{"job_id":"..."}` immediately; poll for results |
| `GET`  | `/evolve/{job_id}` | Poll async job; returns `JobStatus` with `status` and optional `result` |
| `POST` | `/analytics/best-up-to` | Best fitness achieved up to a given generation (for Figure-2 UI behaviour) |
| `POST` | `/export/fitness-csv` | Download fitness curves as a CSV file |

All `POST` endpoints accept a JSON body matching the `EvolutionRequest` schema defined in `evolve/models.py`.

---

## 10. Task Descriptions & Data Formats

RaiderEvolve is a **code-evolution** system, not a machine-learning training system. There is no training dataset. All data is generated at runtime by the evaluator.

### 10.1 Pacman Weighted-Maze Path-Finding

**Goal:** Evolve a `search(start, goal, grid)` function that returns a list of `(row, col)` tuples representing the lowest-cost path from `start` to `goal`.

**Input grid encoding (static, defined in `evolve/pacman.py`):**

| Character | Meaning | Step cost |
|-----------|---------|-----------|
| `%` | Wall — impassable | — |
| ` ` | Open passage | 1 |
| `M` | Mud cell | 5 |

Two built-in maze layouts are used for evaluation: `mudMaze` (small) and `largeMudMaze` (large). The large maze contains an open room below a narrow corridor — a design that distinguishes BFS/UCS (which flood into the room) from A\* (which stays in the corridor guided by the Manhattan heuristic).

**Fitness formula:**

```
fitness = w1 · score + w2 · survival − w3 · steps
```

| Component | Description |
|-----------|-------------|
| `score` | `1000 − total_path_cost`; mud cells cost 5, open cells cost 1 |
| `survival` | Fraction of runs in which the agent successfully reached the goal |
| `steps` | Average number of path steps (cells traversed) |

Default weights: `w1 = 0.5, w2 = 0.3, w3 = 0.2` (automatically normalised to sum to 1).

**Per-evaluation metrics recorded:**

| Field | Type | Description |
|-------|------|-------------|
| `avg_score` | float | Average `1000 - path_cost` across runs |
| `avg_cost` | float | Average total path cost |
| `avg_steps` | float | Average path length (number of cells) |
| `success_rate` | float | Fraction of runs that reached the goal |
| `eval_time_ms` | float | Wall-clock time for the entire evaluation call (ms) |
| `layouts_used` | list | Names of mazes used in this evaluation |

**Expected evolution chain:**

```
DFS (≈895) → BFS (≈928) → UCS (≈952) → A* (≈960)
```

---

### 10.2 3×3 Matrix Multiplication

**Goal:** Evolve a `matmul(a, b)` function that computes `C = A × B` for 3×3 lists-of-lists, using as few scalar multiplications as possible.

**Baseline:** standard 3-nested-loop algorithm — 27 multiplications, 18 additions.

**Operation counting:** scalar `*` and `+`/`-` are counted at **runtime** via a `_TrackedNum` proxy class that wraps every matrix element. This correctly attributes loop-based operations (e.g., 27 `*` for the baseline) instead of counting AST nodes.

**Correctness check:** four fixed test pairs `(A, B)` with known results. A candidate returning any incorrect value receives `fitness = -10.0`.

**Fitness formula (configurable via UI sliders):**

```
fitness = 1.0
        + (27 - actual_muls) / 27 * 10 * w_muls
        + (18 - actual_adds) / 18 *  3 * w_adds
        + time_savings * 5              * w_time
        + length_savings                * w_length
        + readability_score             * w_readability
```

Default weight values: `w_muls = 1.0`, all others `= 0.2`.

**Known milestones:**

| Algorithm | Multiplications | Approximate fitness |
|-----------|----------------|---------------------|
| Standard 3-loop (baseline) | 27 | 1.0 |
| Laderman 1976 [12] | 23 | ≈ 2.35 |
| Makarov 1970 | 25 | ≈ 1.74 |
| Theoretical lower bound | ≥ 19 | — |

---

### 10.3 Candidate Record Schema

Every evolved function is stored as a `CandidateRecord` (`evolve/models.py`):

```json
{
  "id": "uuid4-string",
  "generation": 2,
  "code": "def search(start, goal, grid): ...",
  "fitness": 951.7,
  "parents": ["parent-uuid-1"],
  "strategy_tag": "llm",
  "mutation_notes": "Switched from BFS to UCS with a priority queue",
  "metrics": {
    "avg_score": 951.7,
    "avg_cost": 48.3,
    "avg_steps": 31,
    "success_rate": 1.0,
    "eval_time_ms": 42.1
  }
}
```

---

### 10.4 Exported CSV Format

`POST /export/fitness-csv` returns a CSV with the following columns:

```
strategy, generation, avg_fitness, best_fitness
single_llm, 0, 895.2, 895.2
single_llm, 1, 928.4, 928.4
full, 0, 895.2, 895.2
full, 1, 934.1, 951.7
...
```

---

### 10.5 Per-Member Data Reports

Round 2 data reports are stored in `data/{member}/`:

| File | Format | Description |
|------|--------|-------------|
| `{member}_data.csv` | CSV | Experiment results for one parameter configuration |
| `{member}_data.docx` | Word | Written analysis |

All CSV files share a common column set:

```
strategy, generation, best_fitness, avg_fitness,
best_avg_score, best_avg_cost, best_avg_steps,
best_avg_cells_accessed, best_eval_time_ms, best_success_rate
```

---

## 11. LLM Techniques

| Technique | Where used | Detail |
|-----------|------------|--------|
| **Few-shot prompting** | `context_builder.py` | Top-N ranked historical solutions (with fitness scores) shown to the LLM before asking for improvement |
| **Negative examples** | `context_builder.py` | Worst-performing candidates are shown with a "do not repeat" label to steer the LLM away from dead ends |
| **In-context fitness feedback** | `context_builder.py` | Exact numeric fitness and per-component breakdown (muls, adds, cost, steps) included in every prompt |
| **Chain-of-improvement guidance** | `context_builder.py` | Task-specific instructions name the expected improvement trajectory (DFS→A\*, 27→23 muls) and explain the scoring formula |
| **Role prompting** | `llm_client.py` | System prompt: *"You are an expert code optimizer. Output only Python. No markdown."* |
| **Output format constraint** | `llm_client.py` | Max-token caps (pacman: 1400, matrix: 700) and explicit instruction to return raw code only |
| **Differential prompting** | `controller.py` | `single_llm` strategy receives a stripped, hint-free prompt; `full` strategy always receives the complete guidance — enabling a fair ablation comparison |
| **Exponential-backoff retry** | `llm_client.py` | Up to 7 retries with `Retry-After` header parsing for 429 / 5xx responses |

---

## 12. Tests

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Run from the project root
pytest tests -q
```

The test suite currently covers the `/health` endpoint. The `pytest.ini` sets `pythonpath = .` so imports resolve correctly without installing the package.

---

## 13. Optional Extensions

- **Qdrant semantic search:** `qdrant_store.py` currently writes a 1-D dummy vector (fitness-based). To enable embedding-based similarity retrieval, replace the dummy vector with a real code embedding (e.g., `text-embedding-3-small`) and implement a `semantic_search` method in `MemoryStore`.
- **Additional LLM providers:** Set `OPENAI_BASE_URL` to any OpenAI-compatible endpoint (Mistral, Together, Ollama, etc.).
- **Custom tasks:** Implement a fitness function matching the `(code: str) -> (float, dict)` signature and register it alongside `pacman_fitness` / `matrix_correctness_and_ops`.
- **FAISS local vector index:** Drop-in alternative to Qdrant for fully offline deployments.

---

## 14. References

[10] Zhang, S., et al. **"OpenEvolve: An Open-Source Implementation of DeepMind's AlphaEvolve Framework."** GitHub, 2025. [https://github.com/codelion/openevolve](https://github.com/codelion/openevolve)

[11] Novikov, A., et al. **"AlphaEvolve: A Gemini-based coding agent for designing advanced algorithms."** DeepMind Technical Report, 2025. [https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

[12] Laderman, J. D. **"A noncommutative algorithm of order 23 for multiplying 6×6 matrices using 354 multiplications."** *Journal of the ACM*, 23(1):148–156, 1976. — Pacman game engine adapted from the Berkeley CS188 AI course project: [https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/search.zip](https://inst.eecs.berkeley.edu/~cs188/sp24/assets/projects/search.zip)
