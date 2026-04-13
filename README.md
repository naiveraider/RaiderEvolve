# Evolve System - RaiderEvolve

**Course:** CS-5381-D01, **Group No.3** (Lirong Chen, Jacob Ray, Zhuo Qian, MD A Rahman)

LLM-guided evolutionary optimization (OpenEvolve-style loop) for a Pacman
path-finding agent and a 3×3 matrix-multiplication task. FastAPI backend,
Next.js frontend, Docker Compose packaging.

---

## Project Description

RaiderEvolve is a closed-loop code-evolution framework that treats a Python
function as "DNA" and iteratively refines it across generations. Each
generation runs the OpenEvolve-style cycle:

1. **Select** top-k / elite / diversity-aware parents from the memory store.
2. **Mutate** them via a hybrid strategy. An LLM performs semantic
   refactoring of the candidate code, random perturbation preserves
   diversity, and template rewrites insert proven building blocks.
3. **Evaluate** each child against a task-specific fitness function in a
   sandboxed runner and record metrics (fitness, path cost, steps,
   runtime, cells explored).
4. **Store** the best candidates in a deduplicated memory for the next
   generation.

For the Pacman task the expected evolution chain is
**DFS -> BFS -> UCS -> A\*** on weighted mud mazes, and fitness converges
from about 895 to about 961 within 4 to 5 generations.

## Features

- Two built-in tasks: **Pacman weighted-maze path-finding** (`choose_action`
  style, DFS baseline) and **3×3 matrix multiplication** (correctness +
  operation count).
- Three selection modes: `top_k`, `elite`, `diversity`.
- Three comparison strategies runnable side-by-side from the UI:
  `single_llm` (one-shot), `random_only` (no LLM), and `full`
  (LLM-guided evolution).
- Custom fitness weights (`w1`, `w2`, `w3`) for ablation studies.
- Sandboxed per-candidate evaluation with a hard 3-second timeout and a
  sentinel penalty for crashing candidates.
- Server-Sent Events streaming for live per-generation updates in the UI.
- Deduplicating memory store (code-hash keyed) to avoid redundant LLM
  calls on revisited candidates.
- Docker Compose one-command deployment for graders.

## System Architecture

```
                  ┌──────────────────────────────────────┐
                  │         Browser / User               │
                  └──────────────────┬───────────────────┘
                                     │  HTTP + SSE
                  ┌──────────────────▼───────────────────┐
                  │   Next.js Frontend  (port 3000)      │
                  │   page.tsx, Recharts, AbortCtrl      │
                  └──────────────────┬───────────────────┘
                                     │  /api/backend/*  (proxy)
                  ┌──────────────────▼───────────────────┐
                  │   FastAPI Backend   (port 8000)      │
                  │   main.py  (uvicorn)                 │
                  │   /evolve/stream  /evolve/sync       │
                  │   /analytics/best-up-to              │
                  │   /export/fitness-csv  /health       │
                  └──────────────────┬───────────────────┘
                                     │
                  ┌──────────────────▼───────────────────┐
                  │          Evolution Engine            │
                  │  1 Controller     -> generation loop │
                  │  2 LLM Client     -> OpenAI-compat   │
                  │  3 Evaluator      -> pacman_env /    │
                  │                      matrix_task     │
                  │  4 Memory Store   -> dedup cache     │
                  │  5 Mutations      -> LLM / random /  │
                  │                      template        │
                  └──────────────────┬───────────────────┘
                                     │  HTTPS
                                     ▼
                             OpenAI-compatible LLM
                         (OpenAI / xAI Grok / Groq)
```

Source layout:

| Path | Role |
|------|------|
| `main.py` | FastAPI entrypoint, job registry, SSE streaming |
| `evolve/controller.py` | Generation loop, orchestration |
| `evolve/llm_client.py` | LLM API wrapper with retry + backoff |
| `evolve/pacman_env.py` | Pacman fitness evaluator (sandboxed) |
| `evolve/matrix_task.py` | Matrix-multiply task + scorer |
| `evolve/memory_store.py` | Candidate memory with dedup cache |
| `evolve/random_mutation.py` | Random / template mutation |
| `evolve/selection.py` | Selection strategies |
| `web/` | Next.js frontend (React, Recharts) |
| `data/` | Per-member Round 2 data reports (`{name}_data.csv` / `.docx`) |

## Demo Video

> _Video link placeholder. Replace with the final URL before submission._
>
> `https://<insert-video-url-here>`

A 2 to 3 minute screencast walks through selecting the Pacman task, choosing
the three comparison strategies, running a 4-generation evolution, and
inspecting the fitness / step / runtime curves in the live dashboard.

## Prerequisites

- **OS:** Windows 10/11, macOS, or Linux (Ubuntu 22.04+)
- **Python:** 3.11 or newer
- **Node.js:** 20 LTS or newer (frontend only)
- **Docker & Docker Compose v2** (optional, for the one-command setup)
- **LLM API key** from any OpenAI-compatible provider. Set
  `OPENAI_API_KEY` in `.env`. If the key is missing the backend falls
  back to a deterministic mock and does not make real API calls, so the
  prototype is still runnable without credentials.
- **Hardware:** 8 GB RAM minimum, 16 GB recommended when running
  concurrent evolution strategies. A modern multi-core CPU is sufficient;
  no GPU required.

## Data & Data Formats

RaiderEvolve does not train a model, so there is no training dataset. All
"data" is generated at run time by the evaluator and stored in-memory and
as JSON artifacts:

- **Task inputs (static):** the Pacman evaluator ships with two built-in
  weighted mud mazes (`mudMaze`, `largeMudMaze`) defined in
  `evolve/pacman.py`. `'%'` = wall, `' '` = open (cost 1), `'M'` = mud
  (cost 5).
- **Candidate records (in-memory):** each evolved function is stored as
  a `CandidateRecord` (`evolve/models.py`) with fields `id`, `generation`,
  `code`, `fitness`, `parents`, `strategy_tag`, `mutation_notes`,
  `metrics`.
- **Per-run metrics:** `avg_score`, `avg_cost`, `avg_steps`,
  `avg_cells_accessed`, `success_rate`, `eval_time_ms`, `layouts_used`.
- **Round 2 per-member reports:** `data/{member}/{member}_data.csv` and
  `data/{member}/{member}_data.docx`. CSV columns vary per member since
  each member tests one parameter configuration; all share a common set
  of metrics: `strategy, generation, best_fitness, avg_fitness,
  best_avg_score, best_avg_cost, best_avg_steps,
  best_avg_cells_accessed, best_eval_time_ms, best_success_rate`.
- **Exported CSV from the API:** `POST /export/fitness-csv` returns a CSV
  with columns `strategy, generation, avg_fitness, best_fitness`.

## Table of Contents

- [Local Development](#local-development)
- [Docker (Production)](#docker-production)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Tests](#tests)
- [Optional Extensions](#optional-extensions)

---

## Local Development

### 1. Backend (FastAPI + uvicorn)

Run all commands from the **repository root**:

```bash
# Create virtual environment (first time only)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key (first time only)
cp .env.example .env             # then edit .env and fill in your key

# Start the backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

> uvicorn must be started from the project root so that `.env` is discovered automatically.

Verify the backend is running:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","service":"evolve"}
```

---

### 2. Frontend (Next.js)

Open a second terminal:

```bash
cd web
npm install        # first time only
npm run dev        # hot-reload dev server
```

Open [http://localhost:3000](http://localhost:3000).

In dev mode the frontend proxies `/api/backend/*` to the backend with an extended timeout, which is long enough to handle SSE streaming runs.

If you still see **`ECONNRESET` / `socket hang up`** errors, add the following to `web/.env.local` so the browser calls the backend directly (CORS is already open):

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Then restart `npm run dev`.

> The `util._extend` deprecation warning printed by Node comes from a third-party dependency and is harmless. To suppress it: `NODE_OPTIONS=--no-deprecation npm run dev`

---

## Docker (Production)

### Prerequisites

- Docker and Docker Compose v2 (`docker compose` command)
- A `.env` file in the project root with a valid API key

### Start Everything (recommended)

```bash
# Build images and start all services in the background
docker compose up --build -d
```

| Service | URL |
|---------|-----|
| Frontend | [http://localhost:3000](http://localhost:3000) |
| Backend  | [http://localhost:8000](http://localhost:8000) |

View logs:

```bash
docker compose logs -f            # all services
docker compose logs -f backend    # backend only
docker compose logs -f web        # frontend only
```

Stop services:

```bash
docker compose down
```

---

### Build / Run Services Individually

**Backend only:**

```bash
docker build -t raiderevolve-backend .
docker run --rm -p 8000:8000 \
  --env-file .env \
  raiderevolve-backend
```

**Frontend only:**

```bash
docker build -t raiderevolve-web ./web \
  --build-arg BACKEND_URL=http://host.docker.internal:8000
docker run --rm -p 3000:3000 \
  -e BACKEND_URL=http://host.docker.internal:8000 \
  raiderevolve-web
```

> `host.docker.internal` lets the container reach a backend process running on the host machine. Supported on macOS and Windows Docker Desktop. On Linux add `--add-host=host.docker.internal:host-gateway` to the `docker run` command.

---

## Environment Variables

Create a `.env` file in the project root (use `.env.example` as a template):

```dotenv
# LLM API key — either name works
OPENAI_API_KEY=sk-...
# LLM_API_KEY=sk-...

# Optional: point to an OpenAI-compatible third-party endpoint
# OPENAI_BASE_URL=https://api.x.ai/v1              # xAI Grok
# OPENAI_BASE_URL=https://api.groq.com/openai/v1   # Groq

# Model to use (default: gpt-4o-mini)
# LLM_MODEL=grok-3-mini
# LLM_MODEL=llama3-8b-8192
```

If the key is missing or set to `YOUR_API_KEY`, the backend falls back to a **deterministic mock** and makes no real LLM calls.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/evolve/stream` | **SSE streaming run** — recommended; streams per-generation progress in real time |
| `POST` | `/evolve/sync` | Synchronous run — waits for all strategies to finish, then returns |
| `POST` | `/evolve` | Async job — returns a `job_id` immediately |
| `GET`  | `/evolve/{job_id}` | Poll async job status |
| `POST` | `/analytics/best-up-to` | Best fitness achieved up to a given generation |
| `POST` | `/export/fitness-csv` | Download fitness curves as CSV |

---

## Tests

```bash
# Activate the virtual environment first, then run from the project root
pytest tests -q
```

---

## Optional Extensions

- **Vector memory (Qdrant / FAISS):** `MemoryStore` currently runs in-process with code-hash deduplication. You can plug in Qdrant via the `settings.qdrant_url` hook point to enable embedding-based similarity search across the candidate history.
